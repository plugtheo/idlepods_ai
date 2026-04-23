<#
.SYNOPSIS
    Bootstrap an isolated .venv for every service in the monorepo.

.DESCRIPTION
    For each service directory (context, experience, gateway, inference,
    orchestration, training, shared) this script:
      1. Creates a Python virtual environment at <service>/.venv
      2. Installs the service's runtime dependencies (requirements.txt)
      3. Installs the service's development/test dependencies (requirements-dev.txt)

    Re-running the script is safe — existing venvs are skipped unless -Force
    is passed.

.PARAMETER Force
    Destroy and recreate any existing .venv directories.

.PARAMETER Service
    Limit setup to a single service (e.g. -Service context).
    Defaults to all services.

.EXAMPLE
    # Bootstrap everything
    .\scripts\setup_envs.ps1

    # Rebuild only the context venv from scratch
    .\scripts\setup_envs.ps1 -Service context -Force

.NOTES
    Run from the project root.  Requires Python 3.11+ on your PATH.
#>

[CmdletBinding()]
param(
    [switch]$Force,
    [ValidateSet("context", "experience", "gateway", "inference", "orchestration", "training", "shared")]
    [string]$Service
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helpers ──────────────────────────────────────────────────────────────────

function Write-Step {
    param([string]$Message)
    Write-Host "  $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "  [ok] $Message" -ForegroundColor Green
}

function Write-Skip {
    param([string]$Message)
    Write-Host "  [skip] $Message" -ForegroundColor DarkGray
}

function Write-Fail {
    param([string]$Message)
    Write-Host "  [fail] $Message" -ForegroundColor Red
}

# ── Resolve project root & Python interpreter ─────────────────────────────────

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
$Python = if ($pythonCmd) { $pythonCmd.Source } else { $null }
if (-not $Python) {
    Write-Fail "No 'python' found on PATH. Install Python 3.11+ and try again."
    exit 1
}

$PythonVersion = & $Python --version 2>&1
Write-Host ""
Write-Host "Python : $PythonVersion" -ForegroundColor White
Write-Host "Root   : $ProjectRoot"   -ForegroundColor White
Write-Host ""

# ── Service list ──────────────────────────────────────────────────────────────

$AllServices = @("context", "experience", "gateway", "inference", "orchestration", "training", "shared")

$Targets = if ($Service) { @($Service) } else { $AllServices }

# ── Per-service setup ─────────────────────────────────────────────────────────

$Failed = @()

foreach ($svc in $Targets) {
    $svcDir   = Join-Path $ProjectRoot $svc
    $venvDir  = Join-Path $svcDir ".venv"
    $venvPy   = Join-Path $venvDir "Scripts\python.exe"

    $separator = "-" * (62 - $svc.Length)
    Write-Host "-- $svc $separator" -ForegroundColor White

    # ── Create venv ───────────────────────────────────────────────────────────
    if (Test-Path $venvDir) {
        if ($Force) {
            Write-Step "Removing existing venv..."
            Remove-Item $venvDir -Recurse -Force
        } else {
            Write-Skip "venv already exists (use -Force to recreate)"
        }
    }

    if (-not (Test-Path $venvDir)) {
        Write-Step "Creating venv..."
        & $Python -m venv $venvDir
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Failed to create venv for $svc"
            $Failed += $svc
            continue
        }
        Write-Ok "venv created"
    }

    # ── Upgrade pip silently ──────────────────────────────────────────────────
    Write-Step "Upgrading pip..."
    & $venvPy -m pip install --quiet --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "pip upgrade failed for $svc"
        $Failed += $svc
        continue
    }

    # ── Install runtime deps ──────────────────────────────────────────────────
    $runtimeReqs = Join-Path $svcDir "requirements.txt"
    if (Test-Path $runtimeReqs) {
        Write-Step "Installing runtime deps (requirements.txt)..."
        & $venvPy -m pip install --quiet -r $runtimeReqs
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Runtime install failed for $svc"
            $Failed += $svc
            continue
        }
        Write-Ok "runtime deps installed"
    } else {
        Write-Skip "No requirements.txt"
    }

    # ── Install dev/test deps ─────────────────────────────────────────────────
    $devReqs = Join-Path $svcDir "requirements-dev.txt"
    if (Test-Path $devReqs) {
        Write-Step "Installing dev deps (requirements-dev.txt)..."
        & $venvPy -m pip install --quiet -r $devReqs
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Dev install failed for $svc"
            $Failed += $svc
            continue
        }
        Write-Ok "dev deps installed"
    } else {
        Write-Skip "No requirements-dev.txt"
    }

    Write-Host ""
}

Pop-Location

# ── Summary ───────────────────────────────────────────────────────────────────

if ($Failed.Count -gt 0) {
    Write-Host "Setup failed for: $($Failed -join ', ')" -ForegroundColor Red
    exit 1
} else {
    Write-Host "All environments ready." -ForegroundColor Green
    Write-Host ""
    Write-Host "Run tests for a service:" -ForegroundColor White
    Write-Host "  cd context ; .\.venv\Scripts\pytest tests\" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "Or from the project root:" -ForegroundColor White
    Write-Host "  context\.venv\Scripts\pytest context\tests\" -ForegroundColor DarkGray
}
