#!/usr/bin/env pwsh
<#
.SYNOPSIS
Test debugging_lora and review_lora for spaceless output bug.
Uses direct HTTP calls to the inference service (8010) to isolate adapter quality.

.DESCRIPTION
If adapters are broken, output will be compressed with no spaces:
  BROKEN: "defFibonacciChallengeTheFunction..."
  FIXED:  "def fibonacci(n, memo={}):\n    # Challenge the function..."

.USAGE
  .\test_broken_adapters.ps1
#>

param(
    [string]$InferenceUrl = "http://localhost:8010",
    [int]$MaxTokens = 150,
    [float]$Temperature = 0.7
)

$DebugPreference = "SilentlyContinue"

# ============================================================================
# TEST DATA
# ============================================================================

$TestCases = @(
    @{
        Name = "debugging_lora: Fix function bug"
        Adapter = "debugging_lora"
        Prompt = "Challenge: This function has a bug. Find and fix it.`n`nCode:`ndef add_numbers(a, b):`n    return a+b`n`nDebug this code and provide the corrected version with explanation."
    },
    @{
        Name = "debugging_lora: Algorithm efficiency"
        Adapter = "debugging_lora"
        Prompt = "Challenge: Analyze this code for bugs.`n`ndef fibonacci(n, memo=None):`n    if memo is None:`n        memo = {}`n    if n in memo:`n        return memo[n]`n    if n <= 1:`n        return n`n    memo[n] = fibonacci(n-1,memo)+fibonacci(n-2,memo)`n    return memo[n]`n`nIdentify issues."
    },
    @{
        Name = "review_lora: Code review"
        Adapter = "review_lora"
        Prompt = "Review this code:`n`ndef calculate(x, y, z):`n    total = x + y + z`n    percentage = (total / 300) * 100`n    if percentage >= 90:`n        grade = 'A'`n    else:`n        grade = 'F'`n    return {'total': total, 'grade': grade}"
    },
    @{
        Name = "review_lora: Architecture review"
        Adapter = "review_lora"
        Prompt = "Review this design:`n`nclass UserService:`n    def create_user(self, name, email, password):`n        return self.db.insert('users', {'name': name})`n`nProvide feedback."
    }
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

function Invoke-InferenceAPI {
    param(
        [string]$Prompt,
        [string]$Adapter,
        [int]$MaxTokens = 150,
        [float]$Temperature = 0.7
    )
    
    $Url = "$InferenceUrl/v1/generate"
    
    # Determine model family based on adapter
    $ModelFamily = if ($Adapter -like "*mistral*") { "mistral" } else { "deepseek" }
    
    # Extract role from adapter name (e.g., debugging_lora -> debugging)
    $Role = $Adapter -replace "_lora$", ""
    
    $Body = @{
        model_family = $ModelFamily
        role = $Role
        messages = @(
            @{
                role = "user"
                content = $Prompt
            }
        )
        adapter_name = $Adapter
        max_tokens = $MaxTokens
        temperature = $Temperature
        top_p = 0.95
    } | ConvertTo-Json
    
    try {
        Write-Host "[REQUEST] POST $Url" -ForegroundColor Cyan
        Write-Host "[MODEL] $ModelFamily  [ROLE] $Role  [ADAPTER] $Adapter" -ForegroundColor Cyan
        
        $Response = Invoke-RestMethod -Uri $Url -Method Post -Body $Body -ContentType "application/json" -ErrorAction Stop
        
        return @{
            Success = $true
            Output = $Response.content
            FullResponse = $Response
        }
    }
    catch {
        return @{
            Success = $false
            Error = $_.Exception.Message
            FullResponse = $_
        }
    }
}

function Test-SpacelessOutput {
    param([string]$Output)
    
    if (-not $Output) { return $false }
    
    $HasCamelCaseConcat = $Output -match '(def|class|return|if|else|for|while|import)[a-z][a-z]+[A-Z]'
    
    $Words = $Output -split '\s+' | Where-Object { $_.Length -gt 0 }
    $LongestWord = if ($Words.Count -gt 0) { ($Words | Measure-Object -Property Length -Maximum).Maximum } else { 0 }
    
    $FirstChunk = $Output.Substring(0, [Math]::Min(150, $Output.Length))
    $HasNoLineBreaks = $FirstChunk -notmatch '\n'
    
    return $HasCamelCaseConcat -or ($LongestWord -gt 35) -or ($HasNoLineBreaks -and $Output.Length -gt 100)
}

function Show-Results {
    param(
        [string]$TestName,
        [hashtable]$Result,
        [string]$Adapter
    )
    
    Write-Host "`n$('='*80)" -ForegroundColor Yellow
    Write-Host "TEST: $TestName" -ForegroundColor Yellow
    Write-Host "ADAPTER: $Adapter" -ForegroundColor Yellow
    Write-Host $('='*80) -ForegroundColor Yellow
    
    if (-not $Result.Success) {
        Write-Host "FAILED: $($Result.Error)" -ForegroundColor Red
        return
    }
    
    $Output = $Result.Output
    if (-not $Output) {
        Write-Host "NO OUTPUT GENERATED" -ForegroundColor Red
        return
    }
    
    $IsBroken = Test-SpacelessOutput -Output $Output
    
    if ($IsBroken) {
        Write-Host "LIKELY BROKEN - Spaceless output detected" -ForegroundColor Red
    }
    else {
        Write-Host "OK - Normal spacing detected" -ForegroundColor Green
    }
    
    Write-Host "`nFIRST 300 CHARS:" -ForegroundColor Cyan
    $DisplayText = $Output.Substring(0, [Math]::Min(300, $Output.Length))
    Write-Host $DisplayText -ForegroundColor White
    
    if ($Output.Length -gt 300) {
        Write-Host "..." -ForegroundColor Gray
    }
    
    Write-Host "`nMETRICS:" -ForegroundColor Cyan
    Write-Host "  Length: $($Output.Length) chars" -ForegroundColor Gray
    Write-Host "  Lines: $(($Output -split '\n').Count)" -ForegroundColor Gray
    
    $Words = $Output -split '\s+' | Where-Object { $_.Length -gt 0 }
    $AvgWordLen = if ($Words.Count -gt 0) { [int]($Words | Measure-Object -Property Length -Average).Average } else { 0 }
    Write-Host "  Avg word length: $AvgWordLen chars" -ForegroundColor Gray
    
    $LongestWord = if ($Words.Count -gt 0) { ($Words | Measure-Object -Property Length -Maximum).Maximum } else { 0 }
    Write-Host "  Longest word: $LongestWord chars" -ForegroundColor Gray
    
    if ($IsBroken) {
        Write-Host "`nDIAGNOSIS: Spaceless bug detected - Retraining required" -ForegroundColor Red
    }
}

# ============================================================================
# MAIN
# ============================================================================

Write-Host "`n$('='*80)"
Write-Host "BROKEN ADAPTER TEST - debugging_lora AND review_lora"
Write-Host $('='*80)

Write-Host "`nTarget: $InferenceUrl" -ForegroundColor Cyan
Write-Host "Tests: $($TestCases.Count)" -ForegroundColor Cyan

# Quick health check
try {
    $HealthUrl = "$InferenceUrl/health"
    $Health = Invoke-RestMethod -Uri $HealthUrl -Method Get -ErrorAction Stop
    Write-Host "Service UP" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: Cannot reach service at $InferenceUrl" -ForegroundColor Red
    Write-Host "Run: docker-compose up -d" -ForegroundColor Yellow
    exit 1
}

# Run all tests
$Results = @()
foreach ($TestCase in $TestCases) {
    $Result = Invoke-InferenceAPI -Prompt $TestCase.Prompt -Adapter $TestCase.Adapter -MaxTokens $MaxTokens -Temperature $Temperature
    Show-Results -TestName $TestCase.Name -Result $Result -Adapter $TestCase.Adapter
    
    $Results += @{
        TestName = $TestCase.Name
        Adapter = $TestCase.Adapter
        Broken = (Test-SpacelessOutput -Output $Result.Output)
        Output = $Result.Output
    }
}

# Summary
Write-Host "`n$('='*80)"
Write-Host "SUMMARY"
Write-Host $('='*80)

$BrokenCount = ($Results | Where-Object { $_.Broken }).Count
$TotalCount = $Results.Count

if ($BrokenCount -eq 0) {
    Write-Host "`nAll adapters OK" -ForegroundColor Green
}
else {
    Write-Host "`n$BrokenCount / $TotalCount adapters show spaceless bug" -ForegroundColor Red
    Write-Host "`nBroken adapters:" -ForegroundColor Red
    $Results | Where-Object { $_.Broken } | ForEach-Object {
        Write-Host "  - $($_.Adapter)" -ForegroundColor Red
    }
    Write-Host "`nRetrain:" -ForegroundColor Yellow
    $cmd1 = "docker exec docker-training-1 bash -c 'cd /app && python3 -m training.train_gpu_simple --capabilities debugging --fresh debugging'"
    $cmd2 = "docker exec docker-training-1 bash -c 'cd /app && python3 -m training.train_gpu_simple --capabilities review --fresh review'"
    Write-Host "  $cmd1" -ForegroundColor Gray
    Write-Host "  $cmd2" -ForegroundColor Gray
}

Write-Host ""
