"""
Architecture invariant checks — model-naming and backend-registry refactoring.

Run from any directory:
    python scripts/check_architecture.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

results = []


def check(n, name, fn):
    try:
        fn()
        results.append(f'PASS [{n:02d}] {name}')
    except Exception as e:
        results.append(f'FAIL [{n:02d}] {name} -- {e}')


def check_no_banned_model_literals():
    SCAN_DIRS = ['inference', 'orchestration', 'training', 'shared', 'scripts']
    BANNED = re.compile(r'\b(qwen|deepseek|mistral)\b', re.IGNORECASE)
    ALLOWLIST = {
        os.path.join('shared', 'tests', 'test_no_model_literals.py'),
        os.path.join('shared', 'contracts', 'inference.py'),
        os.path.join('shared', 'contracts', 'models.py'),
    }
    hits = []
    for d in SCAN_DIRS:
        base = root / d
        if not base.exists():
            continue
        for f in base.rglob('*.py'):
            parts = f.relative_to(root).parts
            if any(p in ('tests', '__pycache__', '.claude') for p in parts):
                continue
            rel = str(f.relative_to(root))
            if any(rel == a.replace('\\', '/') or rel == a for a in ALLOWLIST):
                continue
            for n, l in enumerate(f.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
                if BANNED.search(l):
                    hits.append(f' {rel}:{n}: {l.strip()}')
    if hits:
        raise AssertionError('violations:\n' + '\n'.join(hits[:20]))


check(1, 'No banned model-name literals in source', check_no_banned_model_literals)


def check_no_model_family_in_services():
    hits = []
    for d in ['inference/app', 'orchestration/app', 'training/app', 'training/training', 'scripts']:
        p = root / d
        if not p.exists():
            continue
        for f in p.rglob('*.py'):
            for n, l in enumerate(f.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
                if 'model_family' in l:
                    hits.append(f' {f.relative_to(root)}:{n}: {l.strip()}')
    if hits:
        raise AssertionError('\n'.join(hits[:20]))


check(2, 'No model_family in service source', check_no_model_family_in_services)


def check_capability_to_family_deleted():
    src = (root / 'inference/app/backends/factory.py').read_text()
    assert 'CAPABILITY_TO_FAMILY' not in src
    assert 'get_backend_for_capability' not in src


check(3, 'CAPABILITY_TO_FAMILY deleted from factory.py', check_capability_to_family_deleted)


def check_role_model_family_fully_gone():
    hits = []
    for d in ['inference/app', 'orchestration/app', 'training/app', 'training/training', 'shared', 'scripts']:
        p = root / d
        if not p.exists():
            continue
        for f in p.rglob('*.py'):
            if '.claude' in str(f):
                continue
            txt = f.read_text(encoding='utf-8', errors='replace')
            if 'role_model_family' in txt:
                hits.append(str(f.relative_to(root)))
    if hits:
        raise AssertionError('still present in: ' + ', '.join(hits))


check(4, 'role_model_family fully renamed', check_role_model_family_fully_gone)


def check_models_yaml_structure():
    import yaml
    d = yaml.safe_load(open(root / 'models.yaml', encoding='utf-8'))
    assert 'default_backend' in d
    assert 'primary' in d['backends']
    p = d['backends']['primary']
    for field in ['served_url', 'model_id', 'max_model_len', 'backend_type']:
        assert field in p, f'missing {field}'


check(5, 'models.yaml structure valid', check_models_yaml_structure)


def check_contracts_models_exports():
    os.environ.setdefault('MODELS_YAML_PATH', str(root / 'models.yaml'))
    from shared.contracts.models import BackendEntry, ModelsRegistry, load_registry, get_backend_entry
    reg = load_registry()
    assert all(k in reg.__class__.__fields__ for k in ['default_backend', 'backends'])


check(6, 'shared/contracts/models.py exports OK', check_contracts_models_exports)


def check_generate_request_uses_backend():
    os.environ['MODELS_YAML_PATH'] = str(root / 'models.yaml')
    from shared.contracts.inference import GenerateRequest, GenerateResponse, Message
    r = GenerateRequest(
        backend='primary',
        role='coder',
        messages=[Message(role='user', content='x')]
    )
    assert r.backend == 'primary'
    assert not hasattr(r, 'model_family') or r.__class__.__fields__.get('model_family') is None
    resp = GenerateResponse(content='y', backend='primary', role='coder')
    assert resp.backend == 'primary'


check(7, 'GenerateRequest/Response use backend field', check_generate_request_uses_backend)


def check_legacy_alias_shim():
    os.environ['MODELS_YAML_PATH'] = str(root / 'models.yaml')
    os.environ['INFERENCE__ACCEPT_LEGACY_BACKEND_NAMES'] = 'true'
    import importlib
    import shared.contracts.inference as inf
    importlib.reload(inf)
    from shared.contracts.inference import GenerateRequest, Message
    r = GenerateRequest(
        backend='qwen',
        role='coder',
        messages=[Message(role='user', content='x')]
    )
    assert r.backend == 'primary', f'expected primary got {r.backend}'
    del os.environ['INFERENCE__ACCEPT_LEGACY_BACKEND_NAMES']


check(8, 'Legacy alias shim resolves qwen->primary', check_legacy_alias_shim)


def check_inference_settings_clean():
    import importlib
    import inference.app.config.settings as m
    importlib.reload(m)
    s = m.settings
    assert hasattr(s, 'models_yaml_path')
    assert hasattr(s, 'accept_legacy_backend_names')
    for old in ['qwen_url', 'qwen_model_id', 'qwen_auth_token', 'qwen_ssl_verify']:
        assert not hasattr(s, old), f'{old} still present'


check(9, 'inference/app/config/settings clean', check_inference_settings_clean)


def check_orchestration_settings_clean():
    from orchestration.app.config.settings import settings
    assert hasattr(settings, 'role_backend'), 'missing role_backend'
    assert not hasattr(settings, 'role_model_family'), 'role_model_family still present'


check(10, 'orchestration/app/config/settings clean', check_orchestration_settings_clean)


def check_training_settings_clean():
    from training.app.config.settings import settings
    assert hasattr(settings, 'models_yaml_path')
    assert not hasattr(settings, 'qwen_model')


check(11, 'training/app/config/settings clean', check_training_settings_clean)


def check_factory_uses_registry():
    src = (root / 'inference/app/backends/factory.py').read_text()
    assert 'load_registry' in src
    assert 'BackendEntry' in src or 'entry' in src


check(12, 'factory.py uses load_registry', check_factory_uses_registry)


def check_local_vllm_no_tokenizer_hacks():
    src = (root / 'inference/app/backends/local_vllm.py').read_text(encoding='utf-8')
    assert 'backend_name' in src
    assert '_is_deepseek' not in src
    assert 'Metaspace' not in src


check(13, 'local_vllm.py constructor + no hacks', check_local_vllm_no_tokenizer_hacks)


def check_proto_field_renamed():
    src = open(root / 'shared/proto/inference.proto', encoding='utf-8').read()
    assert re.search(r'string\s+backend\s*=\s*1', src), 'field not renamed'
    non_comment_lines = [l for l in src.splitlines() if not l.strip().startswith('//')]
    assert not any('model_family' in l for l in non_comment_lines), 'model_family field still present'


check(14, 'inference.proto field renamed to backend', check_proto_field_renamed)


def check_compose_yml_updated():
    src = open(root / 'docker/compose.yml').read()
    assert 'vllm-primary' in src
    assert 'vllm-qwen' not in src
    assert 'VLLM_MODEL_ID' in src
    assert '/config/models.yaml' in src


check(15, 'docker/compose.yml updated correctly', check_compose_yml_updated)


def check_render_compose_env_exists():
    assert (root / 'scripts/render_compose_env.py').exists()


check(16, 'scripts/render_compose_env.py exists', check_render_compose_env_exists)


def check_render_compose_env_runs():
    r = subprocess.run(
        [sys.executable, str(root / 'scripts/render_compose_env.py')],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert r.returncode == 0, r.stderr[:200]
    assert (root / '.env.vllm').exists() or 'VLLM_' in r.stdout


check(17, 'render_compose_env.py runs without error', check_render_compose_env_runs)


def check_no_tokenizer_hacks_in_training():
    for f in [
        'training/training/lora_trainer.py',
        'training/training/validate_adapter.py'
    ]:
        src = (root / f).read_text(encoding='utf-8')
        for banned in ['_is_deepseek', 'Metaspace', 'ByteLevel', 'tokenizer.json']:
            assert banned not in src, f'{banned} found in {f}'


check(18, 'No tokenizer hacks in lora_trainer/validate_adapter', check_no_tokenizer_hacks_in_training)


def check_train_gpu_simple_old_constants_deleted():
    src = (root / 'training/training/train_gpu_simple.py').read_text()
    for banned in ['DEEPSEEK_ID', 'MISTRAL_ID', '_BOOTSTRAP_MODEL']:
        assert banned not in src, f'{banned} still in train_gpu_simple.py'


check(19, 'train_gpu_simple.py old constants deleted', check_train_gpu_simple_old_constants_deleted)


def check_scripts_no_role_model_family():
    hits = []
    for s in [
        'scripts/seed_adapter_metadata.py',
        'scripts/eval_adapters.py',
        'scripts/e2e_test.py'
    ]:
        p = root / s
        if not p.exists():
            continue
        src = p.read_text(encoding='utf-8')
        if 'ROLE_MODEL_FAMILY' in src:
            hits.append(f'{s}: ROLE_MODEL_FAMILY')
    if hits:
        raise AssertionError('; '.join(hits))


check(20, 'scripts: no ROLE_MODEL_FAMILY', check_scripts_no_role_model_family)


def check_test_no_model_literals_exists():
    assert (root / 'shared/tests/test_no_model_literals.py').exists()


check(21, 'shared/tests/test_no_model_literals.py exists', check_test_no_model_literals_exists)


def check_nodes_uses_role_backend_and_registry():
    src = (root / 'orchestration/app/graph/nodes.py').read_text()
    assert 'role_backend' in src
    assert 'load_registry' in src
    assert '"qwen"' not in src and "'qwen'" not in src


check(22, 'nodes.py uses role_backend + registry', check_nodes_uses_role_backend_and_registry)


def check_grpc_server_uses_backend_field():
    src = (root / 'inference/app/grpc/server.py').read_text()
    assert 'request.backend' in src or '.backend' in src
    assert 'model_family' not in src


check(23, 'grpc/server.py uses backend field', check_grpc_server_uses_backend_field)


def check_inference_grpc_client_uses_backend():
    src = (root / 'orchestration/app/clients/inference_grpc.py').read_text()
    assert 'model_family' not in src


check(24, 'inference_grpc.py uses backend field', check_inference_grpc_client_uses_backend)


def check_test_suite_collects():
    r = subprocess.run(
        [
            sys.executable, '-m', 'pytest', '--collect-only', '-q',
            'inference/tests', 'orchestration/tests', 'shared/tests'
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    errors = [
        l for l in r.stdout.splitlines() + r.stderr.splitlines()
        if ('ERROR' in l or 'error' in l.lower())
    ]
    errors = [e for e in errors if 'warning' not in e.lower()]
    if r.returncode != 0 and errors:
        raise AssertionError('\n'.join(errors[:10]))


check(25, 'Test suite collects without import errors', check_test_suite_collects)


# Print results
print()
print('=' * 60)
passed = sum(1 for r in results if r.startswith('PASS'))
failed = sum(1 for r in results if r.startswith('FAIL'))
for r in results:
    print(r)
print('=' * 60)
print(f' {passed} passed | {failed} failed | {len(results)} total')
print('=' * 60)
