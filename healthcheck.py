import os
import re
import subprocess
import sys
from pathlib import Path

root = Path('.')
results = []


def check(n, name, fn):
    try:
        fn()
        results.append(f'PASS [{n:02d}] {name}')
    except Exception as e:
        results.append(f'FAIL [{n:02d}] {name} -- {e}')


# 1. No banned model-name literals in source
def c1():
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


check(1, 'No banned model-name literals in source', c1)


# 2. No model_family in source (outside contracts)
def c2():
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


check(2, 'No model_family in service source', c2)


# 3. CAPABILITY_TO_FAMILY deleted
def c3():
    src = (root / 'inference/app/backends/factory.py').read_text()
    assert 'CAPABILITY_TO_FAMILY' not in src
    assert 'get_backend_for_capability' not in src


check(3, 'CAPABILITY_TO_FAMILY deleted from factory.py', c3)


# 4. role_model_family fully gone (scan service dirs only, skip this script)
def c4():
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


check(4, 'role_model_family fully renamed', c4)


# 5. models.yaml exists and has required fields
def c5():
    import yaml
    d = yaml.safe_load(open('models.yaml', encoding='utf-8'))
    assert 'default_backend' in d
    assert 'primary' in d['backends']
    p = d['backends']['primary']
    for field in ['served_url', 'model_id', 'max_model_len', 'backend_type']:
        assert field in p, f'missing {field}'


check(5, 'models.yaml structure valid', c5)


# 6. shared/contracts/models.py exports
def c6():
    os.environ.setdefault('MODELS_YAML_PATH', 'models.yaml')
    from shared.contracts.models import BackendEntry, ModelsRegistry, load_registry, get_backend_entry
    reg = load_registry()
    assert all(k in reg.__class__.__fields__ for k in ['default_backend', 'backends'])


check(6, 'shared/contracts/models.py exports OK', c6)


# 7. GenerateRequest/Response use backend field
def c7():
    os.environ['MODELS_YAML_PATH'] = 'models.yaml'
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


check(7, 'GenerateRequest/Response use backend field', c7)


# 8. Legacy alias shim resolves when flag set
def c8():
    os.environ['MODELS_YAML_PATH'] = 'models.yaml'
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


check(8, 'Legacy alias shim resolves qwen->primary', c8)


# 9. inference settings: new fields present, old gone
def c9():
    import importlib
    import inference.app.config.settings as m
    importlib.reload(m)
    s = m.settings
    assert hasattr(s, 'models_yaml_path')
    assert hasattr(s, 'accept_legacy_backend_names')
    for old in ['qwen_url', 'qwen_model_id', 'qwen_auth_token', 'qwen_ssl_verify']:
        assert not hasattr(s, old), f'{old} still present'


check(9, 'inference/app/config/settings clean', c9)


# 10. orchestration settings: role_backend present
def c10():
    from orchestration.app.config.settings import settings
    assert hasattr(settings, 'role_backend'), 'missing role_backend'
    assert not hasattr(settings, 'role_model_family'), 'role_model_family still present'


check(10, 'orchestration/app/config/settings clean', c10)


# 11. training settings: models_yaml_path present, qwen_model gone
def c11():
    from training.app.config.settings import settings
    assert hasattr(settings, 'models_yaml_path')
    assert not hasattr(settings, 'qwen_model')


check(11, 'training/app/config/settings clean', c11)


# 12. factory.py uses registry
def c12():
    src = (root / 'inference/app/backends/factory.py').read_text()
    assert 'load_registry' in src
    assert 'BackendEntry' in src or 'entry' in src


check(12, 'factory.py uses load_registry', c12)


# 13. local_vllm.py: new constructor, no tokenizer hacks
def c13():
    src = (root / 'inference/app/backends/local_vllm.py').read_text(encoding='utf-8')
    assert 'backend_name' in src
    assert '_is_deepseek' not in src
    assert 'Metaspace' not in src


check(13, 'local_vllm.py constructor + no hacks', c13)


# 14. proto field renamed, tag 1 kept
def c14():
    src = open('shared/proto/inference.proto', encoding='utf-8').read()
    # Proto uses alignment spacing: "string              backend       = 1;"
    assert re.search(r'string\s+backend\s*=\s*1', src), 'field not renamed'
    # model_family must not appear as a field definition (comments are OK)
    non_comment_lines = [l for l in src.splitlines() if not l.strip().startswith('//')]
    assert not any('model_family' in l for l in non_comment_lines), 'model_family field still present'


check(14, 'inference.proto field renamed to backend', c14)


# 15. compose.yml: service renamed, vars present, mounts present
def c15():
    src = open('docker/compose.yml').read()
    assert 'vllm-primary' in src
    assert 'vllm-qwen' not in src
    assert 'VLLM_MODEL_ID' in src
    assert '/config/models.yaml' in src


check(15, 'docker/compose.yml updated correctly', c15)


# 16. render_compose_env.py exists
def c16():
    assert (root / 'scripts/render_compose_env.py').exists()


check(16, 'scripts/render_compose_env.py exists', c16)


# 17. render_compose_env.py runs and produces output
def c17():
    r = subprocess.run(
        [sys.executable, 'scripts/render_compose_env.py'],
        capture_output=True,
        text=True
    )
    assert r.returncode == 0, r.stderr[:200]
    assert (root / '.env.vllm').exists() or 'VLLM_' in r.stdout


check(17, 'render_compose_env.py runs without error', c17)


# 18. No _is_deepseek / tokenizer hacks in training
def c18():
    for f in [
        'training/training/lora_trainer.py',
        'training/training/validate_adapter.py'
    ]:
        src = (root / f).read_text(encoding='utf-8')
        for banned in ['_is_deepseek', 'Metaspace', 'ByteLevel', 'tokenizer.json']:
            assert banned not in src, f'{banned} found in {f}'


check(18, 'No tokenizer hacks in lora_trainer/validate_adapter', c18)


# 19. train_gpu_simple.py: no old constants
def c19():
    src = (root / 'training/training/train_gpu_simple.py').read_text()
    for banned in ['DEEPSEEK_ID', 'MISTRAL_ID', '_BOOTSTRAP_MODEL']:
        assert banned not in src, f'{banned} still in train_gpu_simple.py'


check(19, 'train_gpu_simple.py old constants deleted', c19)


# 20. scripts: no hardcoded model lists
def c20():
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


check(20, 'scripts: no ROLE_MODEL_FAMILY', c20)


# 21. test_no_model_literals.py exists
def c21():
    assert (root / 'shared/tests/test_no_model_literals.py').exists()


check(21, 'shared/tests/test_no_model_literals.py exists', c21)


# 22. nodes.py: uses role_backend + load_registry, no qwen literal
def c22():
    src = (root / 'orchestration/app/graph/nodes.py').read_text()
    assert 'role_backend' in src
    assert 'load_registry' in src
    assert '"qwen"' not in src and "'qwen'" not in src


check(22, 'nodes.py uses role_backend + registry', c22)


# 23. gRPC server uses backend field
def c23():
    src = (root / 'inference/app/grpc/server.py').read_text()
    assert 'request.backend' in src or '.backend' in src
    assert 'model_family' not in src


check(23, 'grpc/server.py uses backend field', c23)


# 24. inference_grpc client uses backend
def c24():
    src = (root / 'orchestration/app/clients/inference_grpc.py').read_text()
    assert 'model_family' not in src


check(24, 'inference_grpc.py uses backend field', c24)


# 25. All tests importable (dry-run collect)
def c25():
    r = subprocess.run(
        [
            sys.executable, '-m', 'pytest', '--collect-only', '-q',
            'inference/tests', 'orchestration/tests', 'shared/tests'
        ],
        capture_output=True,
        text=True
    )
    errors = [
        l for l in r.stdout.splitlines() + r.stderr.splitlines()
        if ('ERROR' in l or 'error' in l.lower())
    ]
    errors = [e for e in errors if 'warning' not in e.lower()]
    if r.returncode != 0 and errors:
        raise AssertionError('\n'.join(errors[:10]))


check(25, 'Test suite collects without import errors', c25)


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
