import json, pathlib, datetime
caps = ['planning', 'research', 'criticism']
base = pathlib.Path('/data/lora_checkpoints')
for cap in caps:
    d = base / f'{cap}_lora'
    meta_path = d / 'metadata.json'
    if not meta_path.exists():
        print(f'{cap}_lora  MISSING metadata.json')
        continue
    meta = json.loads(meta_path.read_text())
    has_tok = (d / 'tokenizer.json').exists()
    tok_type = 'n/a'
    if has_tok:
        t = json.loads((d / 'tokenizer.json').read_text())
        tok_type = t.get('pre_tokenizer', {}).get('type', 'unknown')
    print(f'{cap}_lora  v{meta.get("version","?")}  samples={meta.get("samples","?")}  loss={meta.get("loss","?")}  base={meta.get("base_model","?")}  tok_json={has_tok}  pre_tok={tok_type}')

data_dir = pathlib.Path('/data/training_data_curated')
for cap in caps:
    f = data_dir / f'{cap}_dataset.jsonl'
    if f.exists():
        mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
        lines = sum(1 for _ in f.open())
        print(f'{cap}_dataset.jsonl  lines={lines}  modified={mtime:%Y-%m-%d}')
    else:
        print(f'{cap}_dataset.jsonl  MISSING')
