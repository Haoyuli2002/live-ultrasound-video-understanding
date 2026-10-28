"""Shared identities and append-only, configuration-checked evaluation outputs."""
import hashlib
import json
from pathlib import Path


def key(row):
    return (row['video_id'], row['sentence_id'], row['condition'], row.get('visual_control', 'normal'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def indexed(rows):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ValueError(f'Duplicate sample: {k}')
        result[k] = row
    return result


def paired(rows, baseline):
    a, b = indexed(rows), indexed(baseline)
    if a.keys() != b.keys():
        raise ValueError('Prediction and baseline sample identities differ')
    if any(a[k]['target'] != b[k]['target'] for k in a):
        raise ValueError('Prediction and baseline references differ')
    return [(a[k], b[k]) for k in a]


def resume_rows(output, config):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = Path(str(output) + '.config.json')
    if meta.exists():
        if json.loads(meta.read_text()) != config:
            raise ValueError(f'Configuration changed; choose a new output: {output}')
    else:
        if output.exists() and output.stat().st_size:
            raise ValueError(f'Existing output has no manifest: {output}')
        meta.write_text(json.dumps(config, indent=2, sort_keys=True))
    if not output.exists():
        return []
    # A killed process may leave an incomplete final write. Only discard that tail.
    data = output.read_bytes()
    if data and not data.endswith(b'\n'):
        cut = data.rfind(b'\n') + 1
        print(f'[resume] discarding incomplete final line in {output}', flush=True)
        with output.open('r+b') as stream:
            stream.truncate(cut)
        data = data[:cut]
    return [json.loads(line) for line in data.splitlines() if line.strip()]
