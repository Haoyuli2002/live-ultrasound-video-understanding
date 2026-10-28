"""Select reproducible complete condition triplets from a video-disjoint split."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

CONDITIONS = {'before_with_asr', 'through_with_asr', 'before_mask_asr'}


def select(rows, split, limit=0, seed=42):
    if limit < 0 or not rows:
        raise ValueError('Need nonempty samples and a nonnegative target limit')
    if set(r['video_id'] for r in rows) & set(split['train_video_ids']):
        raise ValueError('Evaluation videos overlap training videos')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['video_id'], row['sentence_id'])].append(row)
    for k, group in groups.items():
        if len(group) != 3 or {r['condition'] for r in group} != CONDITIONS:
            raise ValueError(f'Incomplete or duplicated condition triplet: {k}')
        if len({r['target'] for r in group}) != 1:
            raise ValueError(f'Condition references differ: {k}')
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if limit:
        keys = keys[:limit]
    return [r for k in sorted(keys) for r in sorted(groups[k], key=lambda r: r['condition'])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit-targets', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    rows = select([json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()],
                  json.loads(Path(a.split).read_text()), a.limit_targets, a.seed)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
    if out.exists() and out.read_text() != text:
        raise ValueError('Selection changed; use a new evaluation directory')
    out.write_text(text)
    print(f'[eval] {len(rows)} rows, {len(rows)//3} targets, {len({r["video_id"] for r in rows})} videos')


if __name__ == '__main__':
    main()
