"""Resolve the assigned GPU using UUIDs actually returned by nvidia-smi."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def select_gpu(rows, cuda_uuid, environment):
    # PyTorch UUID string representations vary; never pass that representation
    # directly to nvidia-smi. Return the driver's own UUID string after matching.
    pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
    match = re.search(pattern, str(cuda_uuid))
    if match:
        matched = [uuid for _, uuid in rows if match[0].lower() in uuid.lower()]
        if len(matched) == 1:
            return matched[0]
    if len(rows) == 1:
        return rows[0][1]
    for key in ('SLURM_STEP_GPUS', 'SLURM_JOB_GPUS', 'CUDA_VISIBLE_DEVICES'):
        visible = environment.get(key, '').strip()
        if not visible or ',' in visible:
            continue
        matched = [uuid for index, uuid in rows if visible == index or visible == uuid]
        if len(matched) == 1:
            return matched[0]
    raise RuntimeError('Cannot uniquely identify the assigned GPU for monitoring')


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    raw = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                         text=True, capture_output=True, check=True).stdout
    rows = [tuple(value.strip() for value in line.split(',', 1)) for line in raw.splitlines() if ',' in line]
    cuda_uuid = str(torch.cuda.get_device_properties(0).uuid)
    environment = {key: os.environ.get(key, '') for key in ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS')}
    uuid = select_gpu(rows, cuda_uuid, environment)
    # Fail before starting training if this ID cannot actually be queried.
    subprocess.run(['nvidia-smi', '--id=' + uuid, '--query-gpu=uuid', '--format=csv,noheader'],
                   check=True, capture_output=True, text=True)
    args.record.write_text(json.dumps(dict(cuda_uuid=cuda_uuid, driver_gpus=rows,
                                          environment=environment, selected_uuid=uuid), indent=2) + '\n')
    print(uuid)


if __name__ == '__main__':
    main()
