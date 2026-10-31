"""LRZ full-GPU partitions; this module deliberately needs no CUDA packages."""
import os
import sys

PARTITIONS = {
    'h100':'lrz-hgx-h100-94x4',
    'a100':'lrz-dgx-a100-80x8,lrz-hgx-a100-80x4',
    'auto':'lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4',
}


def partitions(profile='auto'):
    if profile not in PARTITIONS:
        raise ValueError('GPU_TYPE must be auto, a100 or h100')
    return PARTITIONS[profile]


if __name__=='__main__':
    print(partitions(sys.argv[1] if len(sys.argv)>1 else os.environ.get('GPU_TYPE','auto')))
