"""Bounded CPU tensor-transfer checks before loading Stage 1 model weights."""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import tempfile

import torch


def configure_sharing(worker_id=None):
    """Also run in spawned workers, which do not inherit Python global state."""
    available = torch.multiprocessing.get_all_sharing_strategies()
    strategy = os.environ.get("STAGE1_SHARING_STRATEGY", "").strip()
    strategy = strategy or ("file_descriptor" if "file_descriptor" in available
                            else "file_system")
    if strategy not in available:
        raise ValueError(f"Unsupported STAGE1_SHARING_STRATEGY={strategy!r}; "
                         f"available={sorted(available)}")
    torch.multiprocessing.set_sharing_strategy(strategy)
    return strategy


def check_socket_path():
    # Both sharing strategies can need a Unix socket below the temp directory.
    with tempfile.TemporaryDirectory(prefix="torch-shm-dir-") as directory:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.bind(os.path.join(directory, "manager.sock"))


def check_tensor_transfer(workers):
    strategy = configure_sharing()
    print(f"DataLoader preflight: torch={torch.__version__}; workers={workers}; "
          f"sharing={strategy}; TMPDIR={tempfile.gettempdir()}", flush=True)
    check_socket_path()
    dataset = torch.utils.data.TensorDataset(torch.arange(workers * 2))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=workers, timeout=20,
        worker_init_fn=configure_sharing)
    actual = [int(batch[0].item()) for batch in loader]
    if actual != list(range(workers * 2)):
        raise RuntimeError(f"Unexpected tensor-transfer results: {actual}")
    print("DataLoader preflight passed (CPU tensor transfer)", flush=True)


def preflight(workers, timeout=90):
    """Isolate the probe from CUDA and bound hangs in queue feeder threads."""
    if workers == 0:
        print("DataLoader preflight: workers=0, no interprocess transfer", flush=True)
        return
    configure_sharing()
    process = subprocess.Popen(
        [sys.executable, "-m", "stage1.runtime", "--workers", str(workers)],
        start_new_session=True)
    try:
        status = process.wait(timeout=timeout)
        if status:
            raise RuntimeError(f"DataLoader preflight failed (exit {status}); "
                               "inspect the traceback above. Use a short writable "
                               "TMPDIR, or DATALOADER_NUM_WORKERS=0 in sbatch "
                               "(--dataloader-num-workers 0 for direct Python).")
    except BaseException:
        # Include workers in cleanup; killing only their parent can leave them alive.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    check_tensor_transfer(args.workers)
