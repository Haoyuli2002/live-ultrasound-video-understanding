"""Serve the bundled snapshot and refresh its four files from the cluster over SSH."""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import threading
import webbrowser

ROOT = Path(__file__).resolve().parent
REMOTE = "/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding/results/v3_training_monitor/"
FILES = [f"{batch}_{metric}" for batch in ("bs4", "bs8") for metric in ("training.log", "gpu.csv")]


def sync(host, folder, stop):
    with TemporaryDirectory() as control:
        ssh = f"ssh -o ConnectTimeout=8 -o ControlMaster=auto -o ControlPersist=60 -o ControlPath={control}/socket"
        args = ["rsync", "-rLt", "-e", ssh]
        for name in FILES:
            args.extend(["--include", name])
        args.extend(["--exclude", "*", f"{host}:{REMOTE}", str(folder) + "/"])
        while not stop.is_set():
            try:
                subprocess.run(args, check=True)
                (folder / "sync.txt").write_text("已同步远端日志", encoding="utf-8")
            except (OSError, subprocess.CalledProcessError):
                (folder / "sync.txt").write_text("同步失败，显示上次取得的数据；请检查终端中的 SSH 提示", encoding="utf-8")
            stop.wait(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="ge75vid2@login-01.ai.lrz.de", help="SSH destination or configured alias")
    parser.add_argument("--port", type=int, default=18769)
    parser.add_argument("--offline", action="store_true", help="Show only the bundled snapshot")
    args = parser.parse_args()
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for name in FILES:
            shutil.copy2(ROOT / "snapshot" / name, folder / name)
        shutil.copy2(ROOT / "index.html", folder / "index.html")
        (folder / "sync.txt").write_text("本地快照" if args.offline else "显示本地快照，正在连接服务器", encoding="utf-8")
        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(folder))
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            stop = threading.Event()
            if not args.offline:
                threading.Thread(target=sync, args=(args.host, folder, stop), daemon=True).start()
            url = f"http://127.0.0.1:{server.server_port}/"
            print(url, flush=True)
            webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                stop.set()


if __name__ == "__main__":
    main()
