# 超声视频 QA demo：10 视频、286 条 QA

此目录自带 10 个完整视频、151 张证据帧、286 条英文 QA 和展示页面。只需 Git LFS、Python 3 和浏览器，不需要模型权重、GPU 或 pip/npm 依赖。视频通过 Git LFS 分发；其他页面、标签和证据图直接包含在 Git 中。

## 首次 clone

先在本机安装 Git 和 Git LFS，然后运行：

```bash
git lfs install
git clone https://github.com/Haoyuli2002/live-ultrasound-video-understanding.git
cd live-ultrasound-video-understanding
git lfs pull --include="demo/ultrasound-qa/media/*.mp4" --exclude=""
python3 demo/ultrasound-qa/serve.py
```

## 已经 clone

在仓库根目录运行：

```bash
git lfs install
git pull --ff-only
git lfs pull --include="demo/ultrasound-qa/media/*.mp4" --exclude=""
python3 demo/ultrasound-qa/serve.py
```

浏览器会自动打开 `http://127.0.0.1:8767/`，视频自动加载，不需要手动选择文件。保持终端运行，Ctrl+C 停止。端口占用时加 `--port 8768`。Windows 可将 `python3` 换为 `py`。macOS 也可在此目录运行 `Start.command`。

请使用 Git clone/pull 与 Git LFS 获取视频；网页上的源码 ZIP 不保证包含实际 LFS 媒体。下载完成后，播放无需联网。不要直接双击主页面 `index.html`，它需要附带的本地服务读取标签并支持视频跳转。

## 展示与数据来源

- 视频和 QA 切换、四种问题类型筛选、搜索、当前 QA 链接。
- 跳到 Query、Answer 或证据时刻；播放 Query–Answer 区间并在 Answer 暂停。
- 即时 QA 的 Query 与 Answer 相同，播放时展示前后各 3 秒上下文，原始标注时间不变。
- `artifact.html` 是额外的单文件轻量预览，可直接双击。浏览证据缩略图无需服务；若需要在该页面播放视频，在“本机视频”中多选本目录 `media/` 下的 MP4。正常启动的主页面无需这一步。

源版本：`stage3_visual_v4_holdout16_20261121`。这些是 holdout 参考标注，非实时模型预测，也不代表独立临床专家审核。QA 保留英文原文：当前观察 73、历史回忆 73、事件通知 70、时序比较 70。原视频及证据图保持原包字节；仅此 10 个视频随 demo 分发。

视频 ID：`CW3TLabogko`、`B2USlWmqOV0`、`ODF8ZLr_52s`、`lREUPXCpK8Y`、`7G56DN38mz8`、`0Cboqf1Qnhc`、`qYM2sft8R2I`、`YllLJNPnpjY`、`v9ar1mhpT-g`、`az6h_u_8nhA`。

`SHA256SUMS` 记录发布文件的完整校验值；Linux 可在本目录运行 `sha256sum -c SHA256SUMS`。字体许可见 `OFL.txt`。
