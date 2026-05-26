"""
批量 VLM 筛选超声视频
=======================
对爬取的所有视频运行 Qwen2-VL-2B 逐帧分析，生成筛选报告。

特性:
- 边跑边保存（每个视频处理完立即写入JSON）
- 断点续跑（自动跳过已处理的视频）
- 超时保护（单个视频超时后仍基于已分析帧做判断）
- tqdm 进度条

使用方法:
    python batch_filter.py
    python batch_filter.py --max-videos 5          # 只跑5个测试
    python batch_filter.py --timeout 600           # 单视频超时10分钟
"""

import json
import time
import glob
import sys
import signal
from pathlib import Path
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    print("pip install tqdm")
    tqdm = None

from video_filter_vlm import load_model, sample_frames, run_stage1, parse_json_from_text

# ============================================================================
# 配置
# ============================================================================

DEFAULT_INPUT_DIR = "UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media"
NUM_FRAMES = 8
DEFAULT_TIMEOUT = 600  # 单视频最大处理时间（秒）


# ============================================================================
# 超时处理
# ============================================================================

class TimeoutError(Exception):
    pass

# 用于在超时时传递已分析的帧结果
_partial_frame_results = []


def timeout_handler(signum, frame):
    raise TimeoutError("视频处理超时")


# ============================================================================
# 规则判断 (替代 Stage 2, 不需要API)
# ============================================================================

def rule_based_decision(frame_results: list) -> dict:
    """基于帧级分析结果做简单规则判断
    
    注意: "医学影像" 和 "超声画面" 都算作有效超声相关帧。
    Qwen2-VL-2B 经常把超声画面标记为 "医学影像"，所以两者合并统计。
    """
    total = len(frame_results)
    if total == 0:
        return {"决策": "丢弃", "判断理由": "无有效帧", "质量评分": 0}

    type_counts = {}
    for r in frame_results:
        t = r.get("帧类型", "其他")
        type_counts[t] = type_counts.get(t, 0) + 1

    # "医学影像" + "超声画面" 都算作有效超声帧
    us_count = type_counts.get("超声画面", 0) + type_counts.get("医学影像", 0)
    lecture_count = type_counts.get("讲课画面", 0)
    ppt_count = type_counts.get("PPT幻灯片", 0)

    us_ratio = us_count / total
    lecture_ratio = (lecture_count + ppt_count) / total

    if us_ratio >= 0.7:
        decision, quality = "保留", int(70 + us_ratio * 30)
        reason = f"超声/医学影像占比{us_ratio:.0%}，适合训练"
    elif us_ratio >= 0.4:
        decision, quality = "需要裁剪", int(40 + us_ratio * 40)
        reason = f"超声/医学影像占比{us_ratio:.0%}，混合内容需裁剪"
    elif lecture_ratio >= 0.6:
        decision, quality = "丢弃", int(20 + (1 - lecture_ratio) * 30)
        reason = f"讲课/PPT占比{lecture_ratio:.0%}，不适合训练"
    else:
        decision, quality = "丢弃", 30
        reason = f"内容混杂，超声/医学影像仅{us_ratio:.0%}"

    anatomy_list = []
    for r in frame_results:
        a = r.get("解剖部位")
        if a and a != "null" and a not in anatomy_list:
            anatomy_list.append(a)

    modalities = [r.get("超声模态", "无") for r in frame_results if r.get("超声模态") != "无"]
    primary_modality = max(set(modalities), key=modalities.count) if modalities else "无"

    return {
        "超声帧占比": round(us_ratio, 2),
        "帧类型统计": type_counts,
        "主要模态": primary_modality,
        "解剖部位": anatomy_list,
        "质量评分": quality,
        "决策": decision,
        "判断理由": reason,
    }


# ============================================================================
# 报告管理 (边跑边保存)
# ============================================================================

def load_existing_report(report_path: Path) -> dict:
    """加载已有的报告（用于断点续跑）"""
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"videos": [], "summary": {}}


def save_report(report_path: Path, results: list, input_dir: str, total_time: float):
    """保存报告到文件"""
    keep = [r for r in results if r.get("决策") == "保留"]
    trim = [r for r in results if r.get("决策") == "需要裁剪"]
    discard = [r for r in results if r.get("决策") == "丢弃"]
    errors = [r for r in results if r.get("决策") == "错误"]

    report = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "input_dir": input_dir,
        "total_videos": len(results),
        "summary": {
            "保留": len(keep),
            "需要裁剪": len(trim),
            "丢弃": len(discard),
            "错误": len(errors),
        },
        "total_time_sec": round(total_time, 1),
        "videos": results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ============================================================================
# 批量处理 (边跑边保存 + 断点续跑)
# ============================================================================

def batch_filter(input_dir: str, num_frames: int = NUM_FRAMES,
                 max_videos: int = None, timeout: int = DEFAULT_TIMEOUT):
    """批量筛选目录下所有视频"""
    global _partial_frame_results

    # 查找所有视频
    videos = []
    for ext in ["*.mp4", "*.webm", "*.mkv", "*.avi"]:
        videos.extend(glob.glob(f"{input_dir}/**/{ext}", recursive=True))
    videos = sorted(videos)

    if not videos:
        print(f"❌ 在 {input_dir} 中未找到视频文件")
        return

    if max_videos:
        videos = videos[:max_videos]

    # 报告路径 (固定名称，方便断点续跑)
    report_path = Path(input_dir).parent / "vlm_filter_report.json"

    # 加载已有报告 (断点续跑)
    existing = load_existing_report(report_path)
    processed_ids = set(v["video_id"] for v in existing.get("videos", []))
    results = existing.get("videos", [])

    # 过滤掉已处理的
    remaining = [v for v in videos if Path(v).stem not in processed_ids]

    print(f"📂 输入目录: {input_dir}")
    print(f"📹 总视频: {len(videos)} | 已处理: {len(processed_ids)} | 待处理: {len(remaining)}")
    print(f"🎞️  每视频采样帧数: {num_frames}")
    print(f"💾 报告路径: {report_path} (边跑边保存)")
    print(f"⏳ 单视频超时: {timeout}s（超时后基于已分析帧判断）")
    print("=" * 70)

    if not remaining:
        print("✅ 所有视频已处理完毕!")
        print_summary(results)
        return

    # 加载模型
    model, processor, device = load_model()

    # 进度条
    pbar = tqdm(total=len(remaining), desc="🔍 VLM筛选", unit="video") if tqdm else None
    total_start = time.time()

    for idx, video_path in enumerate(remaining, 1):
        video_id = Path(video_path).stem
        category = Path(video_path).parent.name

        if pbar:
            pbar.set_postfix_str(f"{category}/{video_id[:12]}")

        print(f"\n[{idx}/{len(remaining)}] {category}/{video_id}")

        _partial_frame_results = []
        frame_results = []
        video_info = None

        try:
            # 设置超时 (仅Unix)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)

            t0 = time.time()

            # 采样帧
            frames, video_info = sample_frames(video_path, num_frames)

            # VLM 分析（逐帧，保存部分结果）
            frame_results = run_stage1(model, processor, device, frames)
            _partial_frame_results = frame_results

            # 规则判断
            decision = rule_based_decision(frame_results)

            elapsed = time.time() - t0

            # 取消超时
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            result = {
                "video_id": video_id,
                "video_path": video_path,
                "category": category,
                "duration_sec": round(video_info["duration"], 1) if video_info else 0,
                "process_time_sec": round(elapsed, 1),
                "analyzed_frames": len(frame_results),
                "total_frames_requested": num_frames,
                **decision,
            }

            icon = {"保留": "✅", "需要裁剪": "⚠️", "丢弃": "❌"}.get(decision["决策"], "?")
            print(f"   {icon} [{decision['质量评分']}/100] {decision['决策']} ({elapsed:.0f}s) [{len(frame_results)}/{num_frames}帧]")

        except TimeoutError:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            elapsed = time.time() - t0

            # 超时但已有部分帧结果 → 基于已有结果做判断
            if len(_partial_frame_results) >= 3:
                decision = rule_based_decision(_partial_frame_results)
                result = {
                    "video_id": video_id,
                    "video_path": video_path,
                    "category": category,
                    "duration_sec": round(video_info["duration"], 1) if video_info else 0,
                    "process_time_sec": round(elapsed, 1),
                    "analyzed_frames": len(_partial_frame_results),
                    "total_frames_requested": num_frames,
                    "超时但有部分结果": True,
                    **decision,
                }
                icon = {"保留": "✅", "需要裁剪": "⚠️", "丢弃": "❌"}.get(decision["决策"], "?")
                print(f"   ⏰ 超时但有{len(_partial_frame_results)}帧结果 → {icon} [{decision['质量评分']}/100] {decision['决策']}")
            else:
                print(f"   ⏰ 超时跳过 (>{timeout}s, 仅{len(_partial_frame_results)}帧)")
                result = {
                    "video_id": video_id, "video_path": video_path,
                    "category": category, "决策": "错误",
                    "判断理由": f"处理超时(>{timeout}s), 仅分析{len(_partial_frame_results)}帧",
                    "质量评分": 0,
                }

        except Exception as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            print(f"   ❌ 错误: {e}")
            result = {
                "video_id": video_id, "video_path": video_path,
                "category": category, "决策": "错误",
                "判断理由": str(e)[:200], "质量评分": 0,
            }

        # 追加结果并立即保存
        results.append(result)
        save_report(report_path, results, input_dir, time.time() - total_start)

        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    total_elapsed = time.time() - total_start
    print_summary(results, total_elapsed)


def print_summary(results: list, elapsed: float = 0):
    """打印汇总"""
    print("\n" + "=" * 70)
    print("📊 批量筛选完成!")
    print("=" * 70)
    if elapsed:
        print(f"   总耗时: {elapsed/60:.1f} 分钟 | 平均: {elapsed/max(len(results),1):.1f}s/视频")

    keep = [r for r in results if r.get("决策") == "保留"]
    trim = [r for r in results if r.get("决策") == "需要裁剪"]
    discard = [r for r in results if r.get("决策") == "丢弃"]
    errors = [r for r in results if r.get("决策") == "错误"]

    print(f"\n   ✅ 保留:     {len(keep)}")
    print(f"   ⚠️  裁剪:     {len(trim)}")
    print(f"   ❌ 丢弃:     {len(discard)}")
    if errors:
        print(f"   🚫 错误:     {len(errors)}")

    if keep:
        print(f"\n   保留的视频:")
        for r in sorted(keep, key=lambda x: x.get("质量评分", 0), reverse=True)[:20]:
            print(f"     [{r['质量评分']}] {r.get('category','?')}/{r['video_id']}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量VLM筛选超声视频")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="单视频超时秒数")
    args = parser.parse_args()

    batch_filter(args.input_dir, args.num_frames, args.max_videos, args.timeout)