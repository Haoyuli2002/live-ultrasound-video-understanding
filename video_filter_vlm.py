"""
超声视频 VLM 筛选器
====================
Stage 1: Qwen2-VL-2B (本地, MPS) → 逐帧分析 → 8个帧级JSON
Stage 2: DeepSeek API → 综合8帧结果 → 视频级最终判断

使用方法:
    python video_filter_vlm.py --video PATH_TO_VIDEO --deepseek-key YOUR_KEY

requirements:
    pip install torch torchvision transformers accelerate qwen-vl-utils
    pip install opencv-python numpy Pillow openai
"""

import json
import time
import re
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_VIDEO = "UltrasoundCrawler_KeyCode_20260323_v2/output/20260502_152417_youtube/media/organ_system_lecture/adJepggTLd4.mp4"
NUM_FRAMES = 8

# Stage 1: VLM Prompt
FRAME_PROMPT = """分析这张医学影像图片。请参考示例格式回答。

示例1 - 如果是超声画面：
{"帧类型":"超声画面","超声模态":"B超","标注程度":"仅机器参数","有人脸":false,"有文字覆盖":false,"解剖部位":"肾脏","置信度":0.9,"简要描述":"B超下右肾纵切面，可见正常肾脏轮廓和皮髓质分界"}

示例2 - 如果是PPT幻灯片：
{"帧类型":"PPT幻灯片","超声模态":"无","标注程度":"重度人工标注","有人脸":false,"有文字覆盖":true,"解剖部位":null,"置信度":0.95,"简要描述":"白色背景的教学幻灯片，展示肝脏解剖图和文字说明"}

示例3 - 如果是讲课画面：
{"帧类型":"讲课画面","超声模态":"无","标注程度":"无标注","有人脸":true,"有文字覆盖":false,"解剖部位":null,"置信度":0.9,"简要描述":"一位医生面对镜头讲解超声扫查技巧"}

现在请分析当前图片，只输出一个JSON："""

# Stage 2: DeepSeek 综合判断的 Prompt
STAGE2_PROMPT_TEMPLATE = """你是超声视频数据集的质量控制专家。以下是一个视频中均匀采样8帧的VLM分析结果。
请综合判断该视频是否适合用于超声AI模型训练。

视频信息：
- ID: {video_id}
- 时长: {duration:.1f}秒
- 帧率: {fps:.0f} FPS

8帧分析结果：
{frame_results_json}

请输出综合判断JSON（只输出JSON，不要其他文字）：
{{"视频ID": "{video_id}", "是超声视频": true/false, "超声帧占比": 0.0-1.0, "主要内容": "实时扫查/教学讲座/PPT展示/混合内容", "主要模态": "B超/彩色多普勒/M模式/混合", "标注程度": "无标注/仅机器参数/轻度人工标注/重度人工标注", "检测到的解剖部位": [], "质量评分": 0-100, "决策": "保留/需要去标注/丢弃", "判断理由": "简要说明"}}"""


# ============================================================================
# Stage 0: 视频帧采样
# ============================================================================

def sample_frames(video_path: str, num_frames: int = NUM_FRAMES):
    """从视频中均匀采样帧 (排除首尾5%)"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total_frames / fps

    print(f"📹 {Path(video_path).stem}")
    print(f"   时长: {duration:.1f}s ({duration/60:.1f}min) | FPS: {fps:.0f} | 总帧: {total_frames}")

    start_idx = int(total_frames * 0.05)
    end_idx = int(total_frames * 0.95)
    indices = np.linspace(start_idx, end_idx, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append({
                "image": Image.fromarray(rgb),
                "timestamp": round(idx / fps, 2),
                "frame_idx": int(idx),
            })
    cap.release()
    print(f"   ✅ 采样 {len(frames)} 帧")
    return frames, {"duration": duration, "fps": fps, "total_frames": total_frames}


# ============================================================================
# Stage 1: Qwen2-VL 逐帧分析
# ============================================================================

def load_model():
    """加载 Qwen2-VL-2B 模型"""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = "cuda"
        print("✅ CUDA GPU")
    else:
        device = "cpu"
        print("⚠️ CPU (慢)")

    print("加载 Qwen2-VL-2B-Instruct ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
    )
    if device in ("mps", "cpu"):
        model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    print("✅ 模型加载完成!")
    return model, processor, device


def parse_json_from_text(text: str) -> dict:
    """从VLM输出中提取JSON"""
    # 直接解析
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # 提取代码块
    for pattern in [r'```json\s*(.+?)```', r'```\s*(.+?)```']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                continue
    # 提取第一个 {...}
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # 嵌套 JSON
    m = re.search(r'\{.+\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"帧类型": "解析失败", "原始输出": text[:300]}


def analyze_frame_vlm(model, processor, device, image: Image.Image, frame_num: int, total: int):
    """单帧VLM分析"""
    import torch
    from qwen_vl_utils import process_vision_info

    t0 = time.time()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": FRAME_PROMPT}
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    result = parse_json_from_text(raw)
    elapsed = time.time() - t0

    desc = result.get('简要描述', result.get('原始输出', '?'))[:35]
    print(f"  [{frame_num:2d}/{total}] {elapsed:.1f}s | {result.get('帧类型', '?')} | {desc}")
    return result, raw


def run_stage1(model, processor, device, frames: list) -> list:
    """运行 Stage 1: 逐帧VLM分析"""
    print("\n🔍 Stage 1: Qwen2-VL 逐帧分析")
    print("=" * 60)

    frame_results = []
    total_start = time.time()

    for i, f in enumerate(frames):
        result, raw = analyze_frame_vlm(model, processor, device, f["image"], i + 1, len(frames))
        result["timestamp"] = f["timestamp"]
        result["_raw_output"] = raw
        frame_results.append(result)

    total_elapsed = time.time() - total_start
    print(f"\n⏱️  Stage 1 完成! 总耗时: {total_elapsed:.1f}s (平均 {total_elapsed/len(frames):.1f}s/帧)")
    return frame_results


# ============================================================================
# Stage 2: DeepSeek 综合判断
# ============================================================================

def run_stage2(frame_results: list, video_info: dict, video_id: str, api_key: str, 
               provider: str = "openai") -> dict:
    """运行 Stage 2: LLM 综合判断
    
    Args:
        provider: "openai" (默认, GPT-4o-mini) 或 "deepseek"
    """
    from openai import OpenAI

    if provider == "deepseek":
        print("\n🤖 Stage 2: DeepSeek 综合判断")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        model_name = "deepseek-chat"
    else:
        print("\n🤖 Stage 2: GPT-4o-mini 综合判断")
        client = OpenAI(api_key=api_key)
        model_name = "gpt-4o-mini"
    
    print("=" * 60)

    # 准备帧结果 (去掉原始输出，太长)
    clean_results = []
    for r in frame_results:
        clean = {k: v for k, v in r.items() if k != "_raw_output"}
        clean_results.append(clean)

    # 构造 prompt
    prompt = STAGE2_PROMPT_TEMPLATE.format(
        video_id=video_id,
        duration=video_info["duration"],
        fps=video_info["fps"],
        frame_results_json=json.dumps(clean_results, ensure_ascii=False, indent=2),
    )

    # 调用 LLM
    t0 = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )
    elapsed = time.time() - t0

    raw_output = response.choices[0].message.content
    result = parse_json_from_text(raw_output)

    print(f"   耗时: {elapsed:.1f}s")
    print(f"   Token使用: {response.usage.total_tokens} tokens")
    print(f"\n📊 综合判断结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


# ============================================================================
# 主流程
# ============================================================================

def analyze_video(video_path: str, deepseek_key: str = None, num_frames: int = NUM_FRAMES):
    """完整分析一个视频"""
    video_id = Path(video_path).stem

    # Stage 0: 采样帧
    frames, video_info = sample_frames(video_path, num_frames)

    # Stage 1: VLM 逐帧分析
    model, processor, device = load_model()
    frame_results = run_stage1(model, processor, device, frames)

    # 打印 Stage 1 统计
    print("\n📋 Stage 1 统计:")
    types = [r.get("帧类型", "?") for r in frame_results]
    for t in set(types):
        print(f"  {t}: {types.count(t)}/{len(types)}帧")

    # Stage 2: DeepSeek 综合判断
    if deepseek_key and deepseek_key != "your-deepseek-api-key-here":
        final_result = run_stage2(frame_results, video_info, video_id, deepseek_key)
    else:
        print("\n⚠️ 未提供 DeepSeek API Key，跳过 Stage 2")
        print("   设置方式: --deepseek-key YOUR_KEY")
        final_result = None

    # 保存报告
    report = {
        "video_id": video_id,
        "video_path": str(video_path),
        "video_info": video_info,
        "stage1_frame_results": [{k: v for k, v in r.items() if k != "_raw_output"} for r in frame_results],
        "stage2_final_result": final_result,
    }

    report_path = Path(video_path).parent / f"{video_id}_vlm_filter_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 报告已保存: {report_path}")

    return report


# ============================================================================
# CLI 入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="超声视频 VLM 筛选器")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO, help="视频文件路径")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="采样帧数 (默认8)")
    parser.add_argument("--deepseek-key", type=str, default=None, help="DeepSeek API Key")
    args = parser.parse_args()

    analyze_video(args.video, args.deepseek_key, args.num_frames)