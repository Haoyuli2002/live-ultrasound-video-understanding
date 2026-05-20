"""Static configuration for crawl seeds and heuristic classifiers."""

YOUTUBE_SEARCH_TERMS = [
    "POCUS ultrasound scanning tutorial",
    "ultrasound probe position tutorial",
    "lung ultrasound scan protocol",
    "cardiac ultrasound scan tutorial",
    "abdominal ultrasound scanning tutorial",
    "ultrasound case reasoning",
    "ultrasound anatomy lecture",
    "bedside ultrasound interpretation",
    "emergency ultrasound tutorial",
    "Dr Sam's Imaging Library ultrasound",
    "POCUS 101 ultrasound",
]

YOUTUBE_CHANNEL_URLS = [
    "https://www.youtube.com/@POCUS101",
    "https://www.youtube.com/@DrSamsImagingLibrary",
]

BILIBILI_SEARCH_TERMS = [
    "超声 扫查 教学",
    "超声 探头 手法",
    "床旁超声 POCUS",
    "超声 病例 讲解",
    "超声 解剖 教学",
    "心脏 超声 扫查",
    "肺超声 扫查",
    "腹部 超声 扫查",
]

ULTRASOUND_KEYWORDS_EN = [
    "ultrasound",
    "sonography",
    "sonographic",
    "echography",
    "pocus",
    "probe",
    "transducer",
    "doppler",
    "b-mode",
    "scan",
    "scanning",
    "view",
]

ULTRASOUND_KEYWORDS_ZH = [
    "超声",
    "扫查",
    "探头",
    "切面",
    "床旁超声",
    "彩超",
    "多普勒",
    "声像图",
    "影像",
    "pocus",
]

VISUAL_HINTS_EN = [
    "image",
    "imaging",
    "view",
    "window",
    "scan",
    "probe",
    "findings",
    "clip",
    "cine",
]

VISUAL_HINTS_ZH = [
    "图像",
    "影像",
    "切面",
    "扫查",
    "探头",
    "显示",
    "回声",
    "声像图",
]

NEGATIVE_NON_VISUAL_HINTS = [
    "podcast",
    "audio only",
    "no image",
    "without image",
    "纯音频",
    "无画面",
    "仅音频",
]

CATEGORY_KEYWORDS_EN = {
    "扫查教学型": [
        "how to scan",
        "scanning technique",
        "probe position",
        "probe placement",
        "knobology",
        "ultrasound tutorial",
        "scan protocol",
        "exam technique",
        "view acquisition",
        "step by step",
    ],
    "病例讲解型": [
        "case",
        "clinical reasoning",
        "diagnosis",
        "case review",
        "findings",
        "differential",
        "pathology",
        "interpretation",
    ],
    "器官系统教学型": [
        "anatomy",
        "organ",
        "radiology lecture",
        "teaching lecture",
        "cardiac",
        "lung",
        "abdomen",
        "liver",
        "kidney",
        "thyroid",
        "vascular",
        "pelvic",
    ],
}

CATEGORY_KEYWORDS_ZH = {
    "扫查教学型": [
        "扫查教学",
        "怎么扫",
        "探头手法",
        "探头摆放",
        "切面定位",
        "实操",
        "操作流程",
        "扫查技巧",
        "床旁扫查",
    ],
    "病例讲解型": [
        "病例",
        "案例",
        "个案",
        "诊断思路",
        "病例分析",
        "鉴别诊断",
        "病理",
        "临床推理",
    ],
    "器官系统教学型": [
        "解剖",
        "器官",
        "系统",
        "课程",
        "讲座",
        "心脏",
        "肺",
        "肝",
        "胆",
        "胰",
        "脾",
        "肾",
        "甲状腺",
        "血管",
        "乳腺",
        "妇产",
        "泌尿",
    ],
}

