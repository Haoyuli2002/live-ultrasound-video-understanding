# Sourced after REPO_ROOT is established. No execution or job submission here.
MODEL_TYPE="${MODEL_TYPE:-qwen2_vl}"
case "$MODEL_TYPE" in
    qwen2_vl)
        BACKBONE="${BACKBONE:-llm/Qwen2-VL-7B-Instruct}"
        BACKBONE_PRECISION="${BACKBONE_PRECISION:-fp16}"
        IMAGE_SIZE="${IMAGE_SIZE:-224}"
        ;;
    qwen3_5)
        BACKBONE="${BACKBONE:-Qwen/Qwen3.5-2B}"
        BACKBONE_PRECISION="${BACKBONE_PRECISION:-bf16}"
        IMAGE_SIZE="${IMAGE_SIZE:-256}"
        ;;
    *) echo "ERROR: MODEL_TYPE must be qwen2_vl or qwen3_5" >&2; exit 1 ;;
esac
