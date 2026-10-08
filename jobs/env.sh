BASE_8B=${BASE_8B:-Qwen/Qwen3-8B}
BASE_4B=${BASE_4B:-Qwen/Qwen3-4B}
BASE_17B=${BASE_17B:-Qwen/Qwen3-1.7B}
STEPS="5 10 15 20 25 30 35 40 45 50 55 60 65 67"

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_vars() {
  case "$1" in
    8B)   BASE=$BASE_8B;  M=8b; EXP=${EXP_8B:-qwen3_8b_grpo_nothink};    AD=checkpoints/walk_variants/gate_adapters;      PFX=grpos2_s; OUT=analysis_out/think_leak/qwen_s2 ;;
    4B)   BASE=$BASE_4B;  M=4b; EXP=${EXP_4B:-qwen3_4b_grpo_nothink};    AD=checkpoints/walk_variants/q_nothink_adapters; PFX=s;        OUT=analysis_out/think_leak4b/nothink ;;
    1.7B) BASE=$BASE_17B; M=17; EXP=${EXP_17B:-qwen3_1p7b_grpo_nothink}; AD=checkpoints/walk_variants/q_nothink17_adapters; PFX=s;      OUT=analysis_out/think_leak4b/nothink17 ;;
    *) echo "unknown model $1"; exit 2 ;;
  esac
  V=""
  for s in $STEPS; do [ -f $AD/$PFX$s/adapter_model.safetensors ] && V="$V s$s=$AD/$PFX$s"; done
  RL=$AD/${PFX}67
}
