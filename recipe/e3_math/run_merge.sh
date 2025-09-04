set -x

MODEL_PATH=checkpoints/verl_e3_math/grpo_qwen3_1.7b_easy/global_step_100/actor

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $MODEL_PATH \
    --target_dir $MODEL_PATH/huggingface
