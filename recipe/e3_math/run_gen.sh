set -x

DATA_PATH=$HOME/data/e3-math/test.parquet
OUTPUT_PATH=$HOME/data/e3-math/grpo_qwen3_1.7b_easy/global_step_100/gen_test.parquet
MODEL_PATH=checkpoints/verl_e3_math/grpo_qwen3_1.7b_easy/global_step_100/actor/huggingface

python -m recipe.e3_math.main_gen \
    data.path=$DATA_PATH \
    data.output_path=$OUTPUT_PATH \
    model.path=$MODEL_PATH \
