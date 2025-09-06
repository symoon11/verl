set -x

model_name=Qwen/Qwen3-1.7B

python -m recipe.e3_math.main_eval --config-name eval_base.yaml \
    model_name=${model_name} $@
