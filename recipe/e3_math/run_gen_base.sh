set -x

model_name=Qwen/Qwen3-1.7B
seed=0

python -m recipe.e3_math.main_gen --config-name gen_base.yaml \
    model_name=${model_name} \
    seed=${seed} $@
