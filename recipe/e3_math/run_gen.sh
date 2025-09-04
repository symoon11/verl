set -x

experiment_name=grpo_qwen3_1.7b_easy
step=100
seed=0

python -m recipe.e3_math.main_gen \
    experiment_name=${experiment_name} \
    step=${step} \
    seed=${seed} $@
