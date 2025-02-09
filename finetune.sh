conda run --name prime_net_env --no-capture-output python3 -u finetune.py \
 --dataset french --nan-pct 0.5 --num-past 48 --num-fut 30 --abl-code E --niters 100 \
 --batch-size 64 --seed 42 --task regression --lr 0.0001 --dev
