na=$1
np=$2
nf=$3

conda run --name py3_11+torch2_6 --no-capture-output python3 -u finetune.py \
 --dataset french --nan-pct $na --num-past $np --num-fut $nf --abl-code E --niters 100 \
 --batch-size 64 --seed 42 --task regression --lr 0.0001 --dev
