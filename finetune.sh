#!/bin/bash

OPTION=$1
GPUS=$2

if [ -z "$OPTION" ]; then
    echo "Lỗi: Bạn chưa nhập tham số."
    echo "Cách dùng: ./script.sh [lsnet_t|lsnet_s|lsnet_b]"
    exit 1
fi

case $OPTION in
lsnet_t) python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 12345 --use_env main.py --model lsnet_t --data-set VNFood --data-path dataset/vnfood-30-100/vnfood_combined_dataset \
    --finetune https://huggingface.co/jameslahm/lsnet/resolve/main/lsnet_t.pth --output_dir finetuned/lsnet_t --epochs 30
;;
lsnet_s) python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 12345 --use_env main.py --model lsnet_s --data-set VNFood --data-path dataset/vnfood-30-100/vnfood_combined_dataset \
    --finetune https://huggingface.co/jameslahm/lsnet/resolve/main/lsnet_s.pth --output_dir finetuned/lsnet_s --epochs 30
;;
lsnet_b) python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 12345 --use_env main.py --model lsnet_b --data-set VNFood --data-path dataset/vnfood-30-100/vnfood_combined_dataset \
    --finetune https://huggingface.co/jameslahm/lsnet/resolve/main/lsnet_b.pth --output_dir finetuned/lsnet_b --weight-decay 0.05 --epochs 20
;;
esac