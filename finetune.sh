#!/bin/bash

OPTION=$1

if [ -z "$OPTION" ]; then
    echo "Lỗi: Bạn chưa nhập tham số."
    echo "Cách dùng: ./script.sh [lsnet_t|lsnet_s|lsnet_b]"
    exit 1
fi

case $OPTION in
lsnet_t) python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 --use_env main.py --model lsnet_t --data-path dataset/vnfood_combined_dataset \
    --finetune https://huggingface.co/jameslahm/lsnet/resolve/main/lsnet_t.pth --output_dir finetuned/lsnet_t_finetuned.pth
;;
lsnet_s) python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 --use_env main.py --model lsnet_s --data-path dataset/vnfood_combined_dataset \
    --finetune https://huggingface.co/jameslahm/lsnet/resolve/main/lsnet_s.pth --output_dir finetuned/lsnet_t_finetuned.pth
;;
lsnet_b) python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 --use_env main.py --model lsnet_b --data-path dataset/vnfood_combined_dataset \
    --finetune https://huggingface.co/jameslahm/lsnet/resolve/main/lsnet_t.pth --output_dir finetuned/lsnet_b_finetuned.pth --weight-decay 0.05
;;
# For training with distillation, please add `--distillation-type hard`
# For LSNet-B, please add `--weight-decay 0.05`