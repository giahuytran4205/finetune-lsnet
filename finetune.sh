python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 --use_env main.py --model lsnet_t --data-path dataset/vnfood_combined_dataset --dist-eval
# For training with distillation, please add `--distillation-type hard`
# For LSNet-B, please add `--weight-decay 0.05`