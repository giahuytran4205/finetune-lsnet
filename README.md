# [LSNet: See Large, Focus Small](https://arxiv.org/abs/2503.23135)

This is modified code from [https://github.com/THU-MIG/lsnet](https://github.com/THU-MIG/lsnet) to fine-tune model for my project.

## Setup
### For Classification
In the project directory, run
```bash
bash ./setup.sh
```

### For Detection & Instance Segmentation
Move to the folder `detection`
```bash
cd detection
```
Run
```bash
bast ./setup.sh
```

## Training (Fine-tuning)
### For Classification
Training lsnet_t on an 8-GPU machine
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model lsnet_t --data-path dataset/vnfood-30-100/vnfood-30-100 --finetune https://huggingface.co/jameslahm/lsnet/resolve/main/lsnet_t.pth --epochs 40
# For LSNet-B, please add `--weight-decay 0.05
```

### For Detection and Instance Segmentation
Training on an 8-GPU machine
```bash
# If you already in this directory, please skip
cd detection

# For RetinaNet
bash ./dist_train.sh configs/retinanet_lsnet_t_fpn_1x_food_coco.py 8

# For Mask R-CNN
bash ./dist_train.sh configs/mask_rcnn_lsnet_t_fpn_1x_coco.py 8
```

## Testing
### For Classification
```bash
python main.py --eval --model lsnet_t --resume ./pretrain/lsnet_t_finetuned.pth --data-path dataset/vnfood-30-100/vnfood-30-100
```

### For Detection and Instance Segmentation
For RetinaNet
```bash
bash ./dist_test.sh configs/retinanet_lsnet_b_fpn_1x_coco.py pretrain/lsnet_b_retinanet.pth 8 --eval bbox --out results.pkl
```

For Mask R-CNN
```bash
bash ./dist_test.sh configs/mask_rcnn_lsnet_b_fpn_1x_coco.py pretrain/lsnet_b_maskrcnn.pth 8 --eval bbox segm --out results.pkl
```

Replace `8` with GPU count on your machine.