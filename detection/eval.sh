# For RetinaNet
bash ./dist_test.sh configs/retinanet_lsnet_t_fpn_1x_coco.py pretrain/lsnet_t_retinanet.pth 8 --eval bbox --out results.pkl

# For Mask R-CNN
bash ./dist_test.sh configs/mask_rcnn_lsnet_t_fpn_1x_coco.py pretrain/lsnet_t_maskrcnn.pth 8 --eval bbox segm --out results.pkl

bash ./dist_test.sh configs/mask_rcnn_lsnet_b_fpn_1x_food_coco.py checkpoints/pretrain/mask_rcnn_lsnet_b_finetuned.pth 1 --eval bbox segm proposal --work-dir ./eval --show --show-dir ./eval