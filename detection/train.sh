# For RetinaNet
bash ./dist_train.sh configs/retinanet_lsnet_t_fpn_1x_food_coco.py 4

# For Mask R-CNN
bash ./dist_train.sh configs/mask_rcnn_lsnet_t_fpn_1x_coco.py 8
