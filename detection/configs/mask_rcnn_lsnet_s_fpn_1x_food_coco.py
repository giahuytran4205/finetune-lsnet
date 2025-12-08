_base_ = './mask_rcnn_lsnet_s_fpn_1x_coco.py'

dataset_type = 'CocoDataset'
data_root = 'dataset/UECFOODPIXCOMPLETE_COCO/'

model = dict(
    backbone=dict(
        pretrained=None,
    ),
    bbox_head=dict(
        num_classes=102
    )
)

# Modify dataset related settings
classes = ('rice', 'eels on rice', 'pilaf', "chicken-'n'-egg on rice", 'pork cutlet on rice', 'beef curry', 'sushi', 'chicken rice', 'fried rice', 'tempura bowl', 'bibimbap', 'toast', 'croissant', 'roll bread', 'raisin bread', 'chip butty', 'hamburger', 'pizza', 'sandwiches', 'udon noodle', 'tempura udon', 'soba noodle', 'ramen noodle', 'beef noodle', 'tensin noodle', 'fried noodle', 'spaghetti', 'Japanese-style pancake', 'takoyaki', 'gratin', 'sauteed vegetables', 'croquette', 'grilled eggplant', 'sauteed spinach', 'vegetable tempura', 'miso soup', 'potage', 'sausage', 'oden', 'omelet', 'ganmodoki', 'jiaozi', 'stew', 'teriyaki grilled fish', 'fried fish', 'grilled salmon', 'salmon meuniere', 'sashimi', 'grilled pacific saury', 'sukiyaki', 'sweet and sour pork', 'lightly roasted fish', 'steamed egg hotchpotch', 'tempura', 'fried chicken', 'sirloin cutlet', 'nanbanzuke', 'boiled fish', 'seasoned beef with potatoes', 'hambarg steak', 'beef steak', 'dried fish', 'ginger pork saute', 'spicy chili-flavored tofu', 'yakitori', 'cabbage roll', 'rolled omelet', 'egg sunny-side up', 'fermented soybeans', 'cold tofu', 'egg roll', 'chilled noodle', 'stir-fried beef and peppers', 'simmered pork', 'boiled chicken and vegetables', 'sashimi bowl', 'sushi bowl', 'fish-shaped pancake with bean jam', 'shrimp with chill source', 'roast chicken', 'steamed meat dumpling', 'omelet with fried rice', 'cutlet curry', 'spaghetti meat sauce', 'fried shrimp', 'potato salad', 'green salad', 'macaroni salad', 'Japanese tofu and vegetable chowder', 'pork miso soup', 'chinese soup', 'beef bowl', 'kinpira-style sauteed burdock', 'rice ball', 'pizza toast', 'dipping noodles', 'hot dog', 'french fries', 'mixed rice', 'goya chanpuru', 'others', 'beverage')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        classes=classes,
        ann_file=data_root + 'train/annotation.json',
        img_prefix=data_root),
    val=dict(
        classes=classes,
        ann_file=data_root + 'test/annotation.json',
        img_prefix=data_root),
    test=dict(
        classes=classes,
        ann_file=data_root + 'test/annotation.json',
        img_prefix=data_root))

load_from = './pretrain/lsnet_s_maskrcnn.pth'