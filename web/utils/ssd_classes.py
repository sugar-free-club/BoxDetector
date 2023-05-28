"""ssd_classes.py

This file was modified from:
http://github.com/AastaNV/TRT_object_detection/blob/master/coco.py
"""

COCO_CLASSES_LIST = [
    'background',  # was 'unlabeled'
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

EGOHANDS_CLASSES_LIST = [
    'background',
    'hand',
]

Face_CLASSES_LIST = [
    'bg',
    'face_mask_wear_incorrect',
    'face_mask',
    'face',
]

Signs_CLASSES_LIST = [
    'mandatory',
    'prohibitory',
    'warning'
]

Traffic_CLASSES_LIST = [
    'bicycle',
    'vehicle',
    'pedestrian',
    'road_sign'
]

pdch_CLASSES_LIST = [
    'back',
    'cat',
    'dog',
    'horse',
    'person'
]

garbage_CLASSES_LIST = [
    'bg',
    'banane',
    'bottle',
    'cardboard'
]

box_CLASSES_LIST = [
    'bg',
    'box'
]
def get_cls_dict(model):
    #print("model name: "+str(model))
    """Get the class ID to name translation dictionary."""
    if model == 'coco':
        cls_list = COCO_CLASSES_LIST
    elif model == 'egohands':
        cls_list = EGOHANDS_CLASSES_LIST
    elif model == 'face':
        cls_list = Face_CLASSES_LIST
    elif model == 'signs':
        cls_list = Signs_CLASSES_LIST
    elif model == 'traffic':
        cls_list = Traffic_CLASSES_LIST
    elif model == '5th':
        cls_list = pdch_CLASSES_LIST
    elif model == 'garbage':
        cls_list = garbage_CLASSES_LIST
    elif model == 'box':
        cls_list = box_CLASSES_LIST
    else:
        raise ValueError('Bad model name')
    return {i: n for i, n in enumerate(cls_list)}
