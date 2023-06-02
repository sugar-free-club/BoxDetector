import cv2

from detector import Detector


detector = Detector('./rtmdet_box.plan', (640, 640), 'rtmdet')
cls_dict_rtmdet = {
    0: 'box'
}
img_path = './test_imgs_complex/1004.png'
img = cv2.imread(img_path)
boxes, confs, clss = detector.detect(img, 0.3)