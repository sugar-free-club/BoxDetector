# -*- coding: utf-8 -*-
import os
import time
import argparse
import subprocess

import cv2

from detector import Detector
from utils.visualization import BBoxVisualization


INPUT_HW = (300, 300)
cls_dict = {
    0: 'bg',
    1: 'box'
}


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('trt_plan')
    
    args = parser.parse_args()
    return args


def detect_video(video, trt_ssd, conf_th, vis, result):
    full_scrn = False
    fps = 0.0
    tic = time.time()
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    #print(str(frame_width)+str(frame_height))
    ##定义输入编码
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    videoWriter = cv2.VideoWriter(result, fourcc, fps, (frame_width,frame_height))
    ##开始循环检测，并将结果写到result.mp4中
    while True:
        ret,img = video.read()
        if img is not None:
            boxes, confs, clss = trt_ssd.detect(img, conf_th)
            #print("boxes,confs,clss: "+ str(boxes)+" "+ str(confs)+" "+str(clss))
            img = vis.draw_bboxes(img, boxes, confs, clss)
            videoWriter.write(img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
            #print("\rfps: "+str(fps),end="")
        else:
            break
    return fps


def detect_dir(dir, detector, conf_th, vis):
    dirs = os.listdir(dir)
    print(dir)
    remove_old_detection_results = subprocess.Popen('rm ./mAP/input/detection-results/*',
                                                    shell=True,
                                                    stdout=subprocess.PIPE,
                                                    stderr=subprocess.STDOUT)
    for i in dirs:
        if os.path.splitext(i)[1] == ".png":
            full_scrn = False
            #print("val/images/"+str(i))
            img = cv2.imread(dir+str(i))
            boxes, confs, clss = detector.detect(img, conf_th)
            new_file = open("./mAP/input/detection-results/"+os.path.splitext(i)[0]+".txt",'w+')
            if len(clss)>0:
                for count in range(0, len(clss)):
                    if clss[count] == 1:
                        new_file.write("box ")
                    new_file.write(str(confs[count])+" ")
                    new_file.write(str(boxes[count][0])+" ")
                    new_file.write(str(boxes[count][1])+" ")
                    new_file.write(str(boxes[count][2])+" ")
                    new_file.write(str(boxes[count][3])+" \n")

    mAP = subprocess.Popen('python3.7 ./mAP/main.py -np -na',
                           shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    mAP_result = str(mAP.stdout.read())
    return mAP_result.split('\\n')[-3]


def bench_fps(detector, test_file):
    result_file_name = "./results/result_video.mp4"
    video = cv2.VideoCapture(test_file)
    vis = BBoxVisualization(cls_dict)
    print("start benching fps!")
    
    fps = detect_video(video, detector, conf_th=0.4, vis=vis, result=result_file_name)
    
    video.release()
    cv2.destroyAllWindows()
    return fps


def bench_map(detector, test_set):
    vis = BBoxVisualization(cls_dict)
    print("start benching map!")
    mAP = detect_dir(test_set, detector, conf_th=0.3, vis=vis)
    return mAP

def detect_one(img, detector, conf_th, vis):
    full_scrn = False
    tic = time.clock()
    ##开始检测，并将结果写到result.jpg中
    boxes, confs, clss = detector.detect(img, conf_th)
    toc = time.clock()
    curr_fps = (toc - tic)
    print("boxes: "+str(boxes))
    print("clss: "+str(clss))
    print("confs: "+str(confs))
    img = vis.draw_bboxes(img, boxes, confs, clss)
    result_path = "./uploads/result.jpg"
    cv2.imwrite(result_path,img)        
    print("time: "+str(curr_fps)+"(sec)")
    return result_path
    
    
def detect_your_image(detector, filename):    
    img = cv2.imread(filename)
    vis = BBoxVisualization(cls_dict)
    print("start detection!")
    filepath = detect_one(img, detector, conf_th=0.3, vis=vis)
    cv2.destroyAllWindows()
    return filepath
    #print("ok")


def main():
    args = parse_args()
    
    engine_path = args.trt_plan
    detector = Detector(engine_path, INPUT_HW)
    fps = bench_fps(detector, './box_test_video.mp4')
    mAP = bench_map(detector, './test_imgs/')
    re  = detect_your_image(detector, '100.png')
    print("Benchmark finished.")
    print("FPS: ", str(fps))
    print("mAP: ", str(mAP))
    

if __name__=="__main__":
    main()