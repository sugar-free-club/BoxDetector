# -*- coding: utf-8 -*-
import os
import time
import argparse
import subprocess

import cv2

from detector import Detector
from utils.visualization import BBoxVisualization


cls_dict_ssd = {
    0: 'bg',
    1: 'box'
}

cls_dict_rtmdet = {
    0: 'box'
}


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('network_type')
    parser.add_argument('trt_plan')
    parser.add_argument('hw')
    
    args = parser.parse_args()
    return args


def detect_video(video, trt_engine, conf_th, vis, result):
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
            boxes, confs, clss = trt_engine.detect(img, conf_th)
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


def detect_dir(dir, detector, conf_th, vis, cls_dict):
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
            img = vis.draw_bboxes(img, boxes, confs, clss)
            cv2.imwrite("./result_img/"+str(i), img)
            new_file = open("./mAP/input/detection-results/"+os.path.splitext(i)[0]+".txt",'w+')
            if len(clss)>0:
                for count in range(0, len(clss)):
                    new_file.write(cls_dict[clss[count]]+" ")
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


def bench_fps(detector, test_file, network_type):
    result_file_name = "./results/result_video.mp4"
    if network_type == "rtmdet":
        cls_dict = cls_dict_rtmdet
    else:
        cls_dict = cls_dict_ssd
    video = cv2.VideoCapture(test_file)
    vis = BBoxVisualization(cls_dict)
    print("start benching fps!")
    
    fps = detect_video(video, detector, conf_th=0.4, vis=vis, result=result_file_name)
    
    video.release()
    cv2.destroyAllWindows()
    return fps


def bench_map(detector, test_set, network_type):
    if network_type == "rtmdet":
        cls_dict = cls_dict_rtmdet
    else:
        cls_dict = cls_dict_ssd
    vis = BBoxVisualization(cls_dict)
    print("start benching map!")
    mAP = detect_dir(test_set, detector, conf_th=0.3, vis=vis, cls_dict=cls_dict)
    return mAP


def main():
    args = parse_args()
    
    nt = args.network_type
    engine_path = args.trt_plan
    hw = int(args.hw)
    detector = Detector(engine_path, (hw, hw), nt)
    fps = bench_fps(detector, './box_test_video.mp4', nt)
    mAP = bench_map(detector, './test_imgs/', nt)
    print("Benchmark finished.")
    print("FPS: ", str(fps))
    print("mAP: ", str(mAP))
    

if __name__=="__main__":
    main()