# -*- coding: UTF-8 -*-
#1
import sys
import os
import subprocess
import time
import argparse
import cv2
import numpy as np
from utils.ssd_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
import ctypes
import tensorrt as trt
import pycuda.driver as cuda


#2
def _preprocess_trt(img, shape=(300, 300)):
    """Preprocess an image before TRT SSD inferencing."""
    img = cv2.resize(img, shape)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    return img


def _postprocess_trt(img, output, conf_th, output_layout):
    """Postprocess TRT SSD output."""
    img_h, img_w, _ = img.shape
    boxes, confs, clss = [], [], []
    #print(len(output))
    for prefix in range(0, len(output), output_layout):
        index = int(output[prefix+0])
        conf = float(output[prefix+2])
        if conf < conf_th:
            continue
        x1 = int(output[prefix+3] * img_w)
        y1 = int(output[prefix+4] * img_h)
        x2 = int(output[prefix+5] * img_w)
        y2 = int(output[prefix+6] * img_h)
        cls = int(output[prefix+1])
        boxes.append((x1, y1, x2, y2))
        confs.append(conf)
        clss.append(cls)
    return boxes, confs, clss
    

#4
class TrtSSD(object):
    """TrtSSD class encapsulates things needed to run TRT SSD."""
    #加载自定义组建，这里如果TensorRT版本小于7.0需要额外生成flattenconcat的自定义组件库
    def _load_plugins(self):
        if trt.__version__[0] < '7':
            ctypes.CDLL("ssd/libflattenconcat.so")
        trt.init_libnvinfer_plugins(self.trt_logger, '')
    #加载通过TAO Toolkit生成的推理引擎
    def _load_engine(self):
        TRTbin = './models/TRT_%s.bin' % self.model
        print(TRTbin)
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    #通过加载的引擎，生成可执行的上下文
    def _create_context(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            ##注意：这里的host_mem需要时用pagelocked memory，以免内存被释放
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()
    #初始化引擎
    def __init__(self, model, input_shape, output_layout=7):
        """Initialize TensorRT plugins, engine and conetxt."""
        cuda.init()
        self.ctx  = cuda.Device(0).make_context()
        self.device = self.ctx.get_device()
        
        self.model = model
        self.input_shape = input_shape
        self.output_layout = output_layout
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.context = self._create_context()
    #释放引擎，释放GPU显存，释放CUDA流
    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs
    #利用生成的可执行上下文执行推理
    def detect(self, img, conf_th=0.3):
        """Detect objects in the input image."""
        img_resized = _preprocess_trt(img, self.input_shape)
        np.copyto(self.host_inputs[0], img_resized.ravel())
        #将处理好的图片从CPU内存中复制到GPU显存
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        #开始执行推理任务
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        #将推理结果输出从GPU显存复制到CPU内存
        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()


        output = self.host_outputs[0]
        #for x in output:
        #    print(str(x),end=' ')
        return _postprocess_trt(img, output, conf_th, self.output_layout)
    
#5
INPUT_HW = (300, 300)
SUPPORTED_MODELS = [
    'ssd_mobilenet_v1_coco',
    'ssd_mobilenet_v1_egohands',
    'ssd_mobilenet_v2_coco',
    'ssd_mobilenet_v2_egohands',
    'ssd_mobilenet_v2_face',
    'ssd_resnet_18_garbage',
    'ssd_resnet_18_box'
]


#6
def detect_video(video, trt_ssd, conf_th, vis,result):
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
            
#7
def detect_one(img, trt_ssd, conf_th, vis, result):
    full_scrn = False
    tic = time.clock()
    ##开始检测，并将结果写到result.jpg中
    boxes, confs, clss = trt_ssd.detect(img, conf_th)
    toc = time.clock()
    curr_fps = (toc - tic)
    print("boxes: "+str(boxes))
    print("clss: "+str(clss))
    print("confs: "+str(confs))
    img = vis.draw_bboxes(img, boxes, confs, clss)
    print(result)
    cv2.imwrite("./uploads/result.jpg",img)        
    print("time: "+str(curr_fps)+"(sec)")
    
    
def detect_your_image(file):    
    filename = file
    result_file_name = "./result_image/"+str(time.time())+filename.split("/")[-1]
    img = cv2.imread(filename)
    cls_dict = get_cls_dict("ssd_resnet_18_box".split('_')[-1])
    model_name ="ssd_resnet18_box"
    trt_ssd = TrtSSD(model_name, INPUT_HW)
    vis = BBoxVisualization(cls_dict)
    print("start detection!")
    detect_one(img, trt_ssd, conf_th=0.3, vis=vis, result=result_file_name)
    cv2.destroyAllWindows()
    #print("ok")
    
    
#10
def get_fps(file):   
    filename = file
    result_file_name = "./result_image/result_video.mp4"
    video = cv2.VideoCapture(filename)
    cls_dict = get_cls_dict("ssd_resnet_18_box".split('_')[-1])
    model_name ="ssd_resnet18_box"
    trt_ssd = TrtSSD(model_name, INPUT_HW)
    vis = BBoxVisualization(cls_dict)
    print("start detection!")
    fps = detect_video(video, trt_ssd, conf_th=0.4, vis=vis, result=result_file_name)
    video.release()
    cv2.destroyAllWindows()
    return fps
    #print(result_file_name)
    
#11读取需要检测的文件夹中所有的图片, 并将检测结果按照"cls {xmin} {ymin} {xmax} {ymax}"的格式写到"mAP/input/detection-results/"文件夹中
def detect_dir(dir, trt_ssd, conf_th, vis):
    dirs = os.listdir(dir)
    print(dir)
    for i in dirs:
        if os.path.splitext(i)[1] == ".png":
            full_scrn = False
            #print("val/images/"+str(i))
            img = cv2.imread(dir+str(i))
            boxes, confs, clss = trt_ssd.detect(img, conf_th)
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



def get_map(dir_name): 
    cls_dict = get_cls_dict("ssd_resnet18_box".split('_')[-1])
    model_name ="ssd_resnet18_box"
    trt_ssd = TrtSSD(model_name, INPUT_HW)
    vis = BBoxVisualization(cls_dict)
    print("start detection!")
    remove_old_detection_results = subprocess.Popen('rm ./mAP/input/detection-results/*', shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    detect_dir(dir_name, trt_ssd = trt_ssd, conf_th=0.3, vis=vis)
    mAP = subprocess.Popen('python3.7 ./mAP/main.py', shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    mAP_result = mAP.stdout.read()
    return mAP_result
    
