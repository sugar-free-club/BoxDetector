# -*- coding: utf-8 -*-
import os
import ctypes
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda


def _preprocess_trt_ssd(img, shape=(300, 300)):
    img = cv2.resize(img, shape)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    return img

def _preprocess_trt_rtmdet(img, shape=(320, 320)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, shape)
    img = np.expand_dims(img.transpose((2, 0, 1)).astype(np.float32), axis=0)
    mean=np.array([103.53, 116.28, 123.675]).reshape((1, 3, 1, 1))
    std=np.array([57.375, 57.12, 58.395]).reshape((1, 3, 1, 1))
    img -= mean
    img /= std
    

    return img

def _postprocess_trt_ssd(img, output, conf_th, output_layout):
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


def _postprocess_trt_rtmdet(output, conf_th, ratios, img_shape):
    # boxes, confs, clss = [], [], []
    num_dets, bboxes, scores, labels = output
    # print(output)
    # print(num_dets[0])
    # num_dets = int(num_dets[0])
    print(num_dets)
    scale_factor = (ratios[0], ratios[1], ratios[0], ratios[1])
    scores = scores[0, :num_dets]
    bboxes = bboxes[0, :num_dets]
    labels = labels[0, :num_dets]
    if num_dets > 0:
        bboxes /= scale_factor

        bboxes[:, 0::2].clamp_(0, img_shape[0])
        bboxes[:, 1::2].clamp_(0, img_shape[1])
        bboxes = bboxes.round().int()

    
    
    #print(len(output))
    # for index in range(0, len(output[-1])):
    #     conf = float(output[2][index])
    #     if conf < conf_th:
    #         continue
    #     x1 = int(output[1][index*4+0] * ratios[0])
    #     y1 = int(output[1][index*4+1] * ratios[1])
    #     x2 = int(output[1][index*4+2] * ratios[0])
    #     y2 = int(output[1][index*4+3] * ratios[1])
    #     cls = int(output[3][index])
    #     boxes.append((x1, y1, x2, y2))
    #     confs.append(conf)
    #     clss.append(cls)
    # print(ratios)
    print(bboxes.shape)
    print("Find boxes: ", str(bboxes))
    # print("Confs: ", str(confs))
    # print("Cls: ", str(clss))
    return bboxes, scores, labels


class Detector(object):
    #加载自定义组建，这里如果TensorRT版本小于7.0需要额外生成flattenconcat的自定义组件库
    def _load_plugins(self):
        trt.init_libnvinfer_plugins(self.trt_logger, '')
    #加载通过TAO Toolkit生成的推理引擎
    def _load_engine(self):
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    #通过加载的引擎，生成可执行的上下文
    def _create_context(self):
        for i, binding in enumerate(self.engine):
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            print("Biding", str(i), "shape: ", str(self.engine.get_binding_shape(binding)))
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
    def __init__(self, engine_path, input_shape, network_type, output_layout=7):
        """Initialize TensorRT plugins, engine and conetxt."""
        cuda.init()
        self.ctx  = cuda.Device(0).make_context()
        self.device = self.ctx.get_device()
        
        self.nt = network_type
        self.engine_path = engine_path
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
        if self.nt == 'ssd':
            img_resized = _preprocess_trt_ssd(img, self.input_shape)
        elif self.nt == 'rtmdet':
            img_resized = _preprocess_trt_rtmdet(img, self.input_shape)
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
        for i in range(len(self.host_outputs)):
            cuda.memcpy_dtoh_async(
                self.host_outputs[i], self.cuda_outputs[i], self.stream)
            print("output ", str(i), " : ", str(self.host_outputs[i].size))
        self.stream.synchronize()

        if self.nt == 'ssd':
            output = self.host_outputs[0]
            res = _postprocess_trt_ssd(img, output, conf_th, self.output_layout)
        elif self.nt == 'rtmdet':
            output = self.host_outputs
            img_shape = img.shape
            ratio_w = self.input_shape[0] / img.shape[0]
            ratio_h = self.input_shape[1] / img.shape[1]
            res = _postprocess_trt_rtmdet(output, conf_th, (ratio_w, ratio_h), img_shape)
        return res
    