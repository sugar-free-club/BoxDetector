import gradio as gr
import os
import subprocess
import sys
import logging
import time

import box_detection
from detector import Detector

INPUT_HW = (300, 300)

engine_path = sys.argv[1]
detector = Detector(engine_path, INPUT_HW)
logging.basicConfig(filename="./test_ssd_fps.log", level=logging.INFO)
inputs = os.path.join(os.path.dirname(__file__), "box_test_video.mp4")
for i in range(10):
    result = box_detection.bench_fps(detector, inputs)
    logging.info(f'modelname={engine_path}, fps={result}')