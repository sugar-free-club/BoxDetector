import gradio as gr
import os
import subprocess
import sys
import time

import box_detection
from detector import Detector

INPUT_HW = (300, 300)

engine_path = './TRT_ssd_resnet18_box.bin'
detector = Detector(engine_path, INPUT_HW)
css = "footer {display: none !important;} .gradio-container {min-height: 0px !important;}"

with gr.Blocks(title="Sugar-free club", css=css) as demo:
    # api
    def run(image):
        '''test_model'''
        inputs = image
        result = box_detection.detect_your_image(detector, inputs)
        return result

    
    def test_mAP(image):
        '''test_mAP'''
        inputs = os.path.join(os.path.dirname(__file__), "images/")
        mAP = str(box_detection.bench_map(detector, inputs))##.replace('Figure(640x480)', '')
        
        print(mAP)
        return mAP
        
        
    def test_fps(image):
        '''test_latency'''
        inputs = os.path.join(os.path.dirname(__file__), "box_test_video.mp4")
        result = box_detection.bench_fps(detector, inputs)
        return result
    # page
    gr.Markdown("<h1>Sugar-free box detection.</h1>")
    with gr.Row():
        image_input = gr.Image(type='filepath', label="Input Image").style(height=350)
        image_output = gr.Image(type='filepath', interactive=False, label="Output Image").style(height=350)
    # buttion (need change color)
    with gr.Row():
        btn_run = gr.Button("Submit")
        btn_clear = gr.Button("Clear")
        btn_mAP = gr.Button("Test mAP")
        btn_fps = gr.Button("Test fps")
    # examples
    with gr.Row():
        gr.Examples(
            examples=[os.path.join(os.path.dirname(__file__), "100.png"), 
                      os.path.join(os.path.dirname(__file__), "101.png"),
                      ],
            inputs=image_input,
            outputs=image_output,
            fn=run
        )
        text = gr.Textbox(label="button test")
        
    #button function
    btn_run.click(
        fn=run,
        inputs=image_input,
        show_progress=True,
        outputs=image_output
    )
    
    btn_mAP.click(
        fn=test_mAP,
        inputs=image_input,
        show_progress=True,
        outputs=text
    )
    
    btn_fps.click(
        fn=test_fps,
        inputs=image_input,
        show_progress=True,
        outputs=text
    )
        
demo.launch(server_name="192.168.43.135") 
