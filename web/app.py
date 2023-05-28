import gradio as gr
import os
import subprocess
import sys
import time

import box_detection as box_detection

css = "footer {display: none !important;} .gradio-container {min-height: 0px !important;}"

with gr.Blocks(title="Sugar-free club", css=css) as demo:
    # api
    def run(image):
        '''test_model'''
        input = image
        box_detection.detect_your_image(input)

    
    def test_mAP(image):
        '''test_mAP'''
        input = "/home/nvidia/8th_CV/mAP/input/images/"
        result = str(box_detection.get_map(input))##.replace('Figure(640x480)', '')
        mAP = result.split('\\n')[-3]
        print(mAP)
        # subprocess.run(["python","/home/lcy/Projects/BoxDetector/src/8th_CV/cv_map.py"])
        
        
    def test_fps(image):
        '''test_latency'''
        input = "/home/nvidia/8th_CV/box_test_video.mp4"
        result = box_detection.get_fps(input)
        # subprocess.run(["python","/home/lcy/Projects/BoxDetector/src/8th_CV/cv_fps.py"])
    # page
    gr.Markdown("<h1>Sugar-free box detection.</h1>")
    with gr.Row():
        image_input = gr.Image(type='filepath', label="Input Image").style(height=350)
        image_output = gr.Image(interactive=False, label="Output Image").style(height=350)
    # buttion (need change color)
    with gr.Row():
        btn_run = gr.Button("Submit")
        btn_clear = gr.Button("Clear")
        btn_mAP = gr.Button("Test mAP")
        btn_latency = gr.Button("Test fps")
    # examples
    with gr.Row():
        gr.Examples(
            examples=[os.path.join(os.path.dirname(__file__), "./test_image/100.png"), 
                      os.path.join(os.path.dirname(__file__), "./test_image/101.png"),
                      ],
            inputs=image_input,
            outputs=image_output,
            fn=run,
            cache_examples=True,
        )
        text = gr.Textbox(label="button test")
        
    #button function
    btn_run.click(
        fn=run,
        inputs=image_input,
        show_progress=True,
        outputs=text
    )
    
    btn_mAP.click(
        fn=test_mAP,
        inputs=image_input,
        show_progress=True,
        outputs=text
    )
    
    btn_mAP.click(
        fn=test_fps,
        inputs=image_input,
        show_progress=True,
        outputs=text
    )
        
demo.launch(server_name="192.168.43.135") 
