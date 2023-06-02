# -*- coding: utf-8 -*-
import gradio as gr
import os
import subprocess
import sys
import time

import box_detection
from detector import Detector

INPUT_HW = (300, 300)

engine_path = '/home/nvidia/SugarFree/BoxDetector/models/ssd-complex/export/ssd_resnet18_point3.bin'
detector = Detector(engine_path, INPUT_HW)
css = "footer {display: none !important;} .gradio-container {min-height: 0px !important;}"
theme = gr.themes.Soft(primary_hue="red", secondary_hue="pink",spacing_size="md",radius_size="sm").set(
            body_background_fill="repeating-linear-gradient(45deg, *primary_100, *primary_100 10px, *primary_50 10px, *primary_50 20px)",
            body_background_fill_dark="repeating-linear-gradient(45deg, *primary_100, *primary_100 10px, *primary_50 10px, *primary_50 20px)",
            button_primary_background_fill="linear-gradient(70deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(70deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(70deg, *primary_300, *secondary_400)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_300",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="20px",
        )

with gr.Blocks(title="Sugar-free club", theme=theme, css=css) as demo:
    # api
    def run(image):
        '''test_model'''
        if image == None:
            raise gr.Error('Please upload image first!')
        inputs = image
        result = box_detection.detect_your_image(detector, inputs)
        return result
    
    def test_mAP():
        '''test_mAP'''
        inputs = os.path.join(os.path.dirname(__file__), "images/")
        mAP = str(box_detection.bench_map(detector, inputs))##.replace('Figure(640x480)', '')
        return [mAP,gr.Button.update(interactive=True),gr.Button.update(interactive=True),gr.Button.update(interactive=True)]
        
    def test_fps():
        '''test_latency'''
        inputs = os.path.join(os.path.dirname(__file__), "box_test_video.mp4")
        result = box_detection.bench_fps(detector, inputs)
        return [result,gr.Button.update(interactive=True),gr.Button.update(interactive=True),gr.Button.update(interactive=True)]
    
    def clear():
        '''empty the input'''
        return [None, None]

    def disable():
        return [gr.Button.update(interactive=False),gr.Button.update(interactive=False),gr.Button.update(interactive=False)]
    # page
    gr.Markdown("<h1>📦Sugar-free box detector</h1>")
    with gr.Row():
        image_input = gr.Image(type='filepath', label="Input Image").style(height=350)
        image_output = gr.Image(type='filepath', interactive=False, label="Output Image").style(height=350)
    # buttion (need change color)
    with gr.Row():
        btn_run = gr.Button("Submit", variant="primary")
        btn_clear = gr.Button("Clear", variant="stop")
        btn_mAP = gr.Button("Test mAP", variant="primary")
        btn_fps = gr.Button("Test fps", variant="primary")
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
        text = gr.Textbox(label="Result").style(height=50)
    gr.HTML(value="<br><br><center>Presented by <a href=https://github.com/Sugar-Free-Club>Sugar-Free club</a> 🐾 | Built with <a href=https://gradio.app/>Gradio</a>🔩</center>")

        
    #button function
    btn_run.click(
        fn=run,
        inputs=image_input,
        show_progress=True,
        outputs=image_output
    )
    
    btn_mAP.click(
        fn=disable,
        outputs=[btn_mAP,btn_fps,btn_run],
    )
    
    btn_fps.click(
        fn=disable,
        outputs=[btn_mAP,btn_fps,btn_run],
    )
    
    btn_mAP.click(
        fn=test_mAP,
        show_progress=True,
        outputs=[text,btn_mAP,btn_fps,btn_run]
    )
    
    btn_fps.click(
        fn=test_fps,
        show_progress=True,
        outputs=[text,btn_mAP,btn_fps,btn_run]
    )
    
    btn_clear.click(
        fn=clear,
        outputs=[image_input, image_output]
    )
        
# demo.launch(server_name="192.168.43.135", favicon_path='icon.jpeg') 
demo.launch(server_name="0.0.0.0", server_port=5000)
# demo.launch(server_name="36.150.110.74",
#             server_port=5000)
