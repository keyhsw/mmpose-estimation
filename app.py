import torch, torchvision
import sys
# sys.path.insert(0, 'test_mmpose/')
try:
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version
except:
    import mim
    mim.install('mmcv-full==1.5.0')
    
import mmpose
import gradio as gr
import cv2
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector

pose_config = 'configs/topdown_heatmap_hrnet_w48_coco_256x192.py'
pose_checkpoint = 'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
det_config = 'configs/faster_rcnn_r50_fpn_1x_coco.py'
det_checkpoint = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# initialize pose model
pose_model = init_pose_model(pose_config, pose_checkpoint, device='cpu')
# initialize detector
det_model = init_detector(det_config, det_checkpoint, device='cpu')

def predict(img):
    mmdet_results = inference_detector(det_model, img)
    person_results = process_mmdet_results(mmdet_results, cat_id=1)

    pose_results, returned_outputs = inference_top_down_pose_model(
      pose_model,
      img,
      person_results,
      bbox_thr=0.3,
      format='xyxy',
      dataset=pose_model.cfg.data.test.type)
    
    vis_result = vis_pose_result(
      pose_model,
      img,
      pose_results,
      dataset=pose_model.cfg.data.test.type,
      show=False)
    
    #vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)

    return vis_result

example_list = ['examples/demo2.png']
title = "Pose estimation"
description = ""
article = ""

# Create the Gradio demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(), 
                    outputs=[gr.Image(label='Prediction')], 
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()