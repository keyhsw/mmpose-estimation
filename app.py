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
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
from PIL import Image
import cv2
import numpy as np

from openxlab.model import download
download(model_repo='houshaowei/mmpose-estimation', 
model_name='faster_rcnn_r50_fpn_1x_coco_20200130-047c8118')

download(model_repo='houshaowei/mmpose-estimation', 
model_name='hrnet_w48_coco_256x192-b9e0b3ab_20200708')

pose_config = 'configs/topdown_heatmap_hrnet_w48_coco_256x192.py'
pose_checkpoint = 'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
det_config = 'configs/faster_rcnn_r50_fpn_1x_coco.py'
det_checkpoint = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# initialize pose model
pose_model = init_pose_model(pose_config, pose_checkpoint, device='cuda')
# initialize detector
det_model = init_detector(det_config, det_checkpoint, device='cuda')

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

    #original_image = Image.open(img)
    width, height, channels = img.shape
    #vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)
    print(f"POSE_RESULTS: {pose_results}")
    
    # define colors for each body part
    body_part = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16
    }
    orange=(51,153,255)
    blue=(255,128,0)
    green=(0,255,0)
    
    # create a black image of the same size as the original image
    black_img = np.zeros((width, height, 3), np.uint8)
    
    # iterate through each person in the POSE_RESULTS data
    for person in pose_results:
        # get the keypoints for this person
        keypoints = person['keypoints']
        
        # draw lines between keypoints to form a skeleton
        skeleton = [("right_eye", "left_eye", orange),("nose", "left_eye", orange), ("left_eye", "left_ear", orange), ("nose", "right_eye", orange), ("right_eye", "right_ear", orange),
                    ("left_shoulder", "left_ear", orange),("right_shoulder", "right_ear", orange), ("left_shoulder", "right_shoulder", orange), ("left_shoulder", "left_elbow", green), ("right_shoulder", "right_elbow",blue),
                    ("left_elbow", "left_wrist",green), ("right_elbow", "right_wrist",blue), ("left_shoulder", "left_hip",orange),
                    ("right_shoulder", "right_hip", orange), ("left_hip", "right_hip", orange), ("left_hip", "left_knee",green),
                    ("right_hip", "right_knee",blue), ("left_knee", "left_ankle",green), ("right_knee", "right_ankle",blue)]
        for start_part, end_part, color in skeleton:
            start_idx = list(body_part.keys()).index(start_part)
            end_idx = list(body_part.keys()).index(end_part)
            if keypoints[start_idx][2] > 0.1 and keypoints[end_idx][2] > 0.1:
                pt1 = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                pt2 = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(black_img, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)
    
        # draw circles at each keypoint
        #for i in range(keypoints.shape[0]):
        #    pt = (int(keypoints[i][0]), int(keypoints[i][1]))
        #    cv2.circle(black_img, pt, 3, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)


    
    # write black_img to a jpg file
    
    cv2.waitKey(0)
    cv2.imwrite("output.jpg", black_img)
    cv2.destroyAllWindows()
    
    return vis_result, "output.jpg"

example_list = ['examples/demo2.png']
title = "MMPose estimation"
description = ""
article = ""

# Create the Gradio demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(), 
                    outputs=[gr.Image(label='Prediction'), gr.Image(label='Poses')], 
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
