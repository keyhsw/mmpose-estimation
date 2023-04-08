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

    #original_image = Image.open(img)
    width, height, channels = img.shape
    #vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)
    print(f"POSE_RESULTS: {pose_results}")
    
    # define colors for each body part
    body_part_colors = {
        "nose": (255, 165, 0),
        "left_eye": (255, 165, 0),
        "right_eye": (255, 165, 0),
        "left_ear": (255, 165, 0),
        "right_ear": (255, 165, 0),
        "left_shoulder": (0, 128, 0),
        "right_shoulder": (0, 0, 255),
        "left_elbow": (0, 128, 0),
        "right_elbow": (0, 0, 255),
        "left_wrist": (0, 128, 0),
        "right_wrist": (0, 0, 255),
        "left_hip": (0, 128, 0),
        "right_hip": (0, 0, 255),
        "left_knee": (0, 128, 0),
        "right_knee": (0, 0, 255),
        "left_ankle": (0, 128, 0),
        "right_ankle": (0, 0, 255)
    }
    
    # create a black image of the same size as the original image
    black_img = np.zeros((width, height, 3), np.uint8)
    
    # iterate through each person in the POSE_RESULTS data
    for person in pose_results:
        # get the keypoints for this person
        keypoints = person['keypoints']
        
        # draw lines between keypoints to form a skeleton
        skeleton = [("right_eye", "left_eye", (255, 165, 0)),("nose", "left_eye", (255, 165, 0)), ("left_eye", "left_ear", (255, 165, 0)), ("nose", "right_eye", (255, 165, 0)), ("right_eye", "right_ear", (255, 165, 0)),
                    ("left_shoulder", "left_ear", (255, 165, 0)),("right_shoulder", "right_ear", (255, 165, 0)), ("left_shoulder", "right_shoulder", (255, 165, 0)), ("left_shoulder", "left_elbow", (0, 128, 0)), ("right_shoulder", "right_elbow",(0, 0, 255)),
                    ("left_elbow", "left_wrist",(0, 128, 0)), ("right_elbow", "right_wrist",(0, 0, 255)), ("left_shoulder", "left_hip",(255, 165, 0)),
                    ("right_shoulder", "right_hip", (255, 165, 0)), ("left_hip", "right_hip", (255, 165, 0)), ("left_hip", "left_knee",(0, 128, 0)),
                    ("right_hip", "right_knee",(0, 0, 255)), ("left_knee", "left_ankle",(0, 128, 0)), ("right_knee", "right_ankle",(0, 0, 255))]
        for start_part, end_part, color in skeleton:
            start_idx = list(body_part_colors.keys()).index(start_part)
            end_idx = list(body_part_colors.keys()).index(end_part)
            if keypoints[start_idx][2] > 0.1 and keypoints[end_idx][2] > 0.1:
                pt1 = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                pt2 = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(black_img, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)
    
        # draw circles at each keypoint
        for i in range(keypoints.shape[0]):
            pt = (int(keypoints[i][0]), int(keypoints[i][1]))
            cv2.circle(black_img, pt, 3, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)


    
    # write black_img to a jpg file
    
    cv2.waitKey(0)
    cv2.imwrite("output.jpg", black_img)
    cv2.destroyAllWindows()
    
    return vis_result, "output.jpg"

example_list = ['examples/demo2.png']
title = "Pose estimation"
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