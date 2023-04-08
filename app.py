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
    
    # create a black image of the same size as the original image
    black_img = np.zeros((height, width, 3), np.uint8)
    
    # iterate through each person in the POSE_RESULTS data
    for person in pose_results:
        # get the keypoints for this person
        keypoints = person['keypoints']
        
        # draw lines between keypoints to form a skeleton
        skeleton = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (8,9), (9,10), (10,11), (8,12), (12,13), (13,14), (0,15), (15,17), (0,16), (16,18)]
        for i, j in skeleton:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(black_img, pt1, pt2, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    
        # draw circles at each keypoint
        for i in range(keypoints.shape[0]):
            if keypoints[i][2] > 0.0: # check if the keypoint is visible
                pt = (int(keypoints[i][0]), int(keypoints[i][1]))
                cv2.circle(black_img, pt, 3, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    
    # write black_img to a jpg file
    cv2.imwrite("output.jpg", black_img)
    cv2.waitKey(0)
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