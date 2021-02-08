#!/usr/bin/python3

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import rospy
#from message_filters import ApproximateTimeSynchronizer, Subscriber
import message_filters
from sensor_msgs.msg import CompressedImage, Image

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
from cv_bridge import CvBridge
import math

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh) 
    
def door_plane(img, xyxy):
    h_min,h_max = int(xyxy[1]),int(xyxy[3])
    w_min,w_max = int(xyxy[0]),int(xyxy[2])
    door_img = img[h_min:h_max,w_min:w_max]
    height, width = h_max - h_min, w_max - w_min

    sample_number_height, height_param, height_start = 10, 0.6, 20
    height_step = math.floor(height_param* height / sample_number_height)
    height_end = height_start+height_step* sample_number_height -1


    sample_number_width, width_param, width_start = 10, 0.8, 10
    width_step = math.floor(width_param* width / sample_number_width)
    width_end = width_start+width_step* sample_number_width -1

    #A_matrix
    height_array = np.array([x for x in range(height_start, height_end, height_step) for y in range(width_start, width_end, width_step)])
    width_array = np.array([x for y in range(width_start, width_end, width_step) for x in range(height_start, height_end, height_step)])
    depth_array = door_img[height_start:height_end:height_step, width_start:width_end:width_step].flatten()
    ones_array = np.ones_like(depth_array)

    if len(height_array) == len(width_array) == len(depth_array):
        integrity_flag = True
    else:
        integrity_flag = False
        return None, None, None, False

    A_matrix = np.vstack((height_array, width_array,depth_array, ones_array)).T
    _, s, vh = np.linalg.svd(A_matrix, full_matrices = False)
    min_idx = np.argmin(s)
    min_vh = vh[:,min_idx]
    n_vector = min_vh[:3]
    vh_norm =  n_vector / np.linalg.norm(n_vector)
    return door_img[int((height_start+height_end)/2), width_start+2], door_img[int((height_start+height_end)/2), width_end-2], vh_norm, integrity_flag

def frame_plane(img, xyxy):
    h_min,h_max = int(xyxy[1]),int(xyxy[3])
    w_min,w_max = int(xyxy[0]),int(xyxy[2])
    door_img = img[h_min:h_max, w_min:w_max]
    height, width = door_img.shape[0], door_img.shape[1]

    sample_number_height, height_param, height_start = 10, 1, 0
    height_step = math.floor(height_param* height / sample_number_height)
    height_end = height_start+height_step* sample_number_height -1


    sample_number_width, width_param, width_start = 10, 1, 0
    width_step = math.floor(width_param* width / sample_number_width)
    width_end = width_start+width_step* sample_number_width -1

    # A_matrix
    height_array = np.array([x for x in range(height_start, height_end, height_step) for y in range(width_start, width_end, width_step)])
    width_array = np.array([x for y in range(width_start, width_end, width_step) for x in range(height_start, height_end, height_step)])

    # Depth Estimate
    depth_lu = door_img[height_start, width_start]
    depth_ru = door_img[height_start, width_end]
    depth_ld = door_img[height_end, width_start]
    depth_rd = door_img[height_end, width_end]

    depth_array_estimate = np.zeros((sample_number_height, sample_number_width), dtype = door_img.dtype)
    depth_array_estimate[0,:] = door_img[height_start,width_start:width_end:width_step]
    depth_array_estimate[:,0] = door_img[height_start:height_end:height_step,width_start]
    depth_array_estimate[:,-1] = door_img[height_start:height_end:height_step,width_end]
    row_indx = 0;
    for x in range(height_start+height_step, height_end, height_step):
        row_indx +=1
        col_indx = 0
        for y in range(width_start+width_step, width_end-width_step, width_step):
            col_indx+=1
            depth_array_estimate[row_indx,col_indx] = door_img[x,y]

    depth_array = depth_array_estimate.flatten()
    ones_array = np.ones_like(width_array)
    assert len(height_array) == len(width_array) == len(depth_array)

    # Stack A matrix
    A_matrix = np.vstack((height_array, width_array,depth_array, ones_array)).T
    _, s, vh = np.linalg.svd(A_matrix, full_matrices = False)
    min_idx = np.argmin(s)
    min_vh = vh[:,min_idx]
    n_vector = min_vh[:3]
    vh_norm =  n_vector / np.linalg.norm(n_vector)
    return door_img[int((height_start+height_end)/2), width_start], door_img[int((height_start+height_end)/2), width_end], vh_norm

    
def detect(image, depth):
    color_arr = np.frombuffer(image.data, np.uint8)
    image_np = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
    im0s = np.expand_dims(image_np, axis=0)
    
    depth_arr = CvBridge().imgmsg_to_cv2(depth, desired_encoding='16UC1')
    #depth_arr = np.frombuffer(depth.data, np.uint8)
    im1s = np.expand_dims(depth_arr, axis=0)

       # Stream results
    view_img = True
    if view_img:
        pass
        window_name1 = 'image'
        window_name2 = 'depth'
        cv2.imshow(window_name1, im0s[0])
        cv2.imshow(window_name2, im1s[0])
        if (cv2.waitKey(30) >= 0): 
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    check_requirements()
    door_belief = 0
    hinge_position = "unknown"
    rospy.init_node('door_detection', anonymous=True)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image, queue_size = 1, buff_size=2**24)
            image_sub = message_filters.Subscriber("/camera/color/image_raw/compressed", CompressedImage, queue_size = 1, buff_size=2**24)
            ats = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10,1 )
            ats.registerCallback(detect)
            rospy.spin()
