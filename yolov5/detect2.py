import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import math

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import pyrealsense2 as rs    

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
        return None, False

    A_matrix = np.vstack((height_array, width_array,depth_array, ones_array)).T
    _, s, vh = np.linalg.svd(A_matrix, full_matrices = False)
    min_idx = np.argmin(s)
    min_vh = vh[:,min_idx]
    n_vector = min_vh[:3]
    vh_norm =  n_vector / np.linalg.norm(n_vector)
    return vh_norm, integrity_flag

def frame_plane(img, xyxy):
    h_min,h_max = int(xyxy[1]),int(xyxy[3])
    w_min,w_max = int(xyxy[0]),int(xyxy[2])
    door_img = img[h_min:h_max, w_min:w_max:1]
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
    return vh_norm
    

def detect(save_img=False):
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)
     

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    #webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #    ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    
    vid_path, vid_writer = None, None
    #if webcam:
    view_img = True
    #    cudnn.benchmark = True  # set True to speed up constant image size inference
    #    dataset_color = LoadStreams(source, img_size=imgsz)
        #dataset_color = LoadStreams(str(4), img_size=imgsz)
        #dataset_depth = LoadStreams(str(2), img_size=imgsz)
    #else:
    #    save_img = True
    #    dataset_color = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    #for (path, img, im0s, vid_cap) in dataset_color:
    while True:
        frameset = pipe.wait_for_frames()
        aligned_frames = align.process(frameset)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        im0s = np.expand_dims(color_image, axis=0)
        im1s = np.expand_dims(depth_image, axis=0)

        # Letterbox
        s = np.stack([letterbox(x, new_shape=imgsz)[0].shape for x in im0s], 0)  # inference shapes
        rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        im0 = [letterbox(x, new_shape=imgsz, auto=rect)[0] for x in im0s]
        im0 = np.stack(im0, 0)
        im0 = im0[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        #print("im0:::::::"+str(np.ascontiguousarray(im0).shape))
        im0 = np.ascontiguousarray(im0)
        #print("im0:::::::"+str(np.ascontiguousarray(im0)))

        #im1 = [letterbox(x, new_shape=imgsz, auto=rect)[0] for x in im1s]
        #im1 = np.stack(im1, 0)
        #im1 = im1[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        #im1 = np.ascontiguousarray(im1)

        # image intrinsic and extrin
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        #print("dataset_depth:::::::::::::;;"+str((dataset_depth)))
        #_,_,im1s,_ = dataset_depth
        img = torch.from_numpy(im0).to(device)
        # print("img:::::::"+str(img))
        img = img.half() if half else img.float()  # uint8 to fp16/32
        #print("img:::::::"+str(img))
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        #window_name = 'image'
        #cv2.imshow(window_name, img)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            #if webcam:  # batch_size >= 1
            s, im0 = '%g: ' % i, im0s[i].copy()
            im1 = im1s[i].copy()
            im1 = cv2.convertScaleAbs(im1, alpha=0.03)
            #else:
             #   p, s, im0, frame = path, '', im0s, getattr(dataset_color, 'frame', 0)

            #p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset_color.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        if (names[int(cls)] == "door"):
                            door_pl, data_integrity = door_plane(im1, xyxy)
                            if data_integrity == False:
                                continue
                            hinge_position = "left"
                            if hinge_position == "left":
                                h_min,h_max = xyxy[1], xyxy[3]
                                w_min,w_max = xyxy[0], xyxy[2]
                                h_min -= 20
                                if h_min < 0: # do not go beyond the image
                                    h_min = 10
                                #h_max += 40
                                w_min -= 20  # do not go beyond the image
                                if w_min < 0:
                                    w_min = 10
                                w_max += 50
                     #           if w_max > im:
                     #               w_max = 10

                            else:
                                h_min,h_max = xyxy[1], xyxy[3]
                                w_min,w_max = xyxy[0], xyxy[2]
                                h_min -= 20
                                if h_min < 0: # do not go beyond the image
                                    h_min = 10
                                #h_max += 40
                                w_min -= 70  # do not go beyond the image
                                if w_min < 0:
                                    w_min = 10
                                w_max += 20

                            print("im1[0].size:::::::::::::::::::"+str(im1.shape))
                            xyxy[1], xyxy[3] = h_min,h_max
                            xyxy[0], xyxy[2] = w_min,w_max
                            frame_pl = frame_plane(im1, xyxy)
                            dot_product = np.dot(door_pl,frame_pl)
                            angle = math.acos(dot_product)
                            global door_belief
                            if door_belief ==0:
                                door_belief = angle
                            else:
                                door_belief = 0.9 * door_belief+ 0.1* angle
                            
                            print("angle difference:::::::::::::::"+str(door_belief))
                        
                        #print("xyxy"+str(int(xyxy[0])))
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                pass
                window_name = 'image'
                cv2.imshow(window_name, im0)

                if (cv2.waitKey(30) >= 0): 
                    break

            # Save results (image with detections)
            if save_img:
                if dataset_color.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


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
    print(opt)
    check_requirements()
    door_belief = 0

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
