import argparse
import time
from pathlib import Path

import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import yaml

class YoloV5Detector():
    def get_classes(self, file):
        """Get classes name.
        # Argument:
            file: classes name for database.
        # Returns
            class_names: List, classes name.
        """
        with open(file) as f:
            d = yaml.load(f, Loader=yaml.FullLoader)  # dict
            return d['names']
    
    def __init__(self, weights='yolov5s.pt', img_size=640, conf_thres=0.25, iou_thres=0.45, agnostic_nms=True, device="", 
                    classes_file = "models/yolov5/coco.yaml", augment=True):
        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.agnostic_nms = agnostic_nms
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(img_size, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        self.names = self.get_classes(classes_file)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.augment = augment
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
          
    def __make_objects_from_detections(self, pred):
        res = []
        for i in range(len(pred["boxes"])):
            xyxy = pred["boxes"][i]
            xyxy = [int(item) for item in xyxy]
            cls = pred["classes"][i]
            score = pred["scores"][i]
            res.append({"bbox":xyxy, "class":cls, "score":float(score)})
        return res    
        
    def detect_image(self, img):
        original_frame = img
        # Padded resize
        img = letterbox(img, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        frame = np.ascontiguousarray(img)

        frame = torch.from_numpy(frame).to(self.device)
        frame = frame.half() if self.half else frame.float()  # uint8 to fp16/32
        frame /= 255.0  # 0 - 255 to 0.0 - 1.0
        if frame.ndimension() == 3:
            frame = frame.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(frame, augment=True)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=self.agnostic_nms)[0]
        # Rescale boxes from img_size to im0 size
        if type(pred) == type(None):
            output = {"boxes": None, "classes": None, "scores": None}
            return output
        pred[:, :4] = scale_coords(frame.shape[2:], pred[:, :4], original_frame.shape).round()
        scores = []
        for *xyxy, conf, cls in pred:
            scores.append(float(conf.cpu().detach().numpy()))
        pred_array = pred.cpu().detach().numpy()
        classes = pred_array[:,-1].astype(int)
        classes = [self.names[elem] for elem in classes]
        output = {"boxes": pred_array[:,:4],
                    "classes": classes, "scores": scores}
        return self.__make_objects_from_detections(output) 

    def draw_img(self, objs, img):
        for pred in objs:
            xyxy = pred["bbox"]
            cls = pred["class"]
            score = pred["score"]
            label = '%s %.2f' % (cls, score)
            cls_index = [i for i, elem in enumerate(self.names) if elem == cls][0]
            plot_one_box(xyxy, img, label=label, color=self.colors[int(cls_index)], line_thickness=3)
            
