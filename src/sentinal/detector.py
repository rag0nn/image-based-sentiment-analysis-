import numpy as np
from .utils import timer, Colors
from .face_recognition.detect import FaceDetector
from .sentiment_model.detect import SentimentClassifier
from .sentiment_model.structs import ModelTypes, Models
from typing import Dict, List, Tuple
import logging
import gdown
import os
import cv2

CHOSEN_MODEL = Models.HeavyResnet

class Prediction:
    
    def __init__(self,
        x:int,
        y:int,
        w:int,
        h:int,
        conf:float,
        pred_lbl:int,
        real_lbl:int=None):
        
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.conf = conf
        self.pred_lbl = pred_lbl
        self.real_lbl = real_lbl
        
    def __repr__(self):
        return f"({self.x},{self.y},{self.w},{self.h}) {self.conf} {self.pred_lbl} {self.real_lbl}"
        
class Sentinal:
    
    def __init__(self, sentiment_model:Models = None,
                 device = None
                 ):        
        if sentiment_model is not None:
            global CHOSEN_MODEL
            CHOSEN_MODEL = sentiment_model
            
        model_type, remote_path, local_path = CHOSEN_MODEL.value
        self._check_models(local_path,remote_path)
        
        self.sentiment_model = SentimentClassifier(
            model_type, 
            local_path,
            device)
        self.face_detector = FaceDetector()
    
    def _check_models(self, local_path, remote_path):
        if os.path.exists(local_path):
            logging.info(f"Model file found: {local_path}")
        else:
            logging.info(f"Model file did'nt find, Downloading...: {remote_path}")
            try:
                gdown.download(remote_path, local_path, quiet=False)
            except Exception as e:
                raise Exception(f"{e} Error occured when downloading model file")
            logging.info("Model file downloaded successfully")


    @timer
    def detect(self, image: np.ndarray) -> List[Prediction]:
        """
        Verilen görüntüdei yüzleri bulur ve duygularını tahmin eder.
        Args:
            image: input image
        Returns:
            predictions: conjugated tuples of predictions like (label, conf)
            annotations: annotations of process, first one is face annotation others sentiment annotations
        """
        predictions = []
        
        # face recognition
        results = self.face_detector.detect_face(image)
        results = self.face_detector.add_margin(image,results)
        face_images = self.face_detector.crop_faces(image, results)

        # sentiment analysis
        for face, detection in zip(face_images, results.detections):
            pred, conf = self.sentiment_model.predict(face)
            
            bbox = detection.bounding_box
            
            predictions.append(
                Prediction(bbox.origin_x,bbox.origin_y,bbox.width,bbox.height,conf,pred)
            )
            
        logging.info(f"Founded {len(predictions)} faces")
            
        return predictions
    
    def visualize(self, image:np.ndarray, predictions:List[Prediction], label_dict:Dict):
        H,W,_ = image.shape
        for pred in predictions:
            
            # roi
            cv2.rectangle(
                image, 
                (pred.x,pred.y),
                (pred.x+pred.w,pred.y+pred.h),
                Colors.RED_PRIMARY.value,
                int(min(H,W)/100))
            
            #label info
            label_rect_gap = int(min(H,W)/30)
            lbl = f"{pred.conf:.2f} {pred.pred_lbl} {label_dict.get(pred.pred_lbl,'Uknown')}"
            font_scale = min(W,H)/500
            font_thickness = int(min(W,H)/200)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(lbl, font, font_scale, font_thickness)
            
            # label ractangle
            cv2.rectangle(
                image,
                (pred.x,pred.y),
                (pred.x+text_width+label_rect_gap, pred.y+text_height+label_rect_gap),
                Colors.RED_PRIMARY.value,
                -1
            )
            
            # label
            cv2.putText(
                image,
                lbl,
                (pred.x+int(label_rect_gap/2),pred.y+text_height+int(label_rect_gap/2)),
                font,
                font_scale,
                Colors.SNOWWHITE.value,
                font_thickness
                )
            
        return image
            
    def close(self):
        self.face_detector.close()
        
        
        
        
        
