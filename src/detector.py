import numpy as np
from utils import timer
from .face_recognition.detect import FaceDetector
from .sentiment_model.detect import SentimentClassifier
from .sentiment_model.structs import ModelTypes
from typing import List, Tuple


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
    
    def __init__(self, 
                 sentiment_model_type = ModelTypes.Resnet101, 
                 sentiment_model_path = None,
                 device = None
                 ):
        self.face_detector = FaceDetector()
        self.sentiment_model = SentimentClassifier(
            sentiment_model_type, 
            sentiment_model_path,
            device)
    
    @timer
    def detect(self, image: np.ndarray) -> Tuple[List[Prediction],List[np.ndarray]]:
        """
        Verilen görüntüdei yüzleri bulur ve duygularını tahmin eder.
        Args:
            image: input image
        Returns:
            predictions: conjugated tuples of predictions like (label, conf)
            annotations: annotations of process, first one is face annotation others sentiment annotations
        """
        annotations = []
        predictions = []
        
        # face recognition
        results = self.face_detector.detect_face(image)
        results = self.face_detector.add_margin(image,results)
        faces_annotated_image = self.face_detector.visualize(image, results)
        face_images = self.face_detector.crop_faces(image, results)
        
        annotations.append(faces_annotated_image)

        for face, detection in zip(face_images, results.detections):
            pred, conf = self.sentiment_model.predict(face)
            sentiment_annotated = self.sentiment_model.visualize(face,pred,conf,"tr")
            
            bbox = detection.bounding_box
            
            annotations.append(sentiment_annotated)
            predictions.append(
                Prediction(bbox.origin_x,bbox.origin_y,bbox.width,bbox.height,conf,pred)
            )
            
        return predictions, annotations
        
        
        
        
        
        
