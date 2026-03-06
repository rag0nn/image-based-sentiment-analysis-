import numpy as np
from utils import timer
from .face_recognition.detect import FaceDetector
from .sentiment_model.detect import ClassifySentiment
from .sentiment_model.structs import ModelTypes
from typing import List, Tuple

class Sentinal:
    
    def __init__(self, 
                 sentiment_model_type = ModelTypes.Resnet50, 
                 sentiment_model_path = None,
                 device = None
                 ):
        self.face_detector = FaceDetector()
        self.sentiment_model = ClassifySentiment(
            sentiment_model_type, 
            sentiment_model_path,
            device)
    
    @timer
    def detect(self, image: np.ndarray) -> Tuple[Tuple[int,float],List[np.ndarray]]:
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

        for face in face_images:
            pred, conf = self.sentiment_model.predict(face)
            sentiment_annotated = self.sentiment_model.visualize(image,pred,conf,"tr")
            annotations.append(sentiment_annotated)
            predictions.append((pred, conf))
            
        return predictions, annotations
        
        
        
        
        
        
