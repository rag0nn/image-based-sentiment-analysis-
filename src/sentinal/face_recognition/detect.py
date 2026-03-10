import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Tuple, Union
from mediapipe.tasks.python.components.containers.detections import DetectionResult
import cv2
import numpy as np
import os

BBOX_MARGINS = (20,20,20,20) # left, top, right, bottom
TEXTMARGINS = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 255, 50) 
  
class FaceDetector:
    
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path=f'{os.path.dirname(__file__)}/detector.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)
    
    def detect_face(self,image:np.ndarray)->DetectionResult:
        """
        Verilen görseldeki yüzleri tespit eder ve döndürür.
        
        Args:
            image: Numpy.ndarray resim objesi
        Returns:
            rgb_annotated_image : Image
            detection_result : Results
        """
        # Resmi Yükle
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image
        )
        return self.detector.detect(mp_image)

    def add_margin(self, image, detection_result):
        global BBOX_MARGINS 
        margin_left, margin_top,margin_right,margin_bottom = BBOX_MARGINS
        h_img, w_img, _ = image.shape
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            # Bounding box koordinatları
            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height
            # Gap ekle ve sınırları kontrol et
            bbox.origin_x = max(0, x - margin_left)
            bbox.origin_y = max(0, y - margin_top)
            bbox.width = min(w_img, x + w + margin_right) - bbox.origin_x
            bbox.height = min(h_img, y + h + margin_bottom) - bbox.origin_y
                    
        return detection_result
            
    def visualize(self,
        image,
        detection_result
    ) -> np.ndarray:
        annotated_image = image.copy()
        h,w,_ = image.shape
        
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, int(h / 200))

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            category_name = '' if category_name is None else category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (TEXTMARGINS + bbox.origin_x,
                            TEXTMARGINS + ROW_SIZE + bbox.origin_y)
            cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
        
        return annotated_image   
    
    def crop_faces(self, image, detection_result) -> List[np.ndarray]:
        """
        REsimdeki yüzleri kırpar ve döndürür, (Alanı en büyükten en küçüğe)
        Args:
            image: INput image
            detection_result: mediapipe detection result
        Return:
            faces: List of cropped images
        """
        
        if not detection_result.detections:
            return []
        
        faces = [] 
        areas = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            area = bbox.width * bbox.height

            # Bounding box koordinatları
            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height
            
            x1 = x
            x2 = x + w
            y1 = y
            y2 = y + h
            
            faces.append(image[y1:y2, x1:x2])
            areas.append(area)
            
        sorted_zipped = sorted( zip(faces, areas), key= lambda x: x[1], reverse=True)

        sorted_faces = [face for face, area in sorted_zipped]
        
        return sorted_faces
    