from face_recognition.face_recognition import detect_face, crop_largest_face
from sentiment_model.structs import EMOTION_DICT_TR,NUM_EMOTIONS
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from PIL import Image
import os
import numpy as np
from utils import timer
import cv2


MODEL_PATH = "/home/enes/Desktop/old/best.pth"
# MODEL_PATH = f"{os.path.dirname(__file__)}/sentiment_model/best.pth"
class SentimentDetector:
    
    def __init__(self):
        self.sentiment_model, self.device = self._load_sentiment_model() 
    
    def _load_sentiment_model(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        global MODEL_PATH
        model_path = MODEL_PATH
        # Model mimarisini oluştur
        model = models.resnet101(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_EMOTIONS)

        # State dict yükle
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        return model, device

    def _predict_image_sentiment(self,model, device, image:Image):
        # dönüşümler
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # dönüşümler uygula
        input_tensor = transform(image).unsqueeze(0).to(device)

        # tahmin et ve output'ları uyarla
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)

            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence
    
    @timer
    def apply(self,frame_np:np.ndarray):
        """
        Args:
            frame_np: input image np.ndarray
        Return:
            predicted_class: Predicted label class
            confidence: Predicted label confidence
            cropped_face_image: Detected and cropped face image
            face_xywh: Detected face coordinates
            face_annotated_image: Annoted with face image
        """

        # face detection
        face_annotated_image, detect_results = detect_face(frame_np)
        cropped_face_image, face_xywh = crop_largest_face(frame_np,detect_results)
        
        # sentiment classification
        frame_pil = Image.fromarray(cropped_face_image)
        predicted_class, confidence = self._predict_image_sentiment(self.sentiment_model, self.device, frame_pil)

        # some outputs
        cv2.putText(face_annotated_image,f"{EMOTION_DICT_TR[predicted_class]} {confidence:.3f}",(face_xywh[0] + 5, face_xywh[1] + 40),cv2.FONT_HERSHEY_DUPLEX,1.6,(0,255,0),2)
        
        return predicted_class, confidence, cropped_face_image, face_xywh, face_annotated_image
        
        
if __name__ == "__main__":
    pass