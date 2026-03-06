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
import logging
from .structs import EMOTION_DICT_TR, EMOTION_DICT, NUM_EMOTIONS, ModelTypes

MODEL_PATH = f"{os.path.dirname(__file__)}/best.pth"

# Yazı parametreleri
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONST_SCALE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (255,255,255)  # yeşil
BG_COLOR = (0, 255, 0)     

class ClassifySentiment:
    
    def __init__(self, model_type:ModelTypes = ModelTypes.Resnet50, model_path = None, device=None):
        """
        Duygu sınıflandırması yapan model
        Args:
            model_type (ModelTypes) : One of the [ModelTypes] model types,
            model_path (str, Optional) : Model '.pth' path. 
            device (str, Optional): cpu, cude etc.
        """
        self.model_type = model_type
        if model_path is not None:
            global MODEL_PATH
            MODEL_PATH = model_path
        self.model, self.device = self._load_model(device) 
    
    def _load_model(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")
            
        if self.model_type == ModelTypes.Resnet101:
            model = models.resnet101(pretrained=False)
        elif self.model_type == ModelTypes.Resnet50:
            model = models.resnet50(pretrained=False)    
        else:
            raise KeyError(f"Invalid input for model_type = {self.model_type}. Must one of detect.ModelTypes ")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_EMOTIONS)

        # State dict yükle
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
        except Exception as e:
            raise Exception(f"{e}\n\nModel not found {MODEL_PATH}")
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        
        logging.info(f"Model {self.model_type.value} loaded succesfully from {MODEL_PATH} to {device}")
        
        return model, device

    @timer
    def predict(self, image:np.ndarray, verbose=True):
        """
        Görseldeki duyguyu tahmin eder,
        Args:
            image: input image
        Returns:
            predicted_class (int): f
            confidence (float): f
        """
            
        image = Image.fromarray(image)
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
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        # tahmin et ve output'ları uyarla
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)

            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        if verbose:
            logging.info(f"Prediction label: {predicted_class} conf: {confidence}")
            
        return predicted_class, confidence
    
    def visualize(self, image: np.ndarray, predicted_class: int, confidence: float, lang: str = "tr") -> np.ndarray:
        """
        Görsele tahmini etiket ve olasılığı ekler.
        
        Args:
            image (np.ndarray): BGR formatında resim
            predicted_class (int): Tahmin edilen sınıf indeksi
            confidence (float): Tahmin olasılığı (0-1)
            lang (str): Label dili ("tr" veya "en")
        
        Returns:
            annotated_image (np.ndarray): Annotated image
        """
        annotated_image = image.copy()
        
        # Label seçimi
        if lang == "tr":
            label_text = EMOTION_DICT_TR.get(predicted_class, "Unknown")
        else:
            label_text = EMOTION_DICT.get(predicted_class, "Unknown")
        
        # Label + confidence
        text = f"{predicted_class}:{label_text}: {confidence*100:.1f}%"
        
        # Text boyutunu al
        (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONST_SCALE, FONT_THICKNESS)
        
        # Text kutusu koordinatları
        x, y = 10, 30  # sol üst köşe
        cv2.rectangle(
            annotated_image,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            BG_COLOR,
            thickness=-1  # dolu kutu
        )
        
        # Text'i çiz
        cv2.putText(
            annotated_image,
            text,
            (x, y),
            FONT,
            FONST_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            lineType=cv2.LINE_AA
        )
        
        return annotated_image