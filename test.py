# from face_recognition.face_recognition import detect_face
from project_test.config import data_img_dict, data_vid_list
from sentiment_model.structs import EMOTION_DICT_TR

import cv2

import torch
import torchvision.transforms as transforms
from PIL import Image


def load_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model mimarisini oluştur
    from torchvision import models
    import torch.nn as nn
    from sentiment_model.structs import NUM_EMOTIONS

    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_EMOTIONS)

    # State dict yükle
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device

import torch.nn.functional as F

def predict_image(model, device, img_path):
    image = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence


def test_sentiment_via_images(model_path: str):

    model, device = load_model(model_path)

    for k, v in data_img_dict.items():
        img_path = v[0]

        predicted_class, confidence = predict_image(model, device, img_path)

        label_text = EMOTION_DICT_TR[predicted_class]
        display_text = f"{label_text} ({confidence*100:.1f}%)"

        # OpenCV ile görseli aç
        img = cv2.imread(img_path)
        img = cv2.resize(img,(600,800))
        # Yazıyı ekle
        cv2.putText(
            img,
            display_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Prediction", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

import torch.nn.functional as F

def predict_video(model_path: str, video_path: str):

    model, device = load_model(model_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Video açılamadı.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV BGR → PIL için RGB dönüşüm
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        label_text = EMOTION_DICT_TR[predicted_class]
        display_text = f"{label_text} ({confidence*100:.1f}%)"

        # Frame üzerine yaz
        cv2.putText(
            frame,
            display_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        frame = cv2.resize(frame,(600,800))
        cv2.imshow("Video Emotion Prediction", frame)

        # q'ya basınca çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    MODEL_PATH = "/home/enes/Desktop/sentiment-analysis/sentiment_model/best.pth"
    # test_sentiment_via_images(MODEL_PATH)
    
    predict_video(MODEL_PATH,data_vid_list[0])