"""
Configürasyon Yapıları
- Duygu Sözlüğü EN,TR
- Duygu Sayısı
...
"""
import os
from enum import Enum

class ModelTypes(Enum):
    Resnet50 = "resnet50"
    Resnet101 = "resnet101"

class Models(Enum):
    MiddleResnet = (
        ModelTypes.Resnet50,
        "https://drive.google.com/uc?id=1IrOLtug-wdfmvjmv7zn6JyK8qvVqpMZc",
        f"{os.path.dirname(__file__)}/resnet_50.pth")
    HeavyResnet = (
        ModelTypes.Resnet101,
        "https://drive.google.com/uc?id=1dkN2RDT6CAU4eWyjDIVJzxnAkwErvSWo",
        f"{os.path.dirname(__file__)}/resnet_101.pth")

EMOTION_DICT = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
    7: 'Contempt',
}

EMOTION_DICT_TR = {
    0: 'Notr',
    1: 'Mutlu',
    2: 'Uzgun',
    3: 'Sasirmis',
    4: 'Korku',
    5: 'Igrenme',
    6: 'Ofke',
    7: 'Kucumseme'
}
NUM_EMOTIONS = len(EMOTION_DICT)

