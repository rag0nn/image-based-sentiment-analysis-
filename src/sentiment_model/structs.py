"""
Configürasyon Yapıları
- Duygu Sözlüğü EN,TR
- Duygu Sayısı
...
"""

from enum import Enum

class ModelTypes(Enum):
    Resnet50 = "resnet50"
    Resnet101 = "resnet101"

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
    3: 'ŞAşırmış',
    4: 'Korku',
    5: 'Igrenme',
    6: 'Ofke',
    7: 'Kucumseme'
}
NUM_EMOTIONS = len(EMOTION_DICT)

