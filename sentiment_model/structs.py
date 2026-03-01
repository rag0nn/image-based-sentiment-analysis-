"""
Configürasyon Yapıları
- Duygu Sözlüğü EN,TR
- Duygu Sayısı
...
"""

EMOTION_DICT = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
    7: 'Contempt',
    # 8: 'No Expression',
    # 9: 'Uncertain',
    # 10: 'Undefined/Not a Real Human',
    # 11: 'Indescribable'
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