# Sentinal (Sentiment Analyzer) Görsel Bazlı Duygu Analizi

RGB görüntüleri işleyerek sınıflandıran python paketi.

Yüz tespiti için Google mediapipe, sınıflandırma için eğitilmiş Resnet modellerini kullanır.

# Bağımlılıklar ve Uyumluluk

Python Versiyonu: __python.3.10.13__

### Paketler
requirements.txt'den hızlıca yüklenebilir
```
pip install -r requirements.txt
```
- mediapipe==0.10.32
- numpy==2.4.3
- opencv_contrib_python==4.13.0.92
- opencv_python==4.13.0.92
- Pillow==12.1.1
- sentinal==0.1.0
- torch==2.10.0
- torchvision==0.25.0

Testler İçin:
- Rerun: ```conda install -c conda-forge rerun-sdk```

**Uyarı**
- Mediapipe içerisinde kullanılan yüz tespiti için gerekli tflite modülü harici olarak indirilebilir. [docs](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector)
- Paket versiyonları test edilen environment'tan alınmıştır

# Veri

- Affectnet41k [Veri seti linki](https://huggingface.co/datasets/ValerianFourel/AffectNetDiffusion-Annotations-Render-And-Images): 
    | Etiket | Görsel Sayısı|
    | ----- | ----- |
    | (Neutral) | 4194 |
    | (Happy) | 6958 |
    | (Sad) | 2964 |
    | (Surprise) | 2968 |
    | (Fear) | 2944 |
    | (Disgust) | 2838 |
    | (Anger) | 2962 |
    | (Contempt) | 2871 |


# Model

| Model | Boyut | Parametre | Accuracy | Macro-F1 |
| --- | --- | --- | --- | --- |
| Resnet101 | ~ 170MB | 42.8 M | 94.26 | 0.9395 |
| Resnet50 | ~ 94.4MB | 23.9 M | 95.64 | 0.9566 |

# Model Eğitimi

### Veri Çoğaltma

Veriler her eğitim safhasında rastgele olarak torch içerisindeki dönüşümler kullanılarak çoğaltılmıştır.

Örnek:
```
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(), # PIL -> PyTorch Tensor dönüşümü, 0-255'ten 0-1 aralığına dönüşüm sağlar
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Imagenetin standart sapma değerleri, başlangıç için iyi
                        std=[0.229, 0.224, 0.225])
])
```

### Metrikler: Confusion Matrix, Macro F1

**Confusion Matrix**: Gerçek ve tahmin edilen sınıfları tablo şeklinde gösterir; doğru ve yanlış sınıflandırmaları hızlıca görmeyi sağlar. Her satır gerçek sınıfı, her sütun tahmin edilen sınıfı temsil eder.

**Macro F1 Score**: Tüm sınıfların F1 skorlarının basit ortalamasını alır; sınıf dengesizliği olan veri setlerinde az örneklem sayısına sahip verilere de eşit odak uygulanır. Bu sayede genel doğruluktan ziyade her etiket başına ortalama doğruluk hesaplanır.

### Cosine Learning Rate Scheduler
Modeller sabit learing rate ile değil, torch içerisindeki cosine learing rate scheduler ile giderek azalacak şekilde kullanılmıştır. Minimum olarak 1e-5 belirlenmiş ve 100 epoch'ta minimuma inecek şekilde aralıklandırılmıştır.