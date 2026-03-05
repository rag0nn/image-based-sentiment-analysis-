"""
Model Eğitim Scripti

veri:
- RGB Görseller: affectnet_41k_AffectOnly/EmocaProcessed_38k/EmocaResized_35k/FLAMEResized/
- Etiketler: Modified_processed_affectnet_paths.csv (header + 420,299 satır)
- Eğitim/Test/Válida: Modified_Corpus_38k_train_split.json, test_split, validation.json
"""

import os
from typing import Tuple
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torchvision import transforms, models
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import tqdm
import os
import logging
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from structs import *


# DEĞİŞKENLER

PATHBASE = Path(os.path.dirname(__file__))
METRICS_PATH = Path(PATHBASE / "training_info")
LOGS_PATH = Path(METRICS_PATH / "logs")

DATA_ROOT = Path("/home/enes/Desktop/sentiment-analysis/data/Affectnet41k")

# DATA_ROOT = Path("/content/drive/MyDrive/sentiment/Affectnet41k")
CSV_PATH = f"{DATA_ROOT}/labels.csv"
OUTPUT_LAST_MODEL_PATH = Path(PATHBASE / "last.pth")
OUTPUT_BEST_MODEL_PATH = Path(PATHBASE / "best.pth")

BATCH_SIZE = 24
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
EXTRA_EPOCH_COUNT = 3

def setup_logging():
    """Logging'i ayarla - konsol ve dosya çıkışı"""
    global LOGS_PATH
    log_dir = LOGS_PATH
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/training_{timestamp}.log"
    
    # Logger oluştur
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Dosya handler (detaylı)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Konsol handler (ana bilgiler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # console_formatter = logging.Formatter('%(message)s')
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file, timestamp

# VERİ

def prepare_dataset_paths(csv_path, data_root_path):
    """
    CSV dosyasından eğitim verisi hazırla
    
    Args:
        csv_path: Modified_processed_affectnet_paths.csv dosya yolu
        data_root_path: StableFaceData klasörü yolu
    
    Returns:
        list: (image_path, label) tuple'larından oluşan liste
    """
    
    logging.info("[INFO] CSV dosyası yükleniyor...")
    df = pd.read_csv(csv_path)

    # Görsel yolu ve etiketleri hazırla
    image_paths = []
    labels = []
    
    labels = df["label"]
    for pth, label in zip(df["image_path"], labels):
        path = f"{DATA_ROOT}/{pth}"
        image_paths.append(path)
        # Dosya var mı kontrol et
        if not os.path.exists(path):
            logging.warning(f"Dosya bulunamadı: {pth}")

    
    logging.info(f"Mevcut ve erişilebilir örnek: {len(image_paths)}")
    
    if len(image_paths) == 0:
        raise ValueError("Hiç görsel bulunamadı! Dosya yollarını kontrol edin.")
    
    return image_paths, labels

def split_dataset(image_paths, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Veri setini eğitim/válida/test olarak böl
    
    Args:
        image_paths: Görsel yolları
        labels: Etiketneleri
        train_ratio: Eğitim oranı (%)
        val_ratio: Válida oranı (%)
        test_ratio: Test oranı (%)
    
    Returns:
        tuple: (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    
    n = len(image_paths)
    indices = np.random.permutation(n) # n'ye kadarki sayı listesi ama karıştırılmış şekilde
    
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    train_indices = indices[:train_idx]
    val_indices = indices[train_idx:val_idx]
    test_indices = indices[val_idx:]
    
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    logging.info(f"Veri Bölünmesi:")
    logging.info(f"  - Eğitim: {len(train_paths)} örnek ({train_ratio*100:.1f}%)")
    logging.info(f"  - Válidasyon: {len(val_paths)} örnek ({val_ratio*100:.1f}%)")
    logging.info(f"  - Test: {len(test_paths)} örnek ({test_ratio*100:.1f}%)")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

def compute_class_weights(labels, num_classes=NUM_EMOTIONS, device=DEVICE):
    """
    Sınıf ağırlıklarını hesapla (inverse frequency).
    Az temsil edilen sınıflara daha yüksek ağırlık verir.
    
    Args:
        labels: Eğitim etiketleri
        num_classes: Toplam sınıf sayısı
        device: Torch cihazı
    
    Returns:
        torch.Tensor: Her sınıf için ağırlık tensörü
    """
    counter = Counter(labels)
    total = sum(counter.values())
    
    weights = []
    for cls_id in range(num_classes):
        count = counter.get(cls_id, 1)  # 0 bölme hatası önlemek için en az 1
        weight = total / (num_classes * count)
        weights.append(weight)
    
    weights_tensor = torch.FloatTensor(weights).to(device)
    
    logging.info("Sınıf Ağırlıkları (Weighted CrossEntropy):")
    for cls_id in range(num_classes):
        count = counter.get(cls_id, 0)
        logging.info(f"  Sınıf {cls_id} ({EMOTION_DICT.get(cls_id, '?')}): "
                     f"örnek={count}, ağırlık={weights_tensor[cls_id]:.4f}")
    
    return weights_tensor

# VERİSETİ

class SentimentDataset(Dataset):
    """RAM Cached Veri Seti Sınıfı
    Tüm görselleri başlangıçta RAM'e yükler, disk I/O darboğazını ortadan kaldırır.
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: Görsel dosyası yolları listesi
            labels: Duygu etiketeleri (0-7)
            transform: Görsel dönüşümleri
        """
        self.transform = transform
        self.images = []   # numpy array olarak RAM'de tutulacak
        self.labels = []
        
        skipped = 0
        logging.info(f"Görseller RAM'e yükleniyor ({len(image_paths)} dosya)...")
        for path, label in tqdm.tqdm(zip(image_paths, labels), total=len(image_paths), desc="RAM Cache"):
            try:
                img = Image.open(path).convert('RGB')
                self.images.append(np.array(img))
                self.labels.append(label)
                img.close()
            except Exception as e:
                skipped += 1
                if skipped <= 50:  # İlk 10 hatayı logla
                    logging.warning(f"HAtalı veriAtlanıyor: {path} - {e}")
        
        if skipped > 0:
            logging.warning(f"Toplam {skipped} görsel atlandı (yüklenemedi)")
        logging.info(f"RAM'e yüklenen: {len(self.images)} görsel")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # RAM'deki numpy array'i PIL Image'e çevir
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
def get_transforms()->Tuple:
    """
    Veriler model eğitmeye gönderilmeden geçirecekleri dönüşümler
    Returns:
        train_transform: eğitim için dönüşümler
        val_test_transform: validasyon ve test için dönüşümler
    """
    ### eğitim için
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(), # PIL -> PyTorch Tensor dönüşümü, 0-255'ten 0-1 aralığına dönüşüm sağlar
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Imagenetin standart sapma değerleri, pretrained olmasaydı kendi veri setimizinkini hesaplayacaktık
                           std=[0.229, 0.224, 0.225])
    ])
    
    ### validasyon için
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform
    
# MODEL HAZIRLIĞI

def create_model(num_classes=NUM_EMOTIONS, pretrained=True):
    """Resnet odelini oluştur"""
    model = models.resnet50(pretrained=pretrained)
    
    # Son katmanı göreve uygun şekilde değiştir
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # Linear(İnput boyutu, output boyutu)
    
    # Modeli yazdır
    logging.info(f"Resnet Modeli Oluşturuldu (Sınıf Sayısı: {num_classes})")
    
    return model

# EĞİTİM 

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Bir eğitim epoch'unu gerçekleştir (AMP destekli)"""
    model.train() # modeli eğitim moduna al
    running_loss = 0.0
    all_preds = []
    all_labels = []
    use_amp = scaler is not None
    
    pbar = tqdm.tqdm(train_loader, desc="Eğitim", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True) # set_to_none daha hızlı
        
        # İleri geçiş (AMP)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Geri geçiş (AMP)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # İstatistikler
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, macro_f1


def validate(model, val_loader, criterion, device):
    """Modeli doğrudan"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm.tqdm(val_loader, desc="Doğrulama", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, macro_f1


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_val_f1, 
                   train_losses, train_f1s, val_losses, val_f1s, scaler=None):
    """Checkpoint'i kaydet (model + optimizer + scheduler + scaler + geçmiş)"""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_f1': best_val_f1,
        'train_losses': train_losses,
        'train_f1s': train_f1s,
        'val_losses': val_losses,
        'val_f1s': val_f1s,
    }
    if scaler is not None:
        ckpt['scaler_state_dict'] = scaler.state_dict()
    torch.save(ckpt, checkpoint_path)
    logging.debug(f"[INFO] Checkpoint kaydedildi: {checkpoint_path} (Epoch: {epoch})")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, scaler=None):
    """
    Eğitimin önceki kısımlarından checkpoint yükler. 
    Args:
        checkppoint_path: pth uzantılı checkpoint dosyası yolu
        model: oluşturulmuş model mimarisi
        optimizer: eğitimde kullanılmış optimzer
        device: eğitimde kullanılmış cihaz
        scaler: AMP GradScaler (opsiyonel)
    
    """
    # CUDA bellek temizliği checkpoint yüklenmeden önce
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("CUDA bellek temizlendi (checkpoint yükleme öncesi)")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Checkpoint formatını kontrol et
    if 'model_state_dict' in checkpoint:
        # Tam checkpoint formatı
        model_state = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and any(k in checkpoint for k in ['conv1.weight', 'layer1.0.conv1.weight']):
        # Direkt model state_dict formatı
        model_state = checkpoint
    else:
        # Eğer ne checkpoint ne de state_dict değilse, direkt model ağırlığı olabilir
        model_state = checkpoint
    
    try:
        # Strict mode: tam uyuma kontrol et
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as e:
        # Uyumsuz keys varsa, onları görmezden gel
        logging.warning(f"[WARNING] Model mimarisi değişmiş olabilir. Non-strict mod kullanılıyor...")
        model.load_state_dict(model_state, strict=False)
    
    # Checkpoint formatında ek bilgiler varsa yükle
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            logging.warning(f"Optimizer state yüklenemedi, yeniden başlanacak: {e}")
    
    if 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            logging.warning(f"Scheduler state yüklenemedi, yeniden başlanacak: {e}")
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception as e:
            logging.warning(f"Scaler state yüklenemedi, yeniden başlanacak: {e}")
    
    start_epoch = checkpoint.get('epoch', 0)
    best_val_f1 = checkpoint.get('best_val_f1', 0)
    train_losses = checkpoint.get('train_losses', [])
    train_f1s = checkpoint.get('train_f1s', [])
    val_losses = checkpoint.get('val_losses', [])
    val_f1s = checkpoint.get('val_f1s', [])
    logging.info(f"Checkpoint yüklendi: {checkpoint_path}")
    logging.info(f"Önceki En İyi Macro F1: {best_val_f1:.4f}")
    logging.info(f"Eğitim {start_epoch + 1}. epoch'tan devam edecek...")
    
    # CUDA bellek temizliği checkpoint yüklemesi sonrası
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return start_epoch, best_val_f1, train_losses, train_f1s, val_losses, val_f1s


def test(model, test_loader, device):
    """Test seti üzerinde değerlendirme yap"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader, desc="Test", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return macro_f1, accuracy, all_preds, all_labels

def run():
    global PATHBASE
    global METRICS_PATH 
    global DATA_ROOT 
    global OUTPUT_LAST_MODEL_PATH 
    global OUTPUT_BEST_MODEL_PATH 
    global BATCH_SIZE 
    global NUM_EPOCHS
    global LEARNING_RATE 
    global DEVICE
    global PATIENCE
    
    # log başlat
    log_file, timestamp = setup_logging()
    logging.info("="*80)
    logging.info(f"EĞITIM SESSION BAŞLADI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*80)
    
    logging.info(f"\n[YAPILANDIRMA]")
    logging.info(f"  - Cihaz: {DEVICE}")
    logging.info(f"  - {torch.cuda.get_device_name(0)}")
    logging.info(f"  - CUDA Kullanılabilir: {torch.cuda.is_available()}")
    logging.info(f"  - Batch Size: {BATCH_SIZE}")
    logging.info(f"  - Epoch Sayısı: {NUM_EPOCHS}")
    logging.info(f"  - Learning Rate: {LEARNING_RATE}")
    
    # eğitim devam ettirme 
    resume_training = False
    if Path(OUTPUT_LAST_MODEL_PATH).exists():
        logging.info(f"\n[!] Önceki checkpoint bulundu: {OUTPUT_LAST_MODEL_PATH}")
        user_input = input("Eğitimi devam ettirmek istiyor musun? (e/h): ").strip().lower()
        resume_training = user_input in ['e', 'evet', 'y', 'yes']
    
    start_epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0
    
    # Veri Hazırlığı
    logging.info("\n[ADIM 1] Veri Hazırlanıyor...")
    image_paths, labels = prepare_dataset_paths(CSV_PATH, DATA_ROOT)
    # image_paths = image_paths[:2000]  # Test için veri azalt
    # labels = labels[:2000]
    
    # Veri setini böl
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        split_dataset(image_paths, labels)
    
    # Sentetik Dönüşümler
    logging.info("\n[ADIM 2] Dataloader'lar Sentezleniyor...")
    train_transform,val_test_transform  = get_transforms()
    
    # Veri setlerini oluştur
    train_dataset = SentimentDataset(train_paths, train_labels, train_transform)
    val_dataset = SentimentDataset(val_paths, val_labels, val_test_transform)
    test_dataset = SentimentDataset(test_paths, test_labels, val_test_transform)
    
    # DataLoader oluştur (pin_memory + persistent_workers ile hızlandırılmış)
    dl_kwargs = dict(num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **dl_kwargs)
    
    logging.info(f"DataLoader Oluşturuldu")
    
    # MODELİ HAZIRLA 
    logging.info("\n[ADIM 3] Model Hazırlanıyor...")
    
    # cuDNN benchmark aktifleştir (sabit input boyutunda hız artışı sağlar)
    torch.backends.cudnn.benchmark = True
    logging.info("cuDNN Benchmark: Aktif")
    
    model = create_model(num_classes=NUM_EMOTIONS, pretrained=True)
    model.to(DEVICE)
    
    # Sınıf ağırlıklarını hesapla
    class_weights = compute_class_weights(train_labels, num_classes=NUM_EMOTIONS, device=DEVICE)
    
    # Loss ve Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    
    # Mixed Precision (AMP) - FP16 ile ~%30-50 hız artışı
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    if use_amp:
        logging.info("Mixed Precision (AMP): Aktif")
    else:
        logging.info("Mixed Precision (AMP): Pasif (CPU modu)")
    
    logging.info(f"Loss: Weighted CrossEntropyLoss")
    logging.info(f"Scheduler: CosineAnnealingLR (T_max={NUM_EPOCHS}, eta_min=1e-6)")
    
    # Checkpoint'ten devam ettir
    train_losses = []
    train_f1s = []
    val_losses = []
    val_f1s = []
    
    if resume_training:
        start_epoch, best_val_f1, train_losses, train_f1s, val_losses, val_f1s = \
            load_checkpoint(OUTPUT_LAST_MODEL_PATH, model, optimizer, scheduler, DEVICE, scaler)
    
    # Kaç yeni epoch yapılacağını hesapla (early stopping için)
    num_new_epochs = NUM_EPOCHS - start_epoch
    
    # ========== EĞİTİM (DÖNGÜ) ==========
    logging.info(f"\n[ADIM 4] Model Eğitiliyor (Epoch {start_epoch + 1}-{NUM_EPOCHS})...  ({num_new_epochs} yeni epoch)\n")
    
    epoch = start_epoch
    
    while True:
        # Epoch range'i kontrol et ve işlemi gerçekleştir
        if epoch < NUM_EPOCHS:
            # Eğitim epoch'unu çalıştır
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"\n{'='*80}")
            logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}  (LR: {current_lr:.6f})")
            logging.info(f"{'='*80}")
            
            # Eğitim
            train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
            train_losses.append(train_loss)
            train_f1s.append(train_f1)
            
            # Doğrulama
            val_loss, val_f1 = validate(model, val_loader, criterion, DEVICE)
            val_losses.append(val_loss)
            val_f1s.append(val_f1)
            
            # Detaylı log
            logging.info(f"  Eğitim   - Loss: {train_loss:.4f}, Macro F1: {train_f1:.4f}")
            logging.info(f"  Doğrulama - Loss: {val_loss:.4f}, Macro F1: {val_f1:.4f}")
            
            # En iyi modeli kaydet
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), OUTPUT_BEST_MODEL_PATH)
                logging.info(f"En iyi model kaydedildi (Macro F1: {val_f1:.4f})")
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1
                logging.info(f"Patience Counter: {patience_counter}/{PATIENCE}")
            
            # Early Stopping kontrol (sadece num_new_epochs > 5 ise çalışsın)
            if patience_counter >= PATIENCE and num_new_epochs > 5:
                logging.info(f"\n[EARLY STOPPING] {PATIENCE} epoch boyunca iyileşme olmadı.")
                logging.info(f"Eğitim durduruluyor. En iyi model yükleniyor...")
                model.load_state_dict(torch.load(OUTPUT_BEST_MODEL_PATH))
                break
            
            # Checkpoint'i kaydet (devam ettirmek için)
            save_checkpoint(OUTPUT_LAST_MODEL_PATH, model, optimizer, scheduler, epoch, best_val_f1,
                           train_losses, train_f1s, val_losses, val_f1s, scaler)
            
            # Learning rate schedule
            scheduler.step()
            
            # CUDA bellek temizliği her epoch'tan sonra
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            epoch += 1
        
        else:
            # Tüm epochs tamamlandı, devam sorgusu yap
            logging.info("\n[!] Eğitim Tamamlandı!")
        
            continue_training = input(f"\nEğitimi 4 epoch daha devam ettirmek istiyor musun? (e/h): ").strip().lower()
            
            if continue_training not in ['e', 'evet', 'y', 'yes']:
                logging.info("Eğitim devamı reddedildi. Test aşamasına geçiliyor...")
                break

            logging.info(f"\n[ADIM 4-DEVAM] Model {EXTRA_EPOCH_COUNT} Epoch Daha Eğitiliyor...\n")
            NUM_EPOCHS += EXTRA_EPOCH_COUNT
            num_new_epochs += EXTRA_EPOCH_COUNT  # Yeni epoch sayısını da güncelle 
    
    # ========== TEST ==========,
    os.system(f'notify-send "MODEL EĞİTİMİ" "Model Eğitimi Tamamlandı"')
    logging.info("\n[ADIM 5] En İyi Model Test Ediliyor...")
    
    # En iyi modeli yükle
    model.load_state_dict(torch.load(OUTPUT_BEST_MODEL_PATH))
    test_f1, test_acc, all_preds, all_labels = test(model, test_loader, DEVICE)
    
    logging.info(f"\n[SONUÇ] En İyi Test Macro F1: {test_f1:.4f}")
    logging.info(f"[SONUÇ] TEn İyi Test Doğruluğu: {test_acc:.2f}%")
    
    # Classification Report - Sadece test setinde bulunan sınıfları kullan
    logging.info("\nDetaylı Rapor:")
    labels_present = sorted(set(all_labels) | set(all_preds))
    target_names = [EMOTION_DICT[i] for i in labels_present]
    report = classification_report(all_labels, all_preds, 
                            labels=labels_present,
                            target_names=target_names,
                            zero_division=0)
    logging.info("\n" + report)

    # Confusion Matrix - Sadece bulunan sınıfları göster
    cm = confusion_matrix(all_labels, all_preds, labels=labels_present)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[EMOTION_DICT[i] for i in labels_present],
                yticklabels=[EMOTION_DICT[i] for i in labels_present])
    plt.title(f'Confusion Matrix - Test Seti ({timestamp})')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    confusion_matrix_file = f'{METRICS_PATH}/confusion_matrix_{timestamp}.png'
    plt.savefig(confusion_matrix_file, dpi=100)
    logging.info(f"Confusion Matrix kaydedildi: {confusion_matrix_file}")
    plt.close()
    
    # Eğitim Grafiği
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Eğitim', linewidth=2)
    plt.plot(val_losses, label='Doğrulama', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Eğrisi')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_f1s, label='Eğitim', linewidth=2)
    plt.plot(val_f1s, label='Doğrulama', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.legend()
    plt.title('Macro F1 Eğrisi')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Eğitim Geçmişi - {timestamp}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    training_history_file = f'{METRICS_PATH}/training_history_{timestamp}.png'
    plt.savefig(training_history_file, dpi=100)
    logging.info(f"[INFO] Eğitim Geçmişi kaydedildi: {training_history_file}")
    plt.close()
    
    # ========== ÖZET DOSYASI OLUŞTUR ==========
    summary_file = f'{METRICS_PATH}/training_summary_{timestamp}.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"EĞİTİM ÖZET RAPORU\n")
        f.write(f"Tarih ve Saat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"YAPILANDIRMA:\n")
        f.write(f"  - Batch Size: {BATCH_SIZE}\n")
        f.write(f"  - Epoch Sayısı: {NUM_EPOCHS}\n")
        f.write(f"  - Learning Rate: {LEARNING_RATE}\n")
        f.write(f"  - Cihaz: {DEVICE}\n")
        f.write(f"  - Başlangıç Epoch: {start_epoch + 1}\n")
        f.write(f"  - Resume Mode: {resume_training}\n\n")
        
        f.write(f"VERİ SETI:\n")
        f.write(f"  - Toplam Örnekler: {len(train_paths) + len(val_paths) + len(test_paths)}\n")
        f.write(f"  - Eğitim: {len(train_paths)}\n")
        f.write(f"  - Doğrulama: {len(val_paths)}\n")
        f.write(f"  - Test: {len(test_paths)}\n\n")
        
        f.write(f"SONUÇLAR:\n")
        f.write(f"  - Test Macro F1: {test_f1:.4f}\n")
        f.write(f"  - Test Doğruluğu: {test_acc:.2f}%\n")
        f.write(f"  - En İyi Doğrulama Macro F1: {best_val_f1:.4f}\n")
        f.write(f"  - Final Eğitim Loss: {train_losses[-1]:.4f}\n")
        f.write(f"  - Final Doğrulama Loss: {val_losses[-1]:.4f}\n")
        f.write(f"  - Final Eğitim Macro F1: {train_f1s[-1]:.4f}\n")
        f.write(f"  - Final Doğrulama Macro F1: {val_f1s[-1]:.4f}\n\n")
        
        f.write(f"DOSYALAR:\n")
        f.write(f"  - Log Dosyası: {log_file}\n")
        f.write(f"  - Confusion Matrix: {confusion_matrix_file}\n")
        f.write(f"  - Eğitim Grafiği: {training_history_file}\n")
        f.write(f"  - Model: best_emotion_model.pth\n\n")
        
        f.write(f"DETAYLı RAPOR:\n")
        f.write(f"{report}\n")
    
    logging.info(f"[INFO] Özet Raporu kaydedildi: {summary_file}")
    logging.info("\n" + "="*80)
    logging.info(f"EĞITIM SESSION TAMAMLANDI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*80)
    logging.info(f"\n TÜM DOSYALAR:\n  - Log: {log_file}\n  - Özet: {summary_file}\n  - Matrix: {confusion_matrix_file}\n  - Grafik: {training_history_file}\n")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Model training script")

    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs")
    
    parser.add_argument("--input", type=str, default=DATA_ROOT,
                        help="Input file path")
    
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Number of patience training loops")
    
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of training epochs")
    
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Number of training epochs")
    
    parser.add_argument("--extra_epochs", type=int, default=EXTRA_EPOCH_COUNT,
                        help="Number of training epochs")

    args = parser.parse_args()
    
    NUM_EPOCHS = args.epochs
    DATA_ROOT = args.input
    PATIENCE = args.patience
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EXTRA_EPOCH_COUNT = args.extra_epochs
        
    run()
