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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import tqdm
import os
import logging
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from structs import *

# DEĞİŞKENLER

PATHBASE = Path(os.path.dirname(__file__))
METRICS_PATH = Path(PATHBASE / "training_info")
LOGS_PATH = Path(METRICS_PATH / "logs")

DATA_ROOT = Path(PATHBASE / "data" / "StableFaceData")
CSV_PATH = f"{DATA_ROOT}/AffectNet41k_FlameRender_Descriptions_Images/Modified_processed_affectnet_paths.csv"
CHECKPOINT_PATH = "training_checkpoint.pth"

BATCH_SIZE = 24
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    # # Eski handler'ları temizle
    # for handler in logger.handlers[:]:
    #     logger.removeHandler(handler)
    
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
    
    # Sadece FLAMEResized klasöründe mevcut olan görselleri seç
    df_processed = df[df['In FLAMEResized'] == 1].copy()
    logging.info(f"Toplam örnek: {len(df)}, İşlenmiş örnek (FLAMEResized): {len(df_processed)}")
    
    # Görsel yolu ve etiketleri hazırla
    image_paths = []
    labels = []
    
    for idx, row in df_processed.iterrows():
        # Görsel yolu oluştur
        subfolder_filename = row['Subfolder_Filename']
        image_path = Path(data_root_path) / 'AffectNet41k_FlameRender_Descriptions_Images' / \
                     'affectnet_41k_AffectOnly' / 'EmocaProcessed_38k' / \
                     'EmocaResized_35k' / 'FLAMEResized' / f"{subfolder_filename}.png"
        
        # Dosya var mı kontrol et
        if image_path.exists():
            image_paths.append(str(image_path))
            labels.append(int(row['Second Column']))  # Duygu etiketi (0-11)
        else:
            if idx % 1000 == 0:
                logging.warning(f"Dosya bulunamadı: {image_path}")
    
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

# VERİSETİ

class SentimentDataset(Dataset):
    """Veri SEti Sınıfı
    Torch Dataset sınıfını kalıtır
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: Görsel dosyası yolları listesi
            labels: Duygu etiketeleri (0-11)
            transform: Görsel dönüşümleri
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # GÖRSEL YÜKLESİ
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Görsel yüklenemedi: {image_path} - {e}")
            # Siyah görsel döndür
            image = Image.new('RGB', (224, 224), color='black')
        
        # ETİKET
        label = self.labels[idx]
        
        # DÖNÜŞÜM
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
    model = models.resnet101(pretrained=pretrained)
    
    # Son katmanı göreve uygun şekilde değiştir
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # Linear(İnput boyutu, output boyutu)
    
    # Modeli yazdır
    logging.info(f"Resnet Modeli Oluşturuldu (Sınıf Sayısı: {num_classes})")
    
    return model

# EĞİTİM 

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Bir eğitim epoch'unu gerçekleştir"""
    model.train() # modeli eğitim moduna al
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm.tqdm(train_loader, desc="Eğitim", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # İleri geçiş
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Geri geçiş
        optimizer.zero_grad() # önceki gradyanları sıfırla
        loss.backward() # geri gradyanları hesapla
        optimizer.step() # ağırlıkları güncelle
        
        # İstatistikler
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Modeli doğrudan"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm.tqdm(val_loader, desc="Doğrulama", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_acc, 
                   train_losses, train_accs, val_losses, val_accs):
    """Checkpoint'i kaydet (model + optimizer + geçmiş)"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
    }, checkpoint_path)
    logging.debug(f"[INFO] Checkpoint kaydedildi: {checkpoint_path} (Epoch: {epoch})")


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """
    Eğitimin önceki kısımlarından checkpoint yükler. 
    Args:
        checkppoint_path: pth uzantılı checkpoint dosyası yolu
        model: oluşturulmuş model mimarisi
        optimizer: eğitimde kullanılmış optimzer
        device: eğitimde kullanılmış cihaz
    
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
    
    start_epoch = checkpoint.get('epoch', 0)
    best_val_acc = checkpoint.get('best_val_acc', 0)
    train_losses = checkpoint.get('train_losses', [])
    train_accs = checkpoint.get('train_accs', [])
    val_losses = checkpoint.get('val_losses', [])
    val_accs = checkpoint.get('val_accs', [])
    logging.info(f"Checkpoint yüklendi: {checkpoint_path}")
    logging.info(f"Önceki En İyi Doğruluk: {best_val_acc:.2f}%")
    logging.info(f"Eğitim {start_epoch + 1}. epoch'tan devam edecek...")
    
    # CUDA bellek temizliği checkpoint yüklemesi sonrası
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return start_epoch, best_val_acc, train_losses, train_accs, val_losses, val_accs


def test(model, test_loader, device):
    """Test seti üzerinde değerlendirme yap"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader, desc="Test", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    return accuracy, all_preds, all_labels

def run():
    global PATHBASE
    global METRICS_PATH 
    global DATA_ROOT 
    global CHECKPOINT_PATH 
    global BATCH_SIZE 
    global NUM_EPOCHS
    global LEARNING_RATE 
    global DEVICE
    
    # log başlat
    log_file, timestamp = setup_logging()
    logging.info("="*80)
    logging.info(f"EĞITIM SESSION BAŞLADI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*80)
    
    logging.info(f"\n[YAPILANDIRMA]")
    logging.info(f"  - Cihaz: {DEVICE}")
    logging.info(f"  - CUDA Kullanılabilir: {torch.cuda.is_available()}")
    logging.info(f"  - Batch Size: {BATCH_SIZE}")
    logging.info(f"  - Epoch Sayısı: {NUM_EPOCHS}")
    logging.info(f"  - Learning Rate: {LEARNING_RATE}")
    
    # eğitim devam ettirme 
    resume_training = False
    if Path(CHECKPOINT_PATH).exists():
        logging.info(f"\n[!] Önceki checkpoint bulundu: {CHECKPOINT_PATH}")
        user_input = input("Eğitimi devam ettirmek istiyor musun? (e/h): ").strip().lower()
        resume_training = user_input in ['e', 'evet', 'y', 'yes']
    
    start_epoch = 0
    best_val_acc = 0
    
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
    
    # DataLoader oluştur
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # Her iterasyonda veri seti karıştırılır
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    logging.info(f"DataLoader Oluşturuldu")
    
    # MODELİ HAZIRLA 
    logging.info("\n[ADIM 3] Model Hazırlanıyor...")
    model = create_model(num_classes=NUM_EMOTIONS, pretrained=True)
    model.to(DEVICE)
    
    # Loss ve Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Checkpoint'ten devam ettir
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    if resume_training:
        start_epoch, best_val_acc, train_losses, train_accs, val_losses, val_accs = \
            load_checkpoint(CHECKPOINT_PATH, model, optimizer, DEVICE)
    
    # ========== EĞİTİM (DÖNGÜ) ==========
    logging.info(f"\n[ADIM 4] Model Eğitiliyor (Epoch {start_epoch + 1}-{NUM_EPOCHS})...\n")
    
    epoch = start_epoch
    
    while True:
        # Epoch range'i kontrol et ve işlemi gerçekleştir
        if epoch < NUM_EPOCHS:
            # Eğitim epoch'unu çalıştır
            logging.info(f"\n{'='*80}")
            logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            logging.info(f"{'='*80}")
            
            # Eğitim
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Doğrulama
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Detaylı log
            logging.info(f"  Eğitim   - Loss: {train_loss:.4f}, Doğruluk: {train_acc:.2f}%")
            logging.info(f"  Doğrulama - Loss: {val_loss:.4f}, Doğruluk: {val_acc:.2f}%")
            
            # En iyi modeli kaydet
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_emotion_model.pth')
                logging.info(f"  ✓ En iyi model kaydedildi (Doğruluk: {val_acc:.2f}%)")
            
            # Checkpoint'i kaydet (devam ettirmek için)
            save_checkpoint(CHECKPOINT_PATH, model, optimizer, epoch, best_val_acc,
                           train_losses, train_accs, val_losses, val_accs)
            
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
            
            logging.info(f"\n[ADIM 4-DEVAM] Model 4 Epoch Daha Eğitiliyor...\n")
            NUM_EPOCHS += 4
    
    # ========== TEST ==========
    logging.info("\n[ADIM 5] Model Test Ediliyor...")
    
    # En iyi modeli yükle
    model.load_state_dict(torch.load('best_emotion_model.pth'))
    test_acc, all_preds, all_labels = test(model, test_loader, DEVICE)
    
    logging.info(f"\n[SONUÇ] Test Doğruluğu: {test_acc:.2f}%")
    
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
    plt.plot(train_accs, label='Eğitim', linewidth=2)
    plt.plot(val_accs, label='Doğrulama', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk (%)')
    plt.legend()
    plt.title('Doğruluk Eğrisi')
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
        f.write(f"  - Test Doğruluğu: {test_acc:.2f}%\n")
        f.write(f"  - En İyi Doğrulama Doğruluğu: {best_val_acc:.2f}%\n")
        f.write(f"  - Final Eğitim Loss: {train_losses[-1]:.4f}\n")
        f.write(f"  - Final Doğrulama Loss: {val_losses[-1]:.4f}\n")
        f.write(f"  - Final Eğitim Doğruluğu: {train_accs[-1]:.2f}%\n")
        f.write(f"  - Final Doğrulama Doğruluğu: {val_accs[-1]:.2f}%\n\n")
        
        f.write(f"DOSYALAR:\n")
        f.write(f"  - Log Dosyası: {log_file}\n")
        f.write(f"  - Confusion Matrix: {confusion_matrix_file}\n")
        f.write(f"  - Eğitim Grafiği: {training_history_file}\n")
        f.write(f"  - Model: best_emotion_model.pth\n\n")
        
        f.write(f"DETAYLı RAPOR:\n")
        f.write(f"{report}\n")
    
    logging.info(f"[INFO] Özet Raporu kaydedildi: {summary_file}")
    logging.info("\n" + "="*80)
    logging.info(f"EĞITIM SESSİYONU TAMAMLANDI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*80)
    logging.info(f"\n📁 TÜM DOSYALAR:\n  - Log: {log_file}\n  - Özet: {summary_file}\n  - Matrix: {confusion_matrix_file}\n  - Grafik: {training_history_file}\n")
    
if __name__ == "__main__":
    run()
