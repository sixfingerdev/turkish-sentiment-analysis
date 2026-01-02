# Turkish Sentiment Analysis Project - Complete Guide
# Türkçe Sentiment Analysis Projesi - Tam Kılavuz

## PROJECT STRUCTURE / PROJE YAPISI

```
turkish-sentiment-analysis/
│
├── 📄 Core Files / Çekirdek Dosyalar
│   ├── model_training.py          # Model eğitim scripti
│   ├── create_dataset.py          # Dataset oluşturma scripti
│   ├── inference.py               # Tahmin scripti
│   ├── quick_start.py             # Hızlı başlama scripti
│   └── setup.py                   # Pip paketi kurulumu
│
├── ⚙️ Configuration / Konfigürasyon
│   ├── config.yaml                # Tüm ayarlar
│   ├── requirements.txt           # Python bağımlılıkları
│   └── .gitignore                 # Git ignore kuralları
│
├── 📚 Documentation / Dokümantasyon
│   ├── README.md                  # Proje readme (Türkçe & İngilizce)
│   └── GUIDE.md                   # Bu dosya
│
├── 📁 data/                       # Veri seti klasörü
│   └── turkish_sentiment_dataset.csv  # Türkçe sentiment verisi (~200 örnek)
│
├── 🤖 models/                     # Eğitilmiş modeller (eğitim sonrası)
│   └── sentiment_model/
│       ├── pytorch_model.bin
│       ├── config.json
│       └── tokenizer files
│
└── 📋 examples/                   # Kullanım örnekleri
    ├── batch_prediction.py        # Toplu tahmin örneği
    ├── custom_inference.py        # Özel inference örneği
    └── evaluation.py              # Model değerlendirme örneği
```

## QUICK START / HIZLI BAŞLAMA

### 1️⃣ INSTALLATION / KURULUM

```bash
# Project directory'ye girin / Go to project directory
cd turkish-sentiment-analysis

# Bağımlılıkları yükleyin / Install dependencies
pip install -r requirements.txt

# VEYA / OR - Hızlı kurulum scripti / Quick setup script
python quick_start.py
```

### 2️⃣ CREATE DATASET / DATASET OLUŞTUR

```bash
# Otomatik olarak 500 örnek (200 pozitif, 200 negatif, 100 nötr) oluştur
python create_dataset.py \
    --output ./data/turkish_sentiment_dataset.csv \
    --positive 200 \
    --negative 200 \
    --neutral 100

# Özel sayıda örnek oluştur / Create custom number of examples
python create_dataset.py --positive 300 --negative 300 --neutral 200
```

### 3️⃣ TRAIN MODEL / MODELI EĞİT

```bash
# Modeli eğit (GPU varsa otomatik kullanılır)
python model_training.py \
    --config ./config.yaml \
    --data ./data/turkish_sentiment_dataset.csv

# Eğitim süresi: ~5-10 dakika (GPU ile) / 30+ dakika (CPU ile)
# Training time: ~5-10 minutes (with GPU) / 30+ minutes (with CPU)
```

### 4️⃣ MAKE PREDICTIONS / TAHMIN YAP

#### A) İnteraktif Mod / Interactive Mode
```bash
python inference.py --model ./models/sentiment_model

# Ardından metin girin / Then enter text when prompted
# Çıkmak için 'quit' yazın / Type 'quit' to exit
```

#### B) Tek Metin Tahmin / Single Text Prediction
```bash
python inference.py \
    --model ./models/sentiment_model \
    --text "Bu ürün çok harika, çok memnunum!"

# Output / Çıktı:
# Sentiment: POSITIVE
# Confidence: 0.9523
```

#### C) CSV Dosyasından Toplu Tahmin / Batch from CSV
```bash
python inference.py \
    --model ./models/sentiment_model \
    --file ./data/test_data.csv

# Sonuçlar kaydedilir / Results saved: test_data_predictions.csv
```

## DETAILED USAGE / DETAYLI KULLANIM

### Dataset Oluşturma / Creating Dataset

```python
from create_dataset import create_dataset

# Türkçe veri seti oluştur
df = create_dataset(
    output_path="./data/turkish_sentiment.csv",
    positive_samples=200,
    negative_samples=200,
    neutral_samples=100,
    seed=42
)

# Dataset'i pandas ile yükle
import pandas as pd
df = pd.read_csv("./data/turkish_sentiment.csv")
print(df.head())
print(df["sentiment"].value_counts())
```

### Model Eğitme / Training Model

```python
from model_training import TurkishSentimentTrainer

# Trainer'ı initialize et
trainer = TurkishSentimentTrainer(config_path="./config.yaml")

# Modeli eğit
trainer.train(data_path="./data/turkish_sentiment_dataset.csv")

# Eğitilmiş model ve tokenizer otomatik kaydedilir
# ./models/sentiment_model/ klasöründe
```

### Tahmin Yapma / Making Predictions

```python
from inference import SentimentAnalyzer

# Analyzer oluştur
analyzer = SentimentAnalyzer(
    model_path="./models/sentiment_model",
    config_path="./config.yaml"
)

# ✨ Tek metin tahmin
result = analyzer.predict("Bu ürün çok iyi!")
print(result[0]['sentiment'])     # Output: 'positive'
print(result[0]['confidence'])    # Output: 0.95 (örnek)

# ✨ Toplu tahmin
texts = [
    "Harika bir deneyim!",
    "Çok kötü, tavsiye etmem",
    "Normal bir ürün"
]
results = analyzer.batch_predict(texts)

# ✨ CSV dosyasından tahmin
df_results = analyzer.predict_from_file(
    "./data/test_data.csv",
    text_column="text"
)
df_results.to_csv("predictions.csv", index=False)
```

## CONFIGURATION / KONFIGÜRASYON

config.yaml dosyasını düzenleyin:

```yaml
model:
  name: "dbmdz/bert-base-turkish-cased"  # Model ismi
  max_length: 128                         # Max token uzunluğu
  output_dir: "./models/sentiment_model" # Model kayıt yeri

training:
  num_epochs: 5                    # Eğitim epoch sayısı
  batch_size: 16                   # Batch boyutu (GPU RAM'e göre)
  learning_rate: 2e-5              # Öğrenme hızı
  weight_decay: 0.01               # Regularization
  warmup_steps: 100                # Warmup step sayısı

lora:
  use_lora: true                   # LoRA fine-tuning kullan
  r: 8                             # Rank (hafızayı azaltır)
  lora_alpha: 16                   # Alpha parametresi
  lora_dropout: 0.05               # Dropout oranı

inference:
  device: "cuda"                   # GPU ("cuda") veya CPU ("cpu")
  batch_size: 32                   # Tahmin batch boyutu
```

## EXAMPLES / ÖRNEKLER

### Örnek 1: Batch Prediction / Toplu Tahmin

```bash
python examples/batch_prediction.py
```

### Örnek 2: Custom Inference / Özel Tahmin

```bash
python examples/custom_inference.py
```

### Örnek 3: Model Evaluation / Model Değerlendirme

```bash
python examples/evaluation.py
```

## TROUBLESHOOTING / SORUN GIDERME

### Problem 1: Out of Memory (OOM)
```
Çözüm / Solution:
1. config.yaml'da batch_size'ı azaltın (16 → 8 veya 4)
2. max_length'i azaltın (128 → 64)
3. CPU mode kullanın: device: "cpu"
```

### Problem 2: Slow Training
```
Çözüm / Solution:
1. GPU'nun kurulu olduğundan emin olun
2. CUDA version'ını kontrol edin
3. PyTorch'u GPU versiyonu ile kurun:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problem 3: Model Not Found
```
Çözüm / Solution:
1. Modeli eğittiğinizden emin olun:
   python model_training.py --config config.yaml --data data/turkish_sentiment_dataset.csv
2. Model yolu doğru mu kontrol edin:
   ./models/sentiment_model/
3. Model dosyalarını kontrol edin:
   - pytorch_model.bin
   - config.json
   - tokenizer files
```

### Problem 4: Encoding Issues
```
Çözüm / Solution:
Dosyaları UTF-8 encoding'i ile açın:

Python'da:
with open('file.csv', encoding='utf-8') as f:
    ...

Terminal'de:
set PYTHONIOENCODING=utf-8
```

## PERFORMANCE METRICS / PERFORMANS METRİKLERİ

Model eğitim sonrası beklenen metrikler:

```
Accuracy:  85-92%
Precision: 85-91%
Recall:    85-91%
F1-Score:  85-91%
```

(Metrikler dataset ve hyperparameters'a bağlıdır)

## API REFERENCE / API REFERANSI

### SentimentAnalyzer Class

```python
from inference import SentimentAnalyzer

# Initialize
analyzer = SentimentAnalyzer(
    model_path: str = "./models/sentiment_model",
    config_path: str = "./config.yaml"
)

# Methods

# predict(texts, return_scores=True) -> List[Dict]
# Tek veya birden fazla metin için tahmin yapan
results = analyzer.predict("Bu ürün çok iyi!")

# batch_predict(texts, batch_size=32) -> List[Dict]
# Toplu tahmin yapan
results = analyzer.batch_predict(["Text 1", "Text 2", "Text 3"])

# predict_from_file(file_path, text_column="text") -> pd.DataFrame
# CSV dosyasından tahmin yapan
df_results = analyzer.predict_from_file("data.csv")
```

## OUTPUT FORMAT / ÇIKTI FORMATI

```python
# Tek tahmin çıktısı / Single prediction output:
{
    "text": "Bu ürün çok harika!",
    "sentiment": "positive",
    "confidence": 0.9523,
    "scores": {
        "negative": 0.0123,
        "neutral": 0.0354,
        "positive": 0.9523
    }
}
```

## SYSTEM REQUIREMENTS / SİSTEM GEREKSİNİMLERİ

- Python 3.8+
- RAM: Min 4GB (GPU varsa 2GB yeterli)
- GPU: CUDA 11.8+ (opsiyonel, eğitim için önerilir)
- Disk: Min 1GB (model + veri seti için)

## FILE SIZES / DOSYA BOYUTLARı

- Model: ~440 MB (BERT base)
- Dataset: ~50 KB (500 örnek)
- LoRA Weights: ~2 MB (eğer sadece LoRA kaydedilirse)

## TIPS & TRICKS / İPUÇLARı

1. **Fine-tuning için daha fazla veri ekleyin / Add more data for better results:**
   - Dataset boyutunu 1000+ örneğe çıkarın
   - Çeşitli kaynaklardan Türkçe metinler kullanın

2. **Daha iyi performans için:**
   - Learning rate'ı düşürün (2e-5 → 1e-5)
   - Epoch sayısını arttırın (5 → 10)
   - Batch size'ı arttırın (eğer RAM varsa)

3. **Hızlı eğitim için:**
   - LoRA rank'ını azaltın (8 → 4)
   - Max length'i azaltın (128 → 64)
   - Batch size'ı azaltın

4. **Üretim ortamı için:**
   - Model'i quantize edin (boyut: 110 MB)
   - ONNX format'ına çevirin (hız)
   - Model'i serve edin (FastAPI ile)

## NEXT STEPS / SONRAKI ADIMLAR

1. ✓ Dataset oluşturdunuz / You created dataset
2. ✓ Modeli eğittiniz / You trained the model
3. → Tahmin yapın / Make predictions
4. → Modeli optimize edin / Optimize the model
5. → Üretim ortamına deploy edin / Deploy to production
6. → Model versiyonlaması yönetin / Manage model versions

## SUPPORT & RESOURCES / DESTEK & KAYNAKLAR

- **Hugging Face**: https://huggingface.co/
- **Transformers Docs**: https://huggingface.co/docs/transformers/
- **PEFT Documentation**: https://huggingface.co/docs/peft/
- **Turkish BERT**: https://huggingface.co/dbmdz/bert-base-turkish-cased

## VERSION / VERSİYON

- Project Version: 1.0.0
- Created: January 2, 2026
- Last Updated: January 2, 2026

---

**Happy coding! / İyi kodlamalar!**
