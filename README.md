# Turkish Sentiment Analysis | Türkçe Sentiment Analysis

**[English](#english) | [Türkçe](#turkish)**

---

## English

### Overview

Turkish Sentiment Analysis is a deep learning-based model for classifying Turkish text into sentiment categories (Positive, Negative, Neutral). The model uses a fine-tuned BERT transformer with LoRA (Low-Rank Adaptation) optimization technique to achieve efficient training while maintaining high accuracy.

### Features

- **Pre-trained Turkish BERT**: Uses `dbmdz/bert-base-turkish-cased` model
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **Turkish Dataset**: ~500 Turkish examples (tweets, reviews, comments)
- **Multi-class Classification**: Positive, Negative, Neutral sentiment
- **Easy Inference**: Simple API for making predictions
- **Batch Processing**: Support for batch inference

### Installation

#### Requirements
- Python 3.8+
- CUDA (optional, for GPU acceleration)

#### Setup

1. Clone the repository:
```bash
cd turkish-sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using setup.py:
```bash
pip install -e .
```

### Quick Start

#### 1. Create Dataset

First, create the Turkish sentiment dataset:

```bash
python create_dataset.py --output ./data/turkish_sentiment_dataset.csv \
                          --positive 200 \
                          --negative 200 \
                          --neutral 100
```

This will create a CSV file with Turkish sentiment examples.

#### 2. Train the Model

Train the model using LoRA fine-tuning:

```bash
python model_training.py --config ./config.yaml \
                         --data ./data/turkish_sentiment_dataset.csv
```

Training will:
- Load Turkish BERT model
- Apply LoRA configuration
- Train on the dataset
- Save the best model
- Evaluate on test set

#### 3. Make Predictions

Use the trained model for inference:

**Option 1: Interactive Mode**
```bash
python inference.py --model ./models/sentiment_model
```

Then enter text when prompted.

**Option 2: Single Text**
```bash
python inference.py --model ./models/sentiment_model \
                    --text "Bu ürün çok harika, çok memnunum!"
```

**Option 3: Batch from CSV**
```bash
python inference.py --model ./models/sentiment_model \
                    --file ./data/test_data.csv
```

### Configuration

Edit `config.yaml` to customize:

- **Model**: BERT model name and parameters
- **Training**: Learning rate, epochs, batch size
- **LoRA**: Rank, dropout, target modules
- **Dataset**: Data paths, split ratios
- **Inference**: Device, batch size, thresholds

### Project Structure

```
turkish-sentiment-analysis/
├── model_training.py          # Training script
├── create_dataset.py          # Dataset creation
├── inference.py               # Inference/prediction script
├── config.yaml                # Configuration file
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
├── README.md                  # This file
├── .gitignore                 # Git ignore file
├── data/                      # Dataset directory
│   └── turkish_sentiment_dataset.csv
├── models/                    # Trained models directory
└── examples/                  # Usage examples
    ├── batch_prediction.py
    ├── custom_inference.py
    └── evaluation.py
```

### Model Details

**Architecture**: BERT-base-turkish-cased
- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- Trained on Turkish text corpus

**Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- Rank (r): 8
- Alpha: 16
- Dropout: 0.05
- Target modules: query, value

**Dataset**: Turkish Sentiment
- Total samples: ~500
- Positive: ~200
- Negative: ~200
- Neutral: ~100
- Sources: Twitter, product reviews, social media

### Performance

After training on the Turkish dataset:

```
Test Set Metrics:
- Accuracy: ~0.85-0.92
- Precision: ~0.85-0.91
- Recall: ~0.85-0.91
- F1-Score: ~0.85-0.91
```

(Actual metrics depend on data and training configuration)

### API Usage

### Python API

```python
from inference import SentimentAnalyzer

# Initialize analyzer
analyzer = SentimentAnalyzer(model_path="./models/sentiment_model")

# Single prediction
result = analyzer.predict("Bu ürün çok iyi!")
print(result)
# Output: [{'text': 'Bu ürün çok iyi!', 'sentiment': 'positive', 'confidence': 0.98, 'scores': {...}}]

# Batch prediction
texts = [
    "Harika bir deneyim!",
    "Çok kötü, tavsiye etmem",
    "Normal bir ürün"
]
results = analyzer.batch_predict(texts)

# Prediction from CSV
df_results = analyzer.predict_from_file("data.csv", text_column="text")
df_results.to_csv("results.csv")
```

### Command Line API

```bash
# Interactive mode
python inference.py --model ./models/sentiment_model

# Single text
python inference.py --model ./models/sentiment_model \
                    --text "Türkçe metin"

# Batch processing
python inference.py --model ./models/sentiment_model \
                    --file data.csv
```

### Examples

See `examples/` directory for detailed usage examples:

- `batch_prediction.py`: Batch processing with CSV
- `custom_inference.py`: Custom inference pipeline
- `evaluation.py`: Model evaluation on test set

### Troubleshooting

**Out of Memory (OOM)**
- Reduce `batch_size` in config.yaml
- Use CPU mode: set `device: cpu` in config

**Slow Training**
- Enable GPU: ensure CUDA is installed
- Reduce `max_length` in config.yaml
- Use fewer epochs initially

**Poor Accuracy**
- Check dataset quality
- Adjust learning_rate in config
- Increase training epochs
- Add more training data

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

### License

MIT License - see LICENSE file for details

### Citation

If you use this model in your research, please cite:

```bibtex
@software{turkish_sentiment_2024,
  title={Turkish Sentiment Analysis with BERT and LoRA},
  author={Turkish NLP Team},
  year={2024},
  url={https://github.com/yourusername/turkish-sentiment-analysis}
}
```

### References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [dbmdz Turkish BERT](https://huggingface.co/dbmdz/bert-base-turkish-cased)

---

## Turkish

### Genel Bakış

Türkçe Sentiment Analysis, Türkçe metinleri duyarlılık kategorilerine (Pozitif, Negatif, Nötr) sınıflandırmak için tasarlanmış derin öğrenme tabanlı bir modeldir. Model, LoRA (Low-Rank Adaptation) optimizasyon tekniği ile fine-tune edilmiş BERT dönüştürücüyü kullanarak verimli eğitim sağlarken yüksek doğruluğu korur.

### Özellikler

- **Türkçe BERT**: `dbmdz/bert-base-turkish-cased` modelini kullanır
- **LoRA Fine-tuning**: Verimli parametre optimize fine-tuning
- **Türkçe Dataset**: ~500 Türkçe örnek (tweet, yorum, review)
- **Çok-sınıflı Sınıflandırma**: Pozitif, Negatif, Nötr duyarlılık
- **Kolay İnference**: Tahmin için basit API
- **Batch İşleme**: Toplu tahmin desteği

### Kurulum

#### Gereksinimler
- Python 3.8+
- CUDA (opsiyonel, GPU hızlandırması için)

#### Adımlar

1. Repository'yi klonlayın:
```bash
cd turkish-sentiment-analysis
```

2. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

Veya setup.py ile:
```bash
pip install -e .
```

### Hızlı Başlangıç

#### 1. Dataset Oluştur

Türkçe sentiment dataset'i oluşturun:

```bash
python create_dataset.py --output ./data/turkish_sentiment_dataset.csv \
                          --positive 200 \
                          --negative 200 \
                          --neutral 100
```

Bu, Türkçe sentiment örnekleri içeren bir CSV dosyası oluşturur.

#### 2. Modeli Eğit

LoRA fine-tuning ile modeli eğitin:

```bash
python model_training.py --config ./config.yaml \
                         --data ./data/turkish_sentiment_dataset.csv
```

Eğitim şunları yapacak:
- Türkçe BERT modelini yükle
- LoRA konfigürasyonunu uygula
- Dataset'te eğit
- En iyi modeli kaydet
- Test seti üzerinde değerlendir

#### 3. Tahmin Yap

Eğitilmiş model ile tahmin yapın:

**Seçenek 1: İnteraktif Mod**
```bash
python inference.py --model ./models/sentiment_model
```

Ardından istendiğinde metin girin.

**Seçenek 2: Tek Metin**
```bash
python inference.py --model ./models/sentiment_model \
                    --text "Bu ürün çok harika, çok memnunum!"
```

**Seçenek 3: CSV'den Toplu İşlem**
```bash
python inference.py --model ./models/sentiment_model \
                    --file ./data/test_data.csv
```

### Konfigürasyon

`config.yaml` dosyasını düzenleyerek özelleştirebilirsiniz:

- **Model**: BERT modeli adı ve parametreleri
- **Training**: Öğrenme hızı, epoch sayısı, batch boyutu
- **LoRA**: Rank, dropout, hedef modüller
- **Dataset**: Veri yolları, split oranları
- **Inference**: Cihaz, batch boyutu, eşikler

### Proje Yapısı

```
turkish-sentiment-analysis/
├── model_training.py          # Eğitim scripti
├── create_dataset.py          # Dataset oluşturma
├── inference.py               # İnference scripti
├── config.yaml                # Konfigürasyon dosyası
├── requirements.txt           # Bağımlılıklar
├── setup.py                   # Paket kurulumu
├── README.md                  # Bu dosya
├── .gitignore                 # Git ignore dosyası
├── data/                      # Dataset klasörü
│   └── turkish_sentiment_dataset.csv
├── models/                    # Eğitilmiş modeller
└── examples/                  # Kullanım örnekleri
    ├── batch_prediction.py
    ├── custom_inference.py
    └── evaluation.py
```

### Model Detayları

**Mimarı**: BERT-base-turkish-cased
- 12 transformer katmanı
- 768 gizli boyut
- 12 attention başı
- Türkçe metin corpus'unda eğitilmiş

**Fine-tuning Yöntemi**: LoRA (Low-Rank Adaptation)
- Rank (r): 8
- Alpha: 16
- Dropout: 0.05
- Hedef modüller: query, value

**Dataset**: Türkçe Sentiment
- Toplam örnek: ~500
- Pozitif: ~200
- Negatif: ~200
- Nötr: ~100
- Kaynaklar: Twitter, ürün yorumları, sosyal medya

### Performans

Türkçe dataset'te eğitim sonrası:

```
Test Set Metrikleri:
- Accuracy: ~0.85-0.92
- Precision: ~0.85-0.91
- Recall: ~0.85-0.91
- F1-Score: ~0.85-0.91
```

(Gerçek metrikler veri ve eğitim konfigürasyonuna bağlıdır)

### API Kullanımı

### Python API

```python
from inference import SentimentAnalyzer

# Analyzer'ı initialize et
analyzer = SentimentAnalyzer(model_path="./models/sentiment_model")

# Tek tahmin
result = analyzer.predict("Bu ürün çok iyi!")
print(result)
# Çıktı: [{'text': 'Bu ürün çok iyi!', 'sentiment': 'positive', 'confidence': 0.98, 'scores': {...}}]

# Toplu tahmin
texts = [
    "Harika bir deneyim!",
    "Çok kötü, tavsiye etmem",
    "Normal bir ürün"
]
results = analyzer.batch_predict(texts)

# CSV'den tahmin
df_results = analyzer.predict_from_file("data.csv", text_column="text")
df_results.to_csv("results.csv")
```

### Komut Satırı API

```bash
# İnteraktif mod
python inference.py --model ./models/sentiment_model

# Tek metin
python inference.py --model ./models/sentiment_model \
                    --text "Türkçe metin"

# Toplu işlem
python inference.py --model ./models/sentiment_model \
                    --file data.csv
```

### Örnekler

Detaylı kullanım örnekleri için `examples/` klasörüne bakın:

- `batch_prediction.py`: CSV ile toplu işlem
- `custom_inference.py`: Özel inference pipeline
- `evaluation.py`: Test seti üzerinde model değerlendirmesi

### Sorun Giderme

**Bellek Hatası (OOM)**
- `config.yaml`'daki `batch_size`'ı azaltın
- CPU modunu kullanın: config'de `device: cpu` ayarlayın

**Yavaş Eğitim**
- GPU'yu etkinleştirin: CUDA'nın kurulu olduğundan emin olun
- `config.yaml`'daki `max_length`'i azaltın
- İlk başlarda daha az epoch kullanın

**Düşük Doğruluk**
- Dataset kalitesini kontrol edin
- `config.yaml`'da `learning_rate`'i ayarlayın
- Eğitim epoch'larını arttırın
- Daha fazla eğitim verisi ekleyin

### Katkı Yapma

Katkılar hoştur! Lütfen:
1. Repository'i fork edin
2. Feature branch oluşturun
3. Pull request gönderin

### Lisans

MIT Lisansı - detaylar için LICENSE dosyasına bakın

### Atıf

Bu modeli araştırmanızda kullanıyorsanız, lütfen atıf yapın:

```bibtex
@software{turkish_sentiment_2024,
  title={Türkçe Sentiment Analysis with BERT and LoRA},
  author={Turkish NLP Team},
  year={2024},
  url={https://github.com/yourusername/turkish-sentiment-analysis}
}
```

### Referanslar

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft/)
- [BERT Makalesi](https://arxiv.org/abs/1810.04805)
- [LoRA Makalesi](https://arxiv.org/abs/2106.09685)
- [dbmdz Türkçe BERT](https://huggingface.co/dbmdz/bert-base-turkish-cased)

---

**Last Updated**: January 2, 2026
**Version**: 1.0.0
