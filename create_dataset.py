"""
Türkçe Sentiment Analysis için veri seti oluşturma scripti
Turkish Sentiment Analysis Dataset Creation Script
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple

# Örnek Türkçe tweet ve yorum verisi
TURKISH_SENTIMENT_SAMPLES = {
    "positive": [
        "Bu ürün çok harika, tam beklediğim gibi!",
        "Mükemmel hizmet, çok memnun kaldım!",
        "Fiyat-performans oranı çok iyi, tavsiye ederim!",
        "Kargo hızlı geldi, ürün çok güzel!",
        "Harika bir deneyim yaşadım, teşekkürler!",
        "Çok kaliteli ürün, beğenmedim değil!",
        "Beklentimi aştı, çok memnunum!",
        "Kusursuz bir satış deneyimi!",
        "Ürün kalitesi beklentimi çok aştı!",
        "En güzel seçim, kesinlikle satın alırım!",
        "Müşteri hizmetleri çok yardımcı oldu!",
        "Gerçekten değer veriyor, super!",
        "Hızlı teslimat, kaliteli ürün!",
        "Çok beğendim, arkadaşlarıma tavsiye ettim!",
        "Fiyatı uygun, kalitesi harika!",
        "Harika bir seçim yaptım!",
        "Çok memnun, tekrar alacağım!",
        "Kalitesi kontrol edilmiş, kusursuz!",
        "Hemen siparişi verdim, çok iyiydi!",
        "Efsane ürün, efsane hizmet!",
    ],
    "negative": [
        "Çok kötü ürün, para israfı!",
        "Beklentimi çok karşılamadı, hayal kırıklığı!",
        "Kötü kalite, paran değmez!",
        "Hızlı bozuldu, uzun süre dayanmadı!",
        "Müşteri hizmetleri hiç yardımcı olmadı!",
        "Öyle iyi değilmiş, çok hayal kırıklığı!",
        "Kalitesiz ürün, geri iade ettim!",
        "Çok pahalı, bu kadar kötü olması garip!",
        "Hiç tavsiye etmem, paranızı başka yerde harcayın!",
        "Tanımlanan özelliklerle hiç uymuyor!",
        "Teslimat çok geç, ürün de bozuk geldi!",
        "Çok hayal kırıklığı verdi!",
        "Hiç değmedi, geri para iade etsin!",
        "Çok kötü bir seçim yaptım!",
        "Düşük kalite, uzun dayanmaz!",
        "Asla tavsiye etmem arkadaşlara!",
        "Ürün tamamen çöp, saçma sapan!",
        "Kurulum çok zor, sonra çalışmıyor!",
        "Parasını almaya değmez!",
        "Tek kelimeyle berbat!",
    ],
    "neutral": [
        "Ürün standart özelliklere sahip",
        "Fiyatı orta seviyede, kalitesi de öyle",
        "Ne iyi ne kötü, sıradan bir ürün",
        "Beklenildiği gibi, fazla bir şey yok",
        "Açıklamadaki gibi, normal bir ürün",
        "Farklı bir şey yok, tipik bir seçim",
        "Ortalama bir deneyim yaşadım",
        "Ürün beklentileri karşılıyor, başka söyleyecek bir şey yok",
        "Standart kalite, standart fiyat",
        "Bilinen bir marka, güvenilir seçim",
        "Basit bir ürün, işini yapıyor",
        "Ne özel ne de kötü, sade",
        "Normalin içinde, uygun fiyatlı",
        "Teknik özellikler iyi, kullanımı kolay",
        "Ürün iyi paketlenmiş geldi",
        "Beklentilerime tam uygun",
        "Orta fiyat aralığında, makul seçim",
        "Genel olarak memnun, bir şikayetim yok",
        "Normal bir satın alma deneyimi",
        "Standart ürün, standart hizmet",
        "Ürün amacını yerine getiriyor",
        "İçerik açıklamayla uyumlu",
        "Kalitesi kötü değil, iyi de değil",
        "Orta seviye memnuniyet",
        "Beklenildiği şekilde geldi",
    ]
}

def create_dataset(
    output_path: str = "./data/turkish_sentiment_dataset.csv",
    positive_samples: int = 200,
    negative_samples: int = 200,
    neutral_samples: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    Türkçe sentiment dataset oluştur
    Create Turkish sentiment dataset
    
    Args:
        output_path: Çıktı dosyası yolu / Output file path
        positive_samples: Pozitif örnek sayısı / Number of positive samples
        negative_samples: Negatif örnek sayısı / Number of negative samples
        neutral_samples: Nötr örnek sayısı / Number of neutral samples
        seed: Rastgele tohum / Random seed
    
    Returns:
        pd.DataFrame: Oluşturulan veri seti / Created dataset
    """
    
    np.random.seed(seed)
    
    data = []
    
    # Pozitif örnekler
    for i in range(positive_samples):
        text = np.random.choice(TURKISH_SENTIMENT_SAMPLES["positive"])
        data.append({"text": text, "sentiment": "positive"})
    
    # Negatif örnekler
    for i in range(negative_samples):
        text = np.random.choice(TURKISH_SENTIMENT_SAMPLES["negative"])
        data.append({"text": text, "sentiment": "negative"})
    
    # Nötr örnekler
    for i in range(neutral_samples):
        text = np.random.choice(TURKISH_SENTIMENT_SAMPLES["neutral"])
        data.append({"text": text, "sentiment": "neutral"})
    
    # DataFrame oluştur
    df = pd.DataFrame(data)
    
    # Karıştır
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Klasör oluştur
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Kaydet
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"Dataset oluşturuldu / Dataset created: {output_path}")
    print(f"Toplam örnek / Total samples: {len(df)}")
    print(f"\nSınıf dağılımı / Class distribution:")
    print(df["sentiment"].value_counts())
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Türkçe Sentiment Analysis Dataset oluştur"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/turkish_sentiment_dataset.csv",
        help="Çıktı dosyası yolu"
    )
    parser.add_argument(
        "--positive",
        type=int,
        default=200,
        help="Pozitif örnek sayısı"
    )
    parser.add_argument(
        "--negative",
        type=int,
        default=200,
        help="Negatif örnek sayısı"
    )
    parser.add_argument(
        "--neutral",
        type=int,
        default=100,
        help="Nötr örnek sayısı"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgele tohum"
    )
    
    args = parser.parse_args()
    
    create_dataset(
        output_path=args.output,
        positive_samples=args.positive,
        negative_samples=args.negative,
        neutral_samples=args.neutral,
        seed=args.seed
    )
