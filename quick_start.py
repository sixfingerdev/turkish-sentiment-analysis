#!/usr/bin/env python
"""
Turkish Sentiment Analysis - Quick Start Script
Türkçe Sentiment Analysis - Hızlı Başlama Scripti

Bu script tüm adımları otomatik olarak çalıştırır
This script runs all steps automatically
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Komutu çalıştır"""
    print("\n" + "="*80)
    print(f"▶ {description}")
    print("="*80)
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"✗ Hata / Error: {description}")
        return False
    
    print(f"✓ Başarılı / Success: {description}")
    return True

def main():
    """Ana function"""
    
    print("\n" + "="*80)
    print("Turkish Sentiment Analysis - Project Setup")
    print("Türkçe Sentiment Analysis - Proje Kurulumu")
    print("="*80)
    
    steps = [
        (
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing dependencies / Bağımlılıkları yükleme"
        ),
        (
            f"{sys.executable} create_dataset.py --positive 200 --negative 200 --neutral 100",
            "Creating dataset / Dataset oluşturma"
        ),
        (
            f"{sys.executable} model_training.py --config ./config.yaml --data ./data/turkish_sentiment_dataset.csv",
            "Training model / Modeli eğitme (Bu adım uzun sürebilir / This step may take a while)"
        ),
    ]
    
    completed = 0
    
    for cmd, desc in steps:
        if run_command(cmd, desc):
            completed += 1
        else:
            print(f"\n✗ İşlem durdu / Process stopped at step: {desc}")
            break
    
    print("\n" + "="*80)
    print(f"Completion Status: {completed}/{len(steps)} steps completed")
    print(f"Tamamlanma Durumu: {completed}/{len(steps)} adım tamamlandı")
    print("="*80)
    
    if completed == len(steps):
        print("\n✓ Kurulum tamamlandı! / Setup completed!")
        print("\nSonraki adımlar / Next steps:")
        print("1. Tahmin yapın / Make predictions:")
        print(f"   {sys.executable} inference.py --model ./models/sentiment_model --text 'Türkçe metin'")
        print("\n2. Toplu işlem / Batch processing:")
        print(f"   {sys.executable} inference.py --model ./models/sentiment_model --file data.csv")
        print("\n3. Model değerlendirme / Evaluate model:")
        print(f"   {sys.executable} examples/evaluation.py")
        print("\n4. Toplu tahmin örneği / Batch prediction example:")
        print(f"   {sys.executable} examples/batch_prediction.py")
        print("\n5. Özel inference örneği / Custom inference example:")
        print(f"   {sys.executable} examples/custom_inference.py")
    else:
        print("\n✗ Kurulum tamamlanmadı / Setup incomplete")
        print("Lütfen yukarıdaki hataları kontrol edin / Please check errors above")

if __name__ == "__main__":
    main()
