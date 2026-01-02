"""
Model Değerlendirme Örneği
Model Evaluation Example

Test seti üzerinde model performansı değerlendirme
Evaluate model performance on test set
"""

import sys
sys.path.insert(0, '../')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from inference import SentimentAnalyzer

def evaluate_model():
    """Model performansını değerlendir"""
    
    # Dataset yükle
    data_file = "./data/turkish_sentiment_dataset.csv"
    
    if not Path(data_file).exists():
        print(f"Dataset bulunamadı / Dataset not found: {data_file}")
        print("Önce dataset oluşturun / First create dataset: python create_dataset.py")
        return
    
    df = pd.read_csv(data_file)
    
    # Test seti için son 100 örneği kullan
    test_df = df.tail(100).reset_index(drop=True)
    
    print(f"Test seti boyutu / Test set size: {len(test_df)}")
    print(f"Sınıf dağılımı / Class distribution:")
    print(test_df['sentiment'].value_counts())
    
    # Analyzer oluştur
    print("\nAnalyzer yükleniyor / Loading analyzer...")
    analyzer = SentimentAnalyzer(
        model_path="./models/sentiment_model",
        config_path="./config.yaml"
    )
    
    # Tahminler yap
    print("Tahminler yapılıyor / Making predictions...")
    texts = test_df['text'].tolist()
    results = analyzer.batch_predict(texts)
    
    # Sonuçları DataFrame'e dönüştür
    df_results = pd.DataFrame(results)
    
    # True labels ve predicted labels
    y_true = test_df['sentiment'].tolist()
    y_pred = df_results['sentiment'].tolist()
    
    # Metrikleri hesapla
    print("\n" + "="*80)
    print("Model Evaluation Results / Model Değerlendirme Sonuçları")
    print("="*80)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    print(f"\nOverall Metrics / Genel Metrikler:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Per-class metrics
    print(f"\nPer-class Metrics / Sınıf Bazında Metrikler:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])
    
    print("\nConfusion Matrix / Karışıklık Matrisi:")
    print(cm)
    
    # Visualization
    print("\nVisualizations being created / Grafikler oluşturuluyor...")
    
    Path("./outputs").mkdir(parents=True, exist_ok=True)
    
    # Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive'],
                ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix - Turkish Sentiment Analysis')
    plt.tight_layout()
    plt.savefig('./outputs/confusion_matrix.png', dpi=100)
    print("✓ Confusion matrix saved / Kaydedildi: ./outputs/confusion_matrix.png")
    
    # Sentiment Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    sentiment_true = pd.Series(y_true).value_counts()
    sentiment_pred = pd.Series(y_pred).value_counts()
    
    sentiment_true.plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('True Sentiment Distribution')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel('Sentiment')
    
    sentiment_pred.plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('Predicted Sentiment Distribution')
    axes[1].set_ylabel('Count')
    axes[1].set_xlabel('Sentiment')
    
    plt.tight_layout()
    plt.savefig('./outputs/sentiment_distribution.png', dpi=100)
    print("✓ Sentiment distribution saved / Kaydedildi: ./outputs/sentiment_distribution.png")
    
    # Confidence by Sentiment
    df_results['true_sentiment'] = y_true
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for sentiment in ['negative', 'neutral', 'positive']:
        confidences = df_results[df_results['true_sentiment'] == sentiment]['confidence']
        ax.hist(confidences, label=sentiment, alpha=0.7, bins=15)
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Distribution by True Sentiment')
    ax.legend()
    plt.tight_layout()
    plt.savefig('./outputs/confidence_distribution.png', dpi=100)
    print("✓ Confidence distribution saved / Kaydedildi: ./outputs/confidence_distribution.png")
    
    # Detaylı sonuçları kaydet
    output_file = './outputs/evaluation_results.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ Detailed results saved / Kaydedildi: {output_file}")
    
    print("\n" + "="*80)
    print("Evaluation Complete / Değerlendirme Tamamlandı")
    print("="*80)

if __name__ == "__main__":
    evaluate_model()
