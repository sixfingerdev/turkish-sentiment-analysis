"""
Batch Prediction Örneği
Batch Prediction Example

CSV dosyasından tahmin yapma
Making predictions from CSV file
"""

import pandas as pd
from pathlib import Path
from inference import SentimentAnalyzer

def batch_predict_example():
    """CSV dosyasından toplu tahmin örneği"""
    
    # Test verisi oluştur
    test_data = {
        'text': [
            'Bu ürün çok harika, tam beklediğim gibi!',
            'Çok kötü, geri iade ettim!',
            'Normal bir ürün, fiyatı uygun',
            'Mükemmel hizmet, çok memnunum!',
            'Hiç tavsiye etmem, para israfı!',
            'İyi ürün, hızlı teslimat',
            'Kötü kalite, çok hayal kırıklığı',
            'Beklenildiği gibi, iyiydi',
            'Çok iyi, herkese tavsiye ediyorum!',
            'Hemen bozuldu, uzun süre dayanmadı'
        ]
    }
    
    df_test = pd.DataFrame(test_data)
    
    # Test dosyasını kaydet
    test_file = "./data/test_examples.csv"
    df_test.to_csv(test_file, index=False, encoding='utf-8')
    print(f"Test dosyası oluşturuldu / Test file created: {test_file}")
    
    # Analyzer oluştur
    print("\nAnalyzer yükleniyor / Loading analyzer...")
    analyzer = SentimentAnalyzer(
        model_path="./models/sentiment_model",
        config_path="./config.yaml"
    )
    
    # Toplu tahmin yap
    print("\nTahmin yapılıyor / Making predictions...")
    df_results = analyzer.predict_from_file(test_file)
    
    # Sonuçları göster
    print("\n" + "="*80)
    print("Prediction Results / Tahmin Sonuçları")
    print("="*80)
    print(df_results)
    
    # Sonuçları kaydet
    output_file = "./outputs/batch_predictions.csv"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nSonuçlar kaydedildi / Results saved: {output_file}")
    
    # İstatistikler
    print("\n" + "="*80)
    print("Statistics / İstatistikler")
    print("="*80)
    print(f"Toplam tahmin / Total predictions: {len(df_results)}")
    print("\nSentiment Dağılımı / Sentiment Distribution:")
    print(df_results['sentiment'].value_counts())
    
    print("\nOrtalama Confidence / Average Confidence:")
    print(f"Pozitif / Positive: {df_results[df_results['sentiment']=='positive']['confidence'].mean():.4f}")
    print(f"Negatif / Negative: {df_results[df_results['sentiment']=='negative']['confidence'].mean():.4f}")
    print(f"Nötr / Neutral: {df_results[df_results['sentiment']=='neutral']['confidence'].mean():.4f}")

if __name__ == "__main__":
    batch_predict_example()
