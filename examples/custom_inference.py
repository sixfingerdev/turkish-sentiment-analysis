"""
Özel İnference Örneği
Custom Inference Example

Detaylı tahmin örneği
Detailed prediction example
"""

import sys
sys.path.insert(0, '../')

from inference import SentimentAnalyzer
import json

def custom_inference_example():
    """Detaylı tahmin örneği"""
    
    # Örnek Türkçe metinler
    test_texts = [
        "Bu kitap gerçekten harika, tüm sayfaları okuduktan sonra çok etkilendim!",
        "Uygulama çok yavaş ve sık sık çöküyor, hiç tavsiye etmem.",
        "Ürün normal, ne kötü ne iyi ama fiyatı uygun.",
        "En iyi ürün, kesinlikle tekrar alacağım!",
        "Çok hayal kırıklığı, resimdeki gibi değilmiş.",
    ]
    
    # Analyzer oluştur
    print("İnference modeli yükleniyor / Loading inference model...")
    analyzer = SentimentAnalyzer(
        model_path="./models/sentiment_model",
        config_path="./config.yaml"
    )
    
    print("\n" + "="*80)
    print("Turkish Sentiment Analysis - Detailed Results")
    print("="*80)
    
    # Her metin için tahmin yap
    for i, text in enumerate(test_texts, 1):
        print(f"\nExample {i}:")
        print(f"Text: {text}")
        
        # Tahmin yap
        results = analyzer.predict(text)
        result = results[0]
        
        # Sonuçları göster
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        if 'scores' in result:
            print("\nDetailed Scores:")
            for label, score in sorted(result['scores'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {label:10}: {score:.4f}")
        
        print("-" * 80)
    
    # Batch işlem örneği
    print("\n\nBatch Processing Example:")
    print("="*80)
    
    results = analyzer.batch_predict(test_texts)
    
    print(f"\nTotal predictions: {len(results)}")
    
    sentiment_counts = {}
    total_confidence = {}
    
    for result in results:
        sentiment = result['sentiment']
        confidence = result['confidence']
        
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        total_confidence[sentiment] = total_confidence.get(sentiment, 0) + confidence
    
    print("\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        avg_conf = total_confidence[sentiment] / count
        print(f"  {sentiment:10}: {count:3} (avg confidence: {avg_conf:.4f})")
    
    # JSON formatında sonuç
    print("\n\nJSON Output Example:")
    print("="*80)
    json_output = json.dumps(results[0], ensure_ascii=False, indent=2)
    print(json_output)

if __name__ == "__main__":
    custom_inference_example()
