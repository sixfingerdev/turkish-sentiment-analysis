"""
Türkçe Sentiment Analysis Inference Scripti
Turkish Sentiment Analysis Inference Script

Eğitilmiş model ile tahmin yapma
Make predictions with trained model
"""

import torch
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Türkçe Sentiment Analiz Modeli"""
    
    def __init__(
        self,
        model_path: str = "./models/sentiment_model",
        config_path: str = "./config.yaml"
    ):
        """
        Sentiment Analyzer'ı initialize et
        
        Args:
            model_path: Model dosyası yolu
            config_path: Config dosyası yolu
        """
        self.config = self._load_config(config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Model yükleniyor / Loading model: {model_path}")
        
        # Model ve tokenizer yükle
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        logger.info("Model başarıyla yüklendi / Model loaded successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Config dosyasını yükle"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def predict(
        self,
        texts: List[str],
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Metinler için sentiment tahmini yap
        
        Args:
            texts: Tahmin yapılacak metinler
            return_scores: Tüm sınıf skorlarını dön
        
        Returns:
            Tahmin sonuçları
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        for text in texts:
            # Tokenize et
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.config["model"]["max_length"]
            ).to(self.device)
            
            # Tahmin yap
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # En yüksek skor
            max_score = probabilities.max().item()
            predicted_label_id = logits.argmax(-1).item()
            predicted_label = self.id2label[predicted_label_id]
            
            result = {
                "text": text,
                "sentiment": predicted_label,
                "confidence": max_score,
            }
            
            if return_scores:
                scores = {}
                for label_id, label_name in self.id2label.items():
                    scores[label_name] = probabilities[0][label_id].item()
                result["scores"] = scores
            
            results.append(result)
        
        return results
    
    def batch_predict(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Batch modunda tahmin yap
        
        Args:
            texts: Tahmin yapılacak metinler
            batch_size: Batch boyutu
        
        Returns:
            Tahmin sonuçları
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = self.predict(batch_texts)
            results.extend(batch_results)
        
        return results
    
    def predict_from_file(
        self,
        file_path: str,
        text_column: str = "text"
    ) -> List[Dict]:
        """
        CSV dosyasından tahmin yap
        
        Args:
            file_path: CSV dosyası yolu
            text_column: Metin sütunu adı
        
        Returns:
            Tahmin sonuçları
        """
        import pandas as pd
        
        df = pd.read_csv(file_path)
        texts = df[text_column].tolist()
        
        results = self.batch_predict(texts)
        
        # Sonuçları DataFrame'e ekle
        df_results = pd.DataFrame(results)
        
        return df_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Turkish Sentiment Analysis Inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/sentiment_model",
        help="Model dosyası yolu"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Tahmin yapılacak metin"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="CSV dosyası yolu"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Config dosyası yolu"
    )
    
    args = parser.parse_args()
    
    # Analyzer oluştur
    analyzer = SentimentAnalyzer(
        model_path=args.model,
        config_path=args.config
    )
    
    if args.text:
        # Tek metin tahmin
        logger.info(f"Text: {args.text}")
        results = analyzer.predict(args.text)
        
        for result in results:
            print("\n" + "="*60)
            print(f"Text: {result['text']}")
            print(f"Sentiment: {result['sentiment'].upper()}")
            print(f"Confidence: {result['confidence']:.4f}")
            
            if "scores" in result:
                print("\nAll scores:")
                for label, score in result["scores"].items():
                    print(f"  {label}: {score:.4f}")
            print("="*60)
    
    elif args.file:
        # Dosyadan tahmin
        logger.info(f"File: {args.file}")
        df_results = analyzer.predict_from_file(args.file)
        
        output_file = Path(args.file).stem + "_predictions.csv"
        df_results.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"Sonuçlar kaydedildi / Results saved: {output_file}")
        
        print("\n" + "="*60)
        print("Predictions Summary:")
        print("="*60)
        print(df_results.head(10))
    
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("Turkish Sentiment Analysis - Interactive Mode")
        print("Çıkmak için 'quit' yazın / Type 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            text = input("Metin girin / Enter text: ").strip()
            
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            results = analyzer.predict(text)
            
            for result in results:
                print("\n" + "-"*60)
                print(f"Sentiment: {result['sentiment'].upper()}")
                print(f"Confidence: {result['confidence']:.4f}")
                
                if "scores" in result:
                    print("\nAll scores:")
                    for label, score in result["scores"].items():
                        print(f"  {label}: {score:.4f}")
                print("-"*60 + "\n")

if __name__ == "__main__":
    main()
