"""
Türkçe Sentiment Analysis Model Eğitim Scripti
Turkish Sentiment Analysis Model Training Script

dbmdz/bert-base-turkish-cased modelini LoRA fine-tuning ile eğit
Train dbmdz/bert-base-turkish-cased model with LoRA fine-tuning
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass
import logging
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import Dataset, load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Logging ayarı
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Eğitim konfigürasyonu"""
    model_name: str
    max_length: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    output_dir: str
    use_lora: bool
    seed: int
    device: str

class TurkishSentimentTrainer:
    """Türkçe Sentiment Analysis Trainer"""
    
    def __init__(self, config_path: str = "./config.yaml"):
        """
        Trainer'ı initialize et
        
        Args:
            config_path: Config dosyası yolu
        """
        self.config = self._load_config(config_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Device: {self.device}")
        
        set_seed(self.config["training"]["seed"])
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
    def _load_config(self, config_path: str) -> Dict:
        """Config dosyasını yükle"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_dataset(self, data_path: str) -> Tuple[Dataset, Dataset, Dataset]:
        """
        CSV veri setini yükle ve böl
        Load and split CSV dataset
        """
        logger.info(f"Dataset yükleniyor / Loading dataset: {data_path}")
        df = pd.read_csv(data_path)
        
        # Etiketleri sayıya çevir
        df['label'] = df['sentiment'].map(self.label2id)
        
        # Train/val/test split
        train_data, temp_data = train_test_split(
            df, 
            test_size=0.2, 
            random_state=self.config["training"]["seed"],
            stratify=df['label']
        )
        
        val_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,
            random_state=self.config["training"]["seed"],
            stratify=temp_data['label']
        )
        
        logger.info(f"Train set: {len(train_data)}")
        logger.info(f"Val set: {len(val_data)}")
        logger.info(f"Test set: {len(test_data)}")
        
        # Hugging Face Dataset'e çevir
        train_dataset = Dataset.from_pandas(train_data[['text', 'label']])
        val_dataset = Dataset.from_pandas(val_data[['text', 'label']])
        test_dataset = Dataset.from_pandas(test_data[['text', 'label']])
        
        return train_dataset, val_dataset, test_dataset
    
    def preprocess_dataset(
        self, 
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Veri setini tokenize et ve preprocess et
        """
        logger.info("Dataset preprocessing başlıyor...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["name"]
        )
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.config["model"]["max_length"]
            )
        
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing train dataset"
        )
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing val dataset"
        )
        test_dataset = test_dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing test dataset"
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def setup_model(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Model'i setup et (LoRA yapılandırması ile)
        """
        logger.info(f"Model yükleniyor / Loading model: {self.config['model']['name']}")
        
        # Base model yükle
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model"]["name"],
            num_labels=3,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        
        # LoRA konfigürasyonu
        if self.config["model"].get("use_lora", True):
            logger.info("LoRA fine-tuning kurulumu...")
            
            lora_config = LoraConfig(
                r=self.config["lora"]["r"],
                lora_alpha=self.config["lora"]["lora_alpha"],
                target_modules=self.config["lora"]["target_modules"],
                lora_dropout=self.config["lora"]["lora_dropout"],
                bias=self.config["lora"]["bias"],
                task_type=TaskType.SEQ_CLS,
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        # Tokenizer yükle
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["name"]
        )
        
        return model, tokenizer
    
    def compute_metrics(self, eval_pred):
        """Metrikleri hesapla"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    def train(self, data_path: str):
        """
        Modeli eğit
        Train the model
        """
        # Dataset yükle
        train_dataset, val_dataset, test_dataset = self.load_dataset(data_path)
        
        # Preprocessing
        train_dataset, val_dataset, test_dataset = self.preprocess_dataset(
            train_dataset, val_dataset, test_dataset
        )
        
        # Model setup
        model, tokenizer = self.setup_model()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config["model"]["output_dir"],
            num_train_epochs=self.config["training"]["num_epochs"],
            per_device_train_batch_size=self.config["training"]["batch_size"],
            per_device_eval_batch_size=self.config["training"]["batch_size"],
            learning_rate=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
            warmup_steps=self.config["training"]["warmup_steps"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            logging_steps=self.config["training"]["logging_steps"],
            eval_strategy="steps",
            eval_steps=self.config["training"]["eval_steps"],
            save_steps=self.config["training"]["save_steps"],
            save_total_limit=self.config["model"]["save_total_limit"],
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=self.config["training"]["seed"],
            fp16=torch.cuda.is_available(),
            logging_dir='./logs',
            report_to=["tensorboard"],
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Eğit
        logger.info("Eğitim başlıyor / Training started...")
        trainer.train()
        
        # Test set'te değerlendir
        logger.info("Test set'te değerlendirme yapılıyor...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        
        logger.info("\n" + "="*50)
        logger.info("Test Set Results:")
        logger.info("="*50)
        for key, value in test_results.items():
            logger.info(f"{key}: {value:.4f}")
        
        # Model'i kaydet
        logger.info(f"Model kaydediliyor / Saving model to {self.config['model']['output_dir']}")
        model.save_pretrained(self.config["model"]["output_dir"])
        tokenizer.save_pretrained(self.config["model"]["output_dir"])
        
        # Metrikleri kaydet
        self._save_metrics(test_results)
        
        return trainer, test_results
    
    def _save_metrics(self, results: Dict):
        """Metrikleri dosyaya kaydet"""
        output_dir = Path(self.config["model"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = output_dir / "test_metrics.txt"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write("Turkish Sentiment Analysis - Test Results\n")
            f.write("="*50 + "\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Model: {self.config['model']['name']}\n")
            f.write("="*50 + "\n\n")
            
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")
        
        logger.info(f"Metrikleri kaydedildi / Metrics saved: {metrics_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Turkish Sentiment Analysis Model Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Config dosyası yolu"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data/turkish_sentiment_dataset.csv",
        help="Dataset dosyası yolu"
    )
    
    args = parser.parse_args()
    
    # Dataset oluştur (eğer yoksa)
    if not Path(args.data).exists():
        logger.info("Dataset bulunamadı, oluşturuluyor...")
        from create_dataset import create_dataset
        create_dataset(output_path=args.data)
    
    # Trainer oluştur ve eğit
    trainer = TurkishSentimentTrainer(config_path=args.config)
    trainer.train(data_path=args.data)

if __name__ == "__main__":
    main()
