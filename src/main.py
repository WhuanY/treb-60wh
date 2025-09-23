import torch
import pandas as pd
from transformers import BertModel, BertTokenizer

from data import load_imdb_data, DataSpliter, IMDBDataset
from model import BERTForSentiment
from utils import run_experiment, create_dataloaders


def baseline_experiment(data, tokenizer, bert_model, device):
    """
    Baseline experiment: Last layer CLS, Full fine-tuning, 40k samples
    """
    # Split data into train and test
    data_splitter = DataSpliter(data)
    train_data = data_splitter.get_train_data()
    test_data = data_splitter.get_test_data()
    
    print(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples")
    
    # Create datasets
    train_dataset = IMDBDataset(
        texts=train_data['review'].values,
        labels=train_data['sentiment'].values,
        tokenizer=tokenizer
    )
    
    test_dataset = IMDBDataset(
        texts=test_data['review'].values,
        labels=test_data['sentiment'].values,
        tokenizer=tokenizer
    )
    
    # Create dataloaders
    train_dataloader, test_dataloader = create_dataloaders(train_dataset, test_dataset)
    
    # Create model
    model = BERTForSentiment(
        bert_model=bert_model,
        representation_layer="last",
        freeze_bert=False  # Full fine-tuning
    )
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params:,}")
    
    # Run experiment
    results = run_experiment(
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        model=model,
        device=device,
        epochs=4,
        lr=2e-5
    )
    
    print("\nExperiment Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Training Time: {results['training_time']}")
    print(f"Trainable Parameters: {results['trainable_params']:,}")
    
    return results


def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load BERT model and tokenizer
    model_path = "./model/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertModel.from_pretrained(model_path)
    
    # Load IMDB data
    data = load_imdb_data("./data/IMDB Dataset.csv")
    print(f"Loaded data shape: {data.shape}")
    
    # Run baseline experiment
    print("Running baseline experiment...")
    baseline_results = baseline_experiment(data, tokenizer, bert_model, device)
    
    # Print table row format
    print("\nTable Row 1:")
    print(f"| 1 | Last layer CLS | Full fine-tuning | 40k | {baseline_results['accuracy']*100:.2f} | {baseline_results['f1_score']:.2f} | {baseline_results['training_time']} | {baseline_results['trainable_params']/1000000:.1f} M |")


if __name__ == "__main__":
    main()
