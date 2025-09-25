from argparse import ArgumentParser
import torch
import pandas as pd
from transformers import BertModel, BertTokenizer

from data import load_imdb_data, DataSpliter, IMDBDataset
from model import BERTForSentiment
from utils import init_seed, run_experiment, create_dataloaders

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

def parse_args(parser):
    args = parser.parse_args()
    if args.rep_layer != "last":
        if "," in args.rep_layer:
            args.rep_layer = [int(x) for x in args.rep_layer.split(",")]
        elif args.rep_layer.isdigit():
            args.rep_layer = int(args.rep_layer)
        else:
            raise ValueError("rep_layer must be 'last' or an integer representing the layer index")
    print(args)
    return args

def experiment_entry(data, tokenizer, bert_model, device, config):
    """
    Experiment with configurable representation layer, training strategy, and data size.
    """
    # Split data into train and test
    data_splitter = DataSpliter(data, train_size=config.data_size, random_state=42)
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
    
    # Create model based on experiment configuration
    model = BERTForSentiment(
        bert_model=bert_model,
        representation_layer=config.rep_layer,
        training_strategy=config.training_strategy
    )
    
    
    # Run experiment
    results = run_experiment(
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        model=model,
        device=device,
        epochs=10,  # updated to 10 epochs to match experiment settings
        lr=2e-5
    )
    
    print("\nExperiment Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Training Time: {results['training_time']}")
    print(f"Trainable Parameters: {results['trainable_params']:,}")
    
    return results


def main(config):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    init_seed(42, reproducibility=True)
    
    # Load BERT model and tokenizer
    model_path = "./model/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertModel.from_pretrained(model_path)
    
    # Load IMDB data
    data = load_imdb_data("./data/IMDB Dataset.csv")
    print(f"Loaded data shape: {data.shape}")
    
    # Run experiment
    print("Running experiment...")
    results = experiment_entry(data, tokenizer, bert_model, device, config)
    
    # Print table row format
    print("\nTable Row:")
    print(f"| {config.exp_id} | {config.rep_layer} | {config.training_strategy} | {config.data_size} | {results['accuracy']*100:.2f} | {results['f1_score']:.2f} | {results['training_time']} | {results['trainable_params']/1000000:.1f} M |")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--exp_id', type=int, required=True, help="Experiment ID")
    parser.add_argument('--rep_layer', type=str, required=True, help="Representation layer (e.g., 'last', '8', '4')")
    parser.add_argument('--training_strategy', type=str, required=True, help="Training strategy")
    parser.add_argument('--data_size', type=int, required=True, help="Number of training samples (e.g., 40000, 5000)")
    
    # config = parser.parse_args()
    config = parse_args(parser)
    main(config)
