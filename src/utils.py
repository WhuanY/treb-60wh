import random
import numpy as np
import torch
from torch import nn
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
import copy

def init_seed(seed, reproducibility):
    """ init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def run_experiment(
    train_dataloader, 
    val_dataloader, 
    model, 
    device,
    epochs=10,   # updated default epochs to 10
    lr=2e-5
):
    """
    Run a training experiment and return results
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    best_accuracy = 0
    best_state = None
    early_stop_counter = 0
    patience = 2
    start_time = time.time()
    model.to(device)
    
    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch+1} / {epochs} ========")
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Training at epoch {epoch+1}", unit="batch"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_train_loss:.4f}")
        
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        all_preds = []
        all_labels = []
        
        print("\nRunning Validation...")
        for batch in tqdm(val_dataloader, desc="Evaluating at epoch end", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)
                total_eval_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                total_eval_accuracy += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_eval_loss / len(val_dataloader)
        accuracy = total_eval_accuracy / len(val_dataloader.dataset)
        f1 = f1_score(all_labels, all_preds)
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"  No improvement, early stop counter: {early_stop_counter}/{patience}")
        
        if early_stop_counter >= patience:
            print("Early stopping triggered. Restoring best model checkpoint.")
            model.load_state_dict(best_state)
            break
    
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    training_time = f"{minutes}.{seconds/60} mins"
    
    return {
        'accuracy': best_accuracy,
        'f1_score': f1,
        'training_time': training_time,
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


def create_dataloaders(train_dataset, test_dataset, batch_size=48):
    """Create DataLoaders for training and testing"""
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )
    
    return train_dataloader, test_dataloader
