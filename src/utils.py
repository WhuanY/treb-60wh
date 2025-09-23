import torch
from torch import nn
import time
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

def run_experiment(
    train_dataloader, 
    val_dataloader, 
    model, 
    device,
    epochs=4, 
    lr=2e-5
):
    """
    Run a training experiment and return results
    """
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Total number of training steps
    total_steps = len(train_dataloader) * epochs
    
    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Initialize variables to track metrics
    best_accuracy = 0
    start_time = time.time()
    
    # Move model to device
    model.to(device)
    
    # For each epoch
    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch+1} / {epochs} ========")
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_dataloader:
            # Clear any previously calculated gradients
            optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Compute loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            total_train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip the norm of the gradients to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters and learning rate
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_train_loss:.4f}")
        
        # Evaluation phase
        model.eval()
        
        # Track variables for evaluation
        total_eval_accuracy = 0
        total_eval_loss = 0
        all_preds = []
        all_labels = []
        
        # Evaluate data for one epoch
        for batch in val_dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            # No gradient calculation needed for evaluation
            with torch.no_grad():
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                # Compute loss
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)
                total_eval_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                
                # Calculate accuracy
                total_eval_accuracy += (preds == labels).sum().item()
                
                # Collect all predictions and labels for F1 score
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_val_loss = total_eval_loss / len(val_dataloader)
        accuracy = total_eval_accuracy / len(val_dataloader.dataset)
        f1 = f1_score(all_labels, all_preds)
        
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Update best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    
    # Calculate total time
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    # Return experiment results
    return {
        'accuracy': best_accuracy,
        'f1_score': f1,
        'training_time': f"{minutes} mins {seconds} secs",
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


def create_dataloaders(train_dataset, test_dataset, batch_size=16):
    """Create DataLoaders for training and testing"""
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )
    
    return train_dataloader, test_dataloader
