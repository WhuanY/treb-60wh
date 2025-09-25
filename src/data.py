import pandas as pd
import torch
from torch.utils.data import Dataset

class DataSpliter:
    def __init__(self, data, train_size, random_state=42):
        assert train_size <= len(data), "Train size must be less than or equal to the total data size"
        self.data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)  # Shuffle the data
        self.train_size = train_size

    def split(self):
        train_data = self.data[:self.train_size]
        test_data = self.data[self.train_size:]
        return train_data, test_data

    def get_train_data(self):
        return self.split()[0]
          
    def get_test_data(self):
        return self.split()[1]


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = 1 if self.labels[idx] == 'positive' else 0
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_imdb_data(file_path="IMDB Dataset.csv"):
    """Load IMDB dataset from CSV file"""
    data = pd.read_csv(file_path)
    return data
