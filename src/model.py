import torch
from torch import nn

class BERTForSentiment(nn.Module):
    def __init__(self, bert_model, num_labels=2, representation_layer="last", training_strategy="full_fine_tuning"):
        """
        Initialize BERTForSentiment model
        
        Parameters:
            bert_model: pre-trained BERT model
            num_labels: number of labels for classification task
            representation_layer: which layer to use for classification ("last" or layer index)
            training_strategy: training strategy ("full_fine_tuning", "linear_probing", "unfreeze_last_2", "unfreeze_last_4")
        """
        super(BERTForSentiment, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.representation_layer = representation_layer
        print("="*50)
        print("Before Freezing:")
        self._print_parameters()
        # Apply freezing strategy
        if training_strategy == "linear_probing":
            print("[INFO] Freezing all BERT layers.")
            for param in self.bert.parameters():
                param.requires_grad = False
        elif training_strategy == "unfreeze_last_2":
            print("[INFO] Freezing all layers except the last 2 encoder layers.")
            for name, param in self.bert.named_parameters():
                if not any(layer in name for layer in ["encoder.layer.10", "encoder.layer.11"]):
                    param.requires_grad = False
        elif training_strategy == "unfreeze_last_4":
            print("[INFO] Freezing all layers except the last 4 encoder layers.")
            for name, param in self.bert.named_parameters():
                if not any(layer in name for layer in ["encoder.layer.8", "encoder.layer.9", "encoder.layer.10", "encoder.layer.11"]):
                    param.requires_grad = False
        # "full_fine_tuning" does not freeze any layers (default behavior)
        print("After Freezing:")
        self._print_parameters()

    def _print_parameters(self):
        for name, param in self.bert.named_parameters():
            print(f"{name}: {'Trainwable' if param.requires_grad else 'Frozen'}")
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {trainable_params:,}")
        print("="*50)

    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Get all hidden states
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True  # Get all hidden states
        )
        
        # Get the appropriate layer representation
        if self.representation_layer == "last":
            pooled_output = outputs.pooler_output
        elif isinstance(self.representation_layer, list):
            # Average pooling of specified layers' [CLS] token representations
            hidden_states = outputs.hidden_states
            cls_tokens = [hidden_states[layer][:, 0, :] for layer in self.representation_layer]
            avg_cls_token = torch.mean(torch.stack(cls_tokens), dim=0)
            pooled_output = self.bert.pooler.dense(avg_cls_token)
            pooled_output = self.bert.pooler.activation(pooled_output)
        else:
            # Use a single specified intermediate layer's [CLS] token representation
            hidden_states = outputs.hidden_states
            layer_output = hidden_states[self.representation_layer]
            cls_token = layer_output[:, 0, :]
            pooled_output = self.bert.pooler.dense(cls_token)
            pooled_output = self.bert.pooler.activation(pooled_output)
        
        # Apply dropout and classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
