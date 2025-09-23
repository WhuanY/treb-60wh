import torch
from torch import nn

class BERTForSentiment(nn.Module):
    def __init__(self, bert_model, num_labels=2, representation_layer="last", freeze_bert=False):
        """
        Initialize BERTForSentiment model
        
        Parameters:
            bert_model: pre-trained BERT model
            num_labels: number of labels for classification task
            representation_layer: which layer to use for classification ("last" or layer index)
            freeze_bert: whether to freeze BERT parameters or not
        """
        super(BERTForSentiment, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.representation_layer = representation_layer
        
        # Freeze BERT parameters if required
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
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
            # Use the last layer's [CLS] token representation
            pooled_output = outputs.pooler_output
        else:
            # Use a specified intermediate layer's [CLS] token representation
            hidden_states = outputs.hidden_states
            layer_output = hidden_states[self.representation_layer]
            cls_token = layer_output[:, 0, :]  # Take [CLS] token from the specified layer
            pooled_output = self.bert.pooler.dense(cls_token)
            pooled_output = self.bert.pooler.activation(pooled_output)
        
        # Apply dropout and classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
