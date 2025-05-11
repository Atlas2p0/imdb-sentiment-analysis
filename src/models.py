from transformers import BertModel
from torch import nn
import torch
class CustomBertClassifier(nn.Module):
    def __init__(self, model_name, num_classes= 2, dropout_prob= 0.1):
        super().__init__()

        self.bert= BertModel.from_pretrained(model_name)

        # for param in self.bert.parameters():
        #   param.requires_grad= False

        bert_hidden_size= self.bert.config.hidden_size

        self.classifier= nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(bert_hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs= self.bert(
            input_ids= input_ids,
            attention_mask= attention_mask,
            token_type_ids= token_type_ids
        )

        pooled_output= outputs.pooler_output

        logits= self.classifier(pooled_output)

        return logits
    
class SimpleLSTM(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding= nn.Embedding(num_embeddings= vocab_size,embedding_dim= embedding_dim, padding_idx= 0)

        self.lstm= nn.LSTM(embedding_dim, hidden_size, batch_first= True)
        self.dropout= nn.Dropout(p= 0.5)
        self.fc= nn.Linear(hidden_size, 2)
    
    def forward(self, text_tensor: torch.Tensor, lengths: torch.Tensor):

        embedded= self.embedding(text_tensor)
    
        packed_embedded= nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(),
            batch_first= True, enforce_sorted= True
        )
        
        packed_output, (hidden_state, cell_state)= self.lstm(packed_embedded)


        final_hidden_state= hidden_state.squeeze(0)
        
        final_hidden_state= self.dropout(hidden_state)

        prediction= self.fc(final_hidden_state)

        return prediction
    
class SimpleRNN(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding= nn.Embedding(num_embeddings= vocab_size,embedding_dim= embedding_dim, padding_idx= 0)

        self.rnn= nn.RNN(embedding_dim, hidden_size, batch_first= True)
        self.dropout= nn.Dropout(p= 0.5)
        self.fc= nn.Linear(hidden_size, 2)
    
    def forward(self, text_tensor: torch.Tensor):
        embedded= self.embedding(text_tensor)

        output, hidden_state= self.rnn(embedded)


        hidden_state= hidden_state.squeeze(0)
        
        final_hidden_state= self.dropout(hidden_state)

        prediction= self.fc(final_hidden_state)

        return prediction