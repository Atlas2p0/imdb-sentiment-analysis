from torch.utils.data import Dataset
import torch

class LSTMDataset(Dataset):
    def __init__(self, lemmatized_tokens, labels, vocab, max_len):
        
        self.samples= list(zip(lemmatized_tokens, labels))
        
        self.vocab= vocab
        self.max_len= max_len

        self.pad_idx= vocab.get('<PAD>', 0)
        self.unk_idx= vocab.get('<UNK>', 1)
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        lemmatized_tokens, label= self.samples[idx]

        # 1. Numericalize the tokens and labels
        indexed_tokens= [
            self.vocab.get(token, self.unk_idx)
            for token in lemmatized_tokens
        ]
        # Keeping track of original length
        original_length= len(indexed_tokens)
        # If we're not gonna pad then original_length shouldn't be more than max_len
        if original_length > self.max_len:
            original_length= self.max_len

        numerical_labels= 0 if label == 'negative' else 1
        # 2. Pad or Truncate based on Max Length
        if len(indexed_tokens) > self.max_len:
            indexed_tokens= indexed_tokens[:self.max_len]
        else:
            padding_length= self.max_len - len(indexed_tokens)
            padding= [self.pad_idx] * padding_length
            indexed_tokens= indexed_tokens + padding
        # 3. Convert indeces & labels to PyTorch Tensor
        text_tensor= torch.tensor(indexed_tokens, dtype= torch.long)
        label_tensor= torch.tensor(numerical_labels, dtype= torch.long)
        
        return text_tensor, label_tensor, original_length


class RNNDataset(Dataset):

    def __init__(self, lemmatized_tokens, labels, vocab, max_len):
        
        self.samples= list(zip(lemmatized_tokens, labels))
        
        self.vocab= vocab
        self.max_len= max_len

        self.pad_idx= vocab.get('<PAD>', 0)
        self.unk_idx= vocab.get('<UNK>', 1)
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        lemmatized_tokens, label= self.samples[idx]

        # 1. Numericalize the tokens and labels
        indexed_tokens= [
            self.vocab.get(token, self.unk_idx)
            for token in lemmatized_tokens
        ]
        numerical_labels= 0 if label == 'negative' else 1
        # 2. Pad or Truncate based on Max Length
        if len(indexed_tokens) > self.max_len:
            indexed_tokens= indexed_tokens[:self.max_len]
        else:
            padding_length= self.max_len - len(indexed_tokens)
            padding= [self.pad_idx] * padding_length
            indexed_tokens= indexed_tokens + padding
        # 3. Convert indeces & labels to PyTorch Tensor
        text_tensor= torch.tensor(indexed_tokens, dtype= torch.long)
        label_tensor= torch.tensor(numerical_labels, dtype= torch.long)
        
        return text_tensor, label_tensor
    
class BERTSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts= texts
        self.labels= labels
        self.tokenizer= tokenizer
        self.max_len= max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text, label= self.texts[idx], self.labels[idx]

        # Apply Bert Tokenizer
        encoding= self.tokenizer(
            text,
            max_length= self.max_len,
            padding= 'max_length',
            truncation= True,
            return_tensors= 'pt'
        )

        # Get tensors from encoding dictionary
        input_ids= encoding['input_ids'].squeeze(0)
        attention_mask= encoding['attention_mask'].squeeze(0)
        token_type_ids= encoding['token_type_ids'].squeeze(0)

        label_numerical= 0 if label == 'negative' else 1
        label_tensor= torch.tensor(label_numerical, dtype= torch.long)

        return input_ids, attention_mask, token_type_ids, label_tensor