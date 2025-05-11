import os
import pickle
from transformers import AutoTokenizer
import torch
import src.config as cfg
from src.models import SimpleLSTM, SimpleRNN, CustomBertClassifier

def save_vocab(vocab_dict, file_path):
    """Saves a built vocab to a specified file path
    Args:
        vocab_dict: The vocabulary dictionary (token-to-index mapping) to save.
        file_path: The full path to the .pkl file where the vocabulary will be saved.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        with open(file_path, 'wb') as f:
            pickle.dump(vocab_dict, f)
        print(f"Vocab successfully written to {file_path}")
    except Exception as e:
        print(f"Error saving vocabulary {e}")

def load_vocab(file_path):
    """Loads a previously saved vocab (pickle format) from a specified path
    Args:
        file_path: The full path to the .pkl file where the vocabulary was saved
    Returns:
        vocab_dict: loaded token-to-index mapping (vocabulary) from the file_path 
    """
    vocab_dict= None

    try:
        with open(file_path, 'rb') as f:
            vocab_dict= pickle.load(f)
        print(f"Vocabulary Successfully loaded from {file_path}")
    except FileNotFoundError:
        print(f"File not found at {file_path}")
    except Exception as e:
        print(f"Error loading vocabulary {e}")

    return vocab_dict

# Saving Tokenizer
def save_tokenizer(tokenizer, save_directory):
    """Save tokenizer to a specific directory
    This function serves to save the bert tokenizer used in training
    of this project's BERT model to a specific directory
    
    Args:
        tokenizer: the bert tokenizer object to be saved
    """
    try:
        os.makedirs(save_directory, exist_ok= True)
        tokenizer.save_pretrained(save_directory)
        print(f"Tokenizer saved sucessfully at {save_directory}")
    except Exception as e:
        print(f"Error saving tokenizer {e}")

def load_tokenizer(save_directory):
    """Loads a tokenizer from a specific directory
    This function serves to load the bert tokenizer 
    mainly, the one that was used for the training of BERT in this project
    """
    loaded_tokenizer= None
    try:
        loaded_tokenizer= AutoTokenizer.from_pretrained(save_directory)
        print(f"Tokenizer successfully loaded from {save_directory}")
    except Exception as e:
        print(f"Error loading tokenizer from {save_directory}: {e}")
    return loaded_tokenizer

def predict_rnn(index_tensor, model_file_path):
    """Make inference using pretrained RNN
    This function uses the already trained RNN
    to make a prediction on some given tensor of indeces
    
    Args:
        index_tensor: A torch tensor of indeces from a review
        model_file_path: Path to the model that will be used for inference
    Returns:
        sentiment: 'positive' or 'negative' based on the model's prediction
    """
    # Decide on the device to make sure no device errors occur
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set up RNN to load state dict of pretrained RNN
    simple_rnn= SimpleRNN(vocab_size= cfg.RNN_VOCAB_SIZE, embedding_dim= cfg.RNN_EMBEDDING_DIM, hidden_size= cfg.RNN_LSTM_HIDDEN_SIZE)
    # Move input to device
    index_tensor= index_tensor.to(device)
    # Load state_dict of pretrained RNN
    try:
        simple_rnn.load_state_dict(torch.load(model_file_path))
        print(f"RNN loaded successfully from {model_file_path}")
    except Exception as e:
        print(f"Error when loading model: {e}")
    # Move model to device
    simple_rnn.to(device)

    # Inference
    simple_rnn.eval()
    with torch.no_grad():
        prediction= simple_rnn(index_tensor)

    sentiment= 'negative' if prediction.argmax(0) == 0 else 'positive'
    return sentiment




def predict_lstm(index_tensor, length_tensor, model_file_path):
    """Make inference using pretrained LSTM
    This function uses the already trained LSTM
    to make a prediction on some given tensor of indeces
    
    Args:
        index_tensor: A torch tensor of indeces from a review
        length_tensor: The length of the sequence that is required by the pretrained LSTM
        model_file_path: Path to the model that will be used for inference
    Returns:
        sentiment: 'positive' or 'negative' based on the model's prediction
    """
    # Set up device to avoid any device errors
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    # Setup LSTM model for pretrained LSTM loading
    lstm_model= SimpleLSTM(vocab_size= cfg.LSTM_VOCAB_SIZE, embedding_dim= cfg.LSTM_EMBEDDING_DIM, hidden_size= cfg.RNN_LSTM_HIDDEN_SIZE)
    # Move index_tensor and length_tensor to target device
    index_tensor= index_tensor.to(device)
    length_tensor= length_tensor.to(device)

    # Load state dict
    try: 
        lstm_model.load_state_dict(torch.load(model_file_path))
        print(f"LSTM loaded successfully from {model_file_path}")
    except Exception as e:
        print(f"Error when loading model: {e}")

    # Move model to device
    lstm_model.to(device)
    lstm_model.eval()
    # Inference
    with torch.no_grad():
        prediction= lstm_model(index_tensor, length_tensor)

    sentiment= 'negative' if prediction.squeeze().argmax(0) == 0 else 'positive'
    return sentiment

def predict_bert(input_ids, attention_mask, token_type_ids, model_file_path):
    """Make inference using pretrained BERT
    This function uses the already trained BERT
    to make a prediction on some given tensor of indeces
    
    Args:
        input_ids(torch.tensor): The numericalized sequence of tokens, including
                         special tokens and padding.
        attention_mask(torch.tensor): A mask indicating which tokens are actual content (1)
                            and which are padding (0).
        token_type_ids(torch.tensor): A mask indicating the segment/sentence a token belongs to
                            (all zeros for single-sequence classification).
    Returns:
        sentiment: 'positive' or 'negative' based on the model's prediction
    """
    bert_model= CustomBertClassifier(cfg.BERT_MODEL_NAME)
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    input_ids= input_ids.to(device)
    attention_mask= attention_mask.to(device)
    token_type_ids= token_type_ids.to(device)

    try:
        bert_model.load_state_dict(torch.load(model_file_path))
        print(f"BERT loaded successfully from {model_file_path}")
    except Exception as e:
        print(f"Error when loading model: {e}")

    bert_model.to(device)
    bert_model.eval()

    with torch.no_grad():
        pred= bert_model(input_ids, attention_mask, token_type_ids)
    sentiment= 'negative' if pred.argmax(1).item() == 0 else 'positive'
    return sentiment
