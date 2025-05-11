import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
import re
from collections import Counter
import torch
import src.config as cfg


def clean_text(text):
    """Cleans raw text

    Applies regex to rid text of html tags, punctuation and special characters

    Args:
        text (str): Raw text of a movie review

    Returns:
        str: Cleaned raw text without punctuation, html tags or special characters 

    """
    # Define patterns to be removed
    html_pattern= r"<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>"
    punctuation_pattern= r'[^\w\s]'
    special_chars_pattern= r'[@_!#$%^&*()<>?/\|}{~:]'

    # Remove any patters matching the regex
    text= re.sub(html_pattern, '', text)
    text= re.sub(punctuation_pattern, '', text)
    text= re.sub(special_chars_pattern, ' ', text)

    return text

def filter_words(text):
    """Filters words that are not 'stopwords' (such as 'this', 'that', 'the', 'are', 'is', ...etc)
    and returns the filtered set of words

    Args:
        text (str): A string of text that has been assumingly already cleaned
    
    Returns:
        str: A string of words that will be kept for the tokenization process
    
    """
    # nltk.download('stopwords') # -> Use this if nltk('stopwords') isn't already present 
    
    stop_words= set(stopwords.words('english'))
    words= text.split()
    filtered_words= [word for word in words if word not in stop_words]

    return ' '.join(filtered_words)

def remove_stopwords(text):
    """ Helper function to call filter_words() and return the filtered string

    Args:
        text(str): String to be filtered from stopwords
    Returns:
        str: String after removal of stopwords 

    """
    text= filter_words(text)
    return text

def tokenize_text(text):
    """Tokenizes raw text into a list of tokens

    This function takes in text that is assumed to be filtered and cleaned and then transforms it into a list of tokens (words)
    using nltk.word_tokenize() for word tokenization

    Args:
        text(str): string of text to be tokenized
    Returns:
        A list of strings, where each element is a word/token (str)
    Raises:
        nltk.downloader.DownloadError: If the 'punkt' tokenizer is not downloaded

    """
    text= word_tokenize(text)
    return text

def get_wordnet_pos(tag):
    """Converts a Penn Treebank POS tag to a WordNet POS tag.
    This function takes the first character of a POS tag string (like those
    produced by NLTK's pos_tag) and maps it to the corresponding Part-of-Speech
    constant used by NLTK's WordNet lemmatizer.

    Maps:
        'J' (Adjective) -> wordnet.ADJ ('a')
        'N' (Noun)      -> wordnet.NOUN ('n')
        'V' (Verb)      -> wordnet.VERB ('v')
        'R' (Adverb)    -> wordnet.ADV ('r')

    If the first character does not match one of these, it defaults to
    wordnet.NOUN ('n').

    Args:
        tag(str): The input POS tag string (e.g., "JJ", "NN", "VB", "RB", "WPS").

    Returns:
        The corresponding string representation of the WordNet POS tag
        ('a', 'n', 'v', or 'r').
    """
    tag= tag[0].upper()
    tag_dict= {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV }
    # return value for the key if found otherwise, the default (wordnet.NOUN)
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    """Applies WordNet lemmatization to a list of word tokens.

    This function first performs Part-of-Speech (POS) tagging on the input
    list of tokens, then uses the POS tags to determine the correct WordNet
    POS for each word. Finally, it applies the WordNet lemmatizer to reduce
    each word to its base or dictionary form using the derived POS tag.

    Args:
        text(str): A list of string tokens (words) from a single review.
              Assumes the text has already been tokenized.

    Returns:
        A list of strings, where each string is the lemmatized form
        of the corresponding input token.
    """
    lemmatizer= WordNetLemmatizer()
    tagged_words= nltk.pos_tag(text)

    return [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_words]

def preprocess_text_for_rnn(raw_text, vocab, max_len):
    """Converts raw text into input recognized by the pretrained RNN
    This function performs text cleaning, tokenization, lemmatization padding/truncation 
    and maps text into vocab indeces recognized by the pretrained RNN

    Args: 
        raw_text (str): Raw text of a movie review
        vocab: The vocab used to train the RNN from /artifacts/rnn_vocab.pkl
        max_len(int): Max length for the text_tensor decided when training the RNN (saved at config.py) can be found in /Notebooks/RNN.ipynb
    Returns:
        torch.tensor: A tensor of indeces of tokens with padding/truncation applied
    """
    # Text Cleaning, Tokenization and Lemmatization
    processed_text= clean_text(raw_text)
    processed_text= remove_stopwords(processed_text)
    processed_text= tokenize_text(processed_text)
    processed_text= lemmatize_text(processed_text)

    # Get token indeces from vocab
    numericalized_text= [
                        vocab.get(token, cfg.UNK_IDX)
                        for token in processed_text
                    ]
    # Padding/Truncation
    if len(numericalized_text) > max_len:
        numericalized_text= numericalized_text[:max_len]
    else:
        padding_length= max_len - len(numericalized_text)
        padding= [cfg.PAD_IDX] * padding_length
        numericalized_text= numericalized_text + padding
        text_tensor= torch.tensor(numericalized_text, dtype= torch.long)

    return text_tensor

def preprocess_text_for_lstm(raw_text, vocab, max_len):
    """Converts raw text into input recognized by the pretrained LSTM
    This function performs text cleaning, tokenization, lemmatization padding/truncation
    and maps text into vocab indeces recognized by the pretrained LSTM

    Args:
        raw_text(str): Raw text of a movie review
        vocab: The vocab used to train the LSTM from /artifacts/lstm_vocab.pkl
        max_len(int): The max length for the text_tensor decided when training the LSTM (saved at config.py) can be found in /Notebooks/LSTM.ipynb
    Returns:
        text_tensor(torch.tensor): A tensor of indeces of tokens with padding/truncation applied
        original_length(torch.tensor): Length of the tensor expected by the LSTM in its forward() function model definition can be found in src/models.py 
    """
    # Cleaning, Tokenization, Lemmatization of text
    processed_text= clean_text(raw_text)
    processed_text= remove_stopwords(processed_text)
    processed_text= tokenize_text(processed_text)
    processed_text= lemmatize_text(processed_text)
    # Transforming tokens into indeces
    numericalized_text= [
                        vocab.get(token, cfg.UNK_IDX)
                        for token in processed_text
                    ]
    # Getting the length of the index list, and truncating if orig_length > max_len
    original_length= len(numericalized_text)
    if original_length > max_len:
        original_length= max_len

    # Padding/Truncation of index list
    if len(numericalized_text) > max_len:
        numericalized_text= numericalized_text[:max_len]
    else:
        padding_length= max_len - len(numericalized_text)
        padding= [cfg.PAD_IDX] * padding_length
        numericalized_text= numericalized_text + padding

    # Transforming numericalized text and original_length into torch tensors
    # Adding 1 dimension as the pretrained LSTM expects an extra dim for batch
    text_tensor= torch.tensor(numericalized_text, dtype= torch.long)
    text_tensor= text_tensor.unsqueeze(0)
    original_length= torch.tensor([original_length], dtype= torch.long)
    
    return text_tensor, original_length

def preprocess_text_for_bert(raw_text, tokenizer, max_len):
    """Converts raw text into input expected by the pretrained BERT model
    This function uses a pretrained BERT tokenizer that matches the pretrained BERT model to produce the input required

    Args:
        raw_text(str): Raw movie review text
        tokenizer: The loaded Hugging Face tokenizer object compatible with
                   the pre-trained BERT model (e.g., loaded from
                   /artifacts/bert_tokenizer).
        max_len(int): The maximum length for sequence padding and truncation,
                      decided in Notebooks/BERT.ipynb and saved in src/config.py  
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three
        PyTorch tensors, each with shape `[max_len]` and dtype `torch.long`
        (after squeezing the batch dimension).
        The tensors are:
            - input_ids: The numericalized sequence of tokens, including
                         special tokens and padding.
            - attention_mask: A mask indicating which tokens are actual content (1)
                              and which are padding (0).
            - token_type_ids: A mask indicating the segment/sentence a token belongs to
                              (all zeros for single-sequence classification).
    """
    cleaned_text= clean_text(raw_text)
    # Apply bert tokenizer to get encoding of the raw review text
    encoding= tokenizer(cleaned_text, max_length= max_len,
                        padding= 'max_length', truncation= True,
                        return_tensors= 'pt')
    # Squeeze batch dimension for single review prediction
    input_ids= encoding['input_ids'].squeeze(0)
    attention_mask= encoding['attention_mask'].squeeze(0)
    token_type_ids= encoding['token_type_ids'].squeeze(0)

    return input_ids, attention_mask, token_type_ids

def build_vocab(df: pd.DataFrame, tokens_col: str, min_freq: int):
    """Builds a vocabulary (token-to-index and index-to-token mappings).

    This function flattens a DataFrame column containing lists of tokens,
    counts token frequencies, and creates two dictionaries mapping tokens
    to unique integer indices and vice-versa. It includes special tokens
    for padding and unknown tokens and filters regular tokens based on a
    minimum frequency threshold.

    Args:
        df: A Pandas DataFrame containing the tokenized text.
        tokens_col: The name of the column in the DataFrame that contains
                    lists of token strings (e.g., 'lemmatized_tokens').
                    Each element in this column is expected to be a List[str].
        min_freq: The minimum number of times a token must appear in the
                  `tokens_col` to be included in the vocabulary (excluding
                  special tokens).

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: A tuple containing two dictionaries:
            - token_to_idx (Dict[str, int]): Maps token strings to their
                                            corresponding integer indices. Includes
                                            special tokens (PAD, UNK).
            - idx_to_token (Dict[int, str]): Maps integer indices back to
                                            their corresponding token strings.
    """
    tokenized_data= df[tokens_col]
    special_tokens= [cfg.PAD_TOKEN, cfg.UNK_TOKEN]
    
    # Flatten the list of lists into a list of all tokens in our data
    all_tokens= [token for review_tokens in tokenized_data for token in review_tokens]
    # Calculate Token Frequency
    token_counts= Counter(all_tokens)
    # Build initial vocab with pad and unk token as the first two tokens
    token_to_idx= {token: i for i, token in enumerate(special_tokens)}
    idx_to_token= {i: token for i, token in enumerate(special_tokens)}
    # Minimum frequency to filter tokens in our vocab
    min_frequency= min_freq

    # Get tokens sorted by freq and filter based on min freq
    valid_tokens= [token for token, count in token_counts.most_common() if count >= min_frequency and token not in special_tokens]

    # Build vocab using valid_tokens
    for i, token in enumerate(valid_tokens):
        token_to_idx[token]= i + cfg.START_INDEX
        idx_to_token[i + cfg.START_INDEX]= token
    # vocab_size= len(vocab)

    return token_to_idx, idx_to_token

