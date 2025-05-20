# Deep Learning Movie Review Sentiment Analysis

A project implementing and comparing different deep learning models (RNN, LSTM, BERT) for binary sentiment classification on the IMDB movie reviews dataset. Includes data preprocessing pipelines for each model type and a command-line interface for making predictions on new text.

## Usage

https://github.com/user-attachments/assets/efb1afcf-9b95-43c9-8f89-611ec81f6bdf


## Features

*   **Data Preprocessing:** Custom pipelines for cleaning, tokenization (word-based and subword), lemmatization (for RNN/LSTM), numericalization, padding, and generating attention/token type masks (for BERT).
*   **Model Implementations:**
    *   Vanilla RNN trained from scratch.
    *   LSTM trained from scratch.
    *   BERT fine-tuned for sequence classification using the `transformers` library.
*   **Transfer Learning:** Utilizes pre-trained weights for the BERT model.
*   **Prediction Interface:** A command-line interface (`main.py`) to select a trained model and predict the sentiment of user-provided movie reviews.
*   **Code Structure:** Organized into modules (`src/`) for models, preprocessing, datasets, and utilities.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for prediction.

### Prerequisites

*   Python 3.7+
*   Libraries (see Installation)
*   NLTK/SpaCy resources (see Data)
*   CUDA enabled GPU (Recommended for training and faster prediction, though CPU is possible)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Atlas2p0/imdb-sentiment-analysis
    cd imdb-sentiment-analysis
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```
    

3.  **Download NLTK/SpaCy resources:**
    Some preprocessing steps (like tokenization, lemmatization, stop words) might require downloading resources for RNN/LSTM preprocessing. Run these in a Python environment with `nltk` installed:

```python
>>> import nltk
>>> nltk.download('punkt') # For word_tokenize
>>> nltk.download('wordnet') # For WordNetLemmatizer
>>> nltk.download('averaged_perceptron_tagger') # For pos_tag
```
4. **Download Trained Models**:
	Download the trained models from <a href= "https://drive.google.com/file/d/1n9n-8XdGHU_JQWlZzA-h71Rv-RXww7ta/view?usp=drive_link">here</a> and unzip into the `artifacts/` directory at the root of the project
## 📊 Data

The project uses the IMDB movie reviews dataset.

*   **Source:** Link: <a href= "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews">Kaggle IMDB 50k Movie Reviews</a> (e.g., Kaggle: 
*   **Description:** Contains 50k movie reviews labeled as either 'positive' or 'negative'.
*   **Location:** Place the dataset file inside a `data/` directory at the root of the project:

    ```
    sentiment_analysis/
    ├── data/
    │   └── IMDB Dataset.csv
    ├── src/
    └── ...
    ```

*   **Preprocessing:** The dataset needs to be preprocessed before training. This involves cleaning (removing HTML, etc.), tokenization, numericalization, padding/truncation, and for BERT, using the specific BERT tokenizer. The code for this is in the `src/preprocessing.py` and `src/datasets.py` modules and is performed within the training notebooks.

## Models

The project includes implementations of three deep learning models for sentiment analysis:

*   **RNN (Vanilla):** A basic Recurrent Neural Network trained from scratch. Demonstrates fundamental sequence processing. *(Note: Vanilla RNNs may struggle with longer sequences due to vanishing gradients).*
*   **LSTM:** A Long Short-Term Memory network trained from scratch. Improves upon the RNN by using gates to better handle long-range dependencies. Generally achieves good performance on this dataset.
*   **BERT:** A Bidirectional Encoder Representations from Transformers model, fine-tuned on the IMDB dataset. Utilizes transfer learning from a large pre-trained model for state-of-the-art performance. Uses subword tokenization.

Model definitions can be found in `src/models.py`. Hyperparameters and artifact paths are configured in `src/config.py`.

### Results
The model training results can be found in the notebooks at the `Notebooks/` directory at the root of the project
