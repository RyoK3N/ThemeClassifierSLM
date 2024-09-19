# ThemeClassifierSLM

ThemeClassifierSLM is an advanced LSTM-based neural network model designed for theme or topic classification in text data. This repository provides a comprehensive implementation of the model, incorporating several sophisticated techniques to enhance its performance in text analysis tasks. Leveraging positional encoding, residual connections, multi-head attention, and hyperparameter optimization with Optuna, ThemeClassifierSLM delivers robust and accurate theme classification capabilities.

## Table of Contents

__Features__

1. Model Overview
2. ThemeClassifierSLM Directory Structure
3. Installation
4. Data Preprocessing
5. Training the Model
6. Evaluating the Model
7. Making Predictions
8. Usage Examples
9. Hyperparameter Optimization
10. Contributing
11. License
12. Acknowledgements
13. Features

## Model Overview

__Advanced LSTM Architecture:__ Utilizes a multi-layer, bidirectional LSTM with residual connections and layer normalization.
Positional Encoding: Incorporates sine and cosine positional encoding to provide the model with information about the position of tokens in sequences.

__Multi-Head Attention:__ Implements multi-head attention mechanisms to focus on different parts of the input sequence.

__Hyperparameter Optimization:__ Employs Optuna for efficient hyperparameter tuning to maximize model performance.

__Pretrained Embeddings:__ Supports integration of GloVe embeddings for improved semantic understanding.

__Comprehensive Evaluation:__ Provides detailed evaluation metrics including accuracy, precision, recall, F1-score, and confusion matrices.

__User-Friendly Prediction Interface:__ Includes an interactive script for predicting themes of user-provided text inputs.

## ThemeClassifierSLM Directory Structure

The ThemeClassifierSLM directory should look like this after sequentially completing the steps below : 

    ThemeClassifierSLM/
        ├── pycach/
        ├── confusion_matrix_epoch5.png
        ├── data/
        │   ├── 20newsgroups_test_with_lda_words.pkl
        │   ├── 20newsgroups_with_lda_words.pkl
        │   ├── lda_model.pkl
        │   └── vectorizer.pkl
        ├── evaluate.py
        ├── model.py
        ├── models/
        │   ├── accuracy_graph.png
        │   ├── checkpoint_epoch_1.pth
        │   ├── ...
        │   ├── best_params.pkl
        │   ├── final_model.pth
        │   └── loss_graph.png
        ├── predict.py
        ├── preprocess.py
        ├── requirements.txt
        ├── testing.ipynb
        ├── train.py
        └── README.md

## Installation 

#### Prerequisites
    Python 3.7 or higher
    pip package manager
    Virtual Environment (recommended)

#### Steps
1.Clone the Repository

    git clone https://github.com/RyoK3N/ThemeClassifierSLM
    cd ThemeClassifierSLM
    
2.Create a Virtual Environment

    conda create --name slm_env python=3.11
    conda activate slm_env

3.Install Dependencies

    pip install --upgrade pip
    pip install -r requirements.txt
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip

## Data Preprocessing

The preprocessing step involves fetching the 20 Newsgroups dataset, applying Latent Dirichlet Allocation (LDA) for topic modeling, and saving the processed data for training and testing.

#### Steps

1.Run the Preprocessing Script
    
    python preprocess.py

__Arguments:__

    --data_dir: Directory to save the preprocessed data (default: ./data)
    --n_topics: Number of topics for LDA (default: 20)
    --n_top_words: Number of top words per topic (default: 10)
    --n_top_topics: Number of top topics to consider per document (default: 3)

__Example:__
    
    python preprocess.py --data_dir './data' --n_topics 20 --n_top_words 10 --n_top_topics 3



__Outputs:__

This will generate the following files in the data/ directory:

    20newsgroups_with_lda_words.pkl
    20newsgroups_test_with_lda_words.pkl
    lda_model.pkl
    vectorizer.pkl

## Training the Model

The training script handles loading the preprocessed data, initializing the model, performing hyperparameter optimization with Optuna, and training the model.

### Steps

1.Run the Training Script

    python train.py
    
__Arguments:__

    --data_path: Path to the preprocessed training data (default: data/20newsgroups_with_lda_words.pkl)
    --model_save_path: Directory to save models and checkpoints (default: models/)
    --epochs: Number of training epochs (default: 20)
    --batch_size: Training batch size (default: 32)
    --learning_rate: Initial learning rate (default: 1e-4)
    --embed_dim: Embedding dimension (default: 100)
    --n_layers: Number of LSTM layers (default: 2)
    --bidirectional: Use bidirectional LSTM (default: False)
    --dropout: Dropout rate (default: 0.5)
    --max_len: Maximum input sequence length (default: 128)
    --num_workers: Number of data loader workers (default: 4)
    --save_every: Save checkpoint every N epochs (default: 5)
    --glove_path: Path to GloVe embeddings (default: glove.6B.100d.txt)
    --patience: Early stopping patience (default: 5)
    --seed: Random seed for reproducibility (default: 42)
    --optuna_trials: Number of Optuna trials for hyperparameter optimization (default: 50)

__Example:__

    python train.py --data_path 'data/20newsgroups_with_lda_words.pkl' --model_save_path 'models/' --epochs 20 --batch_size 32 --glove_path 'glove.6B.100d.txt'

__Notes:__

__1.Hyperparameter Optimization:__ The script uses Optuna to optimize hyperparameters such as learning rate, number of heads, hidden dimensions, number of layers, dropout rate, and bidirectionality. Ensure that Optuna is installed and properly configured.

__2.Pretrained Embeddings:__ If glove_path is provided and valid, GloVe embeddings will be loaded and integrated into the model. Otherwise, embeddings will be randomly initialized.

__3.Checkpointing:__ The script saves model checkpoints every N epochs and retains the best model based on validation accuracy.

__Monitoring Training__

During training, the script will display progress bars for each epoch and trial, along with training and validation metrics. After training, it will save:

__Outputs:__

    Model checkpoints stored in :  models/ directory (e.g., checkpoint_epoch_5.pth)
    Best hyperparameters stored in : models/best_params.pkl
    Performance graphs stored in : (loss_graph.png, accuracy_graph.png)
    Final trained model stored in : models/final_model.pth


## Evaluating the Model

The evaluation script assesses the trained model's performance on the test dataset, providing comprehensive metrics and visualizations.

#### Steps

1.Run the Evaluation Script

    python evaluate.py --model_path 'models/final_model.pth'

__Arguments:__

    --model_path: Path to the saved model checkpoint (required)
    --params_path: Path to the best hyperparameters file (default: models/best_params.pkl)
    --data_path: Path to the preprocessed test data (default: data/20newsgroups_test_with_lda_words.pkl)
    --batch_size: Evaluation batch size (default: 32)
    --max_len: Maximum input sequence length (default: 128)
    --num_workers: Number of data loader workers (default: 4)
    --save_confusion_matrix: Path to save the confusion matrix plot (default: confusion_matrix.png)

__Example:__

    python evaluate.py --model_path 'models/final_model.pth' --params_path 'models/best_params.pkl' --data_path 'data/20newsgroups_test_with_lda_words.pkl'

__Outputs:__

    Metrics: Displays accuracy, precision, recall, and F1-score.
    Classification Report: Detailed report per class.
    Confusion Matrix: Saves a confusion matrix plot to the specified path (e.g., confusion_matrix.png).

## Making Predictions
The prediction script allows users to input custom text and receive theme classifications using the trained model.

#### Steps
1.Run the Prediction Script

    python predict.py --model_path 'models/final_model.pth'
    
__Arguments:__
    
    --model_path: Path to the saved model checkpoint (required)
    --data_path: Path to the data file to retrieve target names (default: data/20newsgroups_test_with_lda_words.pkl)
    --params_path: Path to the best hyperparameters file (default: models/best_params.pkl)
    --lda_model_path: Path to the saved LDA model (default: models/lda_model.pkl)
    --vectorizer_path: Path to the saved vectorizer (default: models/vectorizer.pkl)
    --glove_path: Path to GloVe embeddings (optional, default: glove.6B.100d.txt)
    --max_len: Maximum input sequence length (default: 128)
    --top_n_topics: Number of top topics to consider for concatenation (default: 3)

__Example:__

    python predict.py --model_path 'models/final_model.pth' --params_path 'models/best_params.pkl' --lda_model_path 'models/lda_model.pkl' --vectorizer_path 'models/vectorizer.pkl'

__Output__

Once the script is running, you can input sentences or words to receive theme predictions.

    Enter a sentence or words to predict its theme. Type 'exit' to quit.
    Input: The latest advancements in quantum computing are fascinating.
    Predicted Theme: sci.space

    Input: I love playing basketball with friends on weekends.
    Predicted Theme: rec.sport.basketball

    Input: exit

The script processes the input text, applies LDA topic modeling to extract relevant topics, and feeds the combined information into the model to predict the theme.
Type exit to terminate the prediction session.
