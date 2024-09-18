import os
import argparse
import joblib
import torch
import torch.nn as nn
from transformers import BertTokenizer
from model import get_model
from tqdm import tqdm
import numpy as np
import logging
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def parse_args():
    parser = argparse.ArgumentParser(description="Predict Theme for a given input sentence")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model checkpoint (e.g., models/checkpoint_epoch_5.pth or models/final_model.pth)')
    parser.add_argument('--data_path', type=str, default='data/20newsgroups_test_with_lda_words.pkl', help='Path to the data')
    parser.add_argument('--params_path', type=str, default='models/best_params.pkl', help='Path to the best hyperparameters file')
    parser.add_argument('--lda_model_path', type=str, default='models/lda_model.pkl', help='Path to the saved LDA model')
    parser.add_argument('--vectorizer_path', type=str, default='models/vectorizer.pkl', help='Path to the saved vectorizer')
    parser.add_argument('--glove_path', type=str, default='glove.6B.100d.txt', help='Path to GloVe embeddings (optional)')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum input sequence length')
    parser.add_argument('--top_n_topics', type=int, default=3, help='Number of top topics to consider for concatenation')
    return parser.parse_args()

def load_glove_embeddings(glove_path, tokenizer, embed_dim):
    """
    Load GloVe embeddings and create an embedding matrix.
    """
    logger.info(f"Loading GloVe embeddings from {glove_path}")
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # initializing the embedding matrix
    vocab_size = tokenizer.vocab_size
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim))

    for word, idx in tokenizer.get_vocab().items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]

    logger.info(f"Loaded {len(embeddings_index)} word vectors from GloVe")
    return torch.tensor(embedding_matrix, dtype=torch.float32)

def load_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from a checkpoint file.
    Handles both complete checkpoint dictionaries and standalone state_dicts.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded 'model_state_dict' from {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint)
        logger.info(f"Loaded state_dict directly from {checkpoint_path}")
    return model

def load_data(data_path):
    """
    Load preprocessed data from a pickle file.
    """
    logger.info(f"Loading data from {data_path}")
    return joblib.load(data_path)

def predict_theme(args, input_text, model, tokenizer, lda_model, vectorizer, device, target_names):
    """
    Predict the theme of the input_text using the trained model.
    """
    # preprocessing the input_text
    # vectorize the input
    input_vectorized = vectorizer.transform([input_text])

    # get the topic distribution
    topic_distribution = lda_model.transform(input_vectorized)  # [1, n_topics]

    # get the top N topics
    top_n_topics = args.top_n_topics
    top_topic_indices = topic_distribution[0].argsort()[:-top_n_topics -1:-1]

    # get the top words for the N topics
    top_words = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx in top_topic_indices:
        
        # get top 10 words for each topic
        top_word_indices = lda_model.components_[topic_idx].argsort()[:-11:-1]
        top_words.extend([feature_names[i] for i in top_word_indices])

    top_words = list(set(top_words))  # remove any duplicates

    # concatenate the input_text with top_words separated by [SEP] tag
    lda_words = " ".join(top_words)
    combined_text = f"{input_text} [SEP] {lda_words}"

    # tokenize
    encoding = tokenizer.encode_plus(
        combined_text,
        add_special_tokens=True,
        max_length=args.max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
        clean_up_tokenization_spaces=False  # set False to avoid FutureWarning
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # pass through model
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

    predicted_label = preds.item()
    predicted_theme = target_names[predicted_label]

    return predicted_theme

def main():
    args = parse_args()

    # enable logging
    logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger(__name__)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")

    # load best hyperparameters tuned using Optuna
    if os.path.exists(args.params_path):
        best_params = joblib.load(args.params_path)
        logger.info(f"Loaded best hyperparameters from {args.params_path}")
    else:
        logger.error(f"Best hyperparameters file not found at {args.params_path}")
        return

    # compute  the hidden_dim
    hidden_dim = best_params['num_heads'] * best_params['hidden_dim_multiplier']
    logger.info(f"Computed hidden_dim: {hidden_dim} (num_heads: {best_params['num_heads']}, hidden_dim_multiplier: {best_params['hidden_dim_multiplier']})")

    # load the LDA model and the vectorizer
    if not os.path.exists(args.lda_model_path):
        logger.error(f"LDA model path {args.lda_model_path} does not exist.")
        return
    if not os.path.exists(args.vectorizer_path):
        logger.error(f"Vectorizer path {args.vectorizer_path} does not exist.")
        return
    lda_model = joblib.load(args.lda_model_path)
    vectorizer = joblib.load(args.vectorizer_path)
    logger.info(f"Loaded LDA model from {args.lda_model_path}")
    logger.info(f"Loaded vectorizer from {args.vectorizer_path}")

    # initialize the BerTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logger.info("Tokenizer initialized")

    # load the data to get target_names
    data = load_data(args.data_path)
    target_names = data['target_names']

    # initialize model with best hyperparameters which were tuned
    model = get_model(
        vocab_size=tokenizer.vocab_size,
        embed_dim=best_params['embed_dim'] if 'embed_dim' in best_params else 100,
        hidden_dim=hidden_dim,
        output_dim=len(target_names),
        n_layers=best_params['n_layers'],
        bidirectional=best_params['bidirectional'],
        dropout=best_params['dropout'],
        pretrained_embeddings=None,  # check
        max_len=args.max_len,
        num_heads=best_params['num_heads']
    )

    # load the model checkpoint
    if not os.path.exists(args.model_path):
        logger.error(f"Model checkpoint path {args.model_path} does not exist.")
        return
    model = load_checkpoint(model, args.model_path, device)
    model = model.to(device)
    model.eval()

    # using paralell processing if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for prediction")
        model = nn.DataParallel(model)

    # loading the glove embeddings if provided
    if args.glove_path and os.path.exists(args.glove_path):
        pretrained_embeddings = load_glove_embeddings(args.glove_path, tokenizer, best_params['embed_dim'] if 'embed_dim' in best_params else 100)
        if isinstance(model, nn.DataParallel):
            model.module.embedding.weight.data.copy_(pretrained_embeddings)
        else:
            model.embedding.weight.data.copy_(pretrained_embeddings)
        model.embedding.weight.requires_grad = False
        logger.info("Loaded and set pre-trained GloVe embeddings")
    else:
        logger.warning(f"GloVe embeddings not found at {args.glove_path}. Using random embeddings.")

    # interactive prediction by taking user input
    print("Enter a sentence or words to predict its theme. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("Input: ")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        if user_input.lower() == 'exit':
            break
        if not user_input.strip():
            print("Please enter a valid input.")
            continue

        predicted_theme = predict_theme(args, user_input, model, tokenizer, lda_model, vectorizer, device, target_names)
        print(f"Predicted Theme: {predicted_theme}\n")

if __name__ == '__main__':
    main()
