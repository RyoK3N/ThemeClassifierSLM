# evaluate.py

import os
import argparse
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from model import get_model
from tqdm.auto import tqdm
import numpy as np
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Custom Theme Classifier SLM")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model checkpoint (e.g., models/checkpoint_epoch_5.pth or models/final_model.pth)')
    parser.add_argument('--params_path', type=str, default='models/best_params.pkl', help='Path to the best hyperparameters file')
    parser.add_argument('--data_path', type=str, default='data/20newsgroups_test_with_lda_words.pkl', help='Path to the preprocessed test data')
    parser.add_argument('--batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum input sequence length')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--save_confusion_matrix', type=str, default='confusion_matrix.png', help='Path to save the confusion matrix plot')
    return parser.parse_args()

class ThemeDataset(Dataset):
    """
    Custom Dataset for Theme Classification.
    """
    def __init__(self, texts, lda_words, labels, tokenizer, max_len):
        self.texts = texts
        self.lda_words = lda_words
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        lda_word_list = self.lda_words[idx]
        label = self.labels[idx]

        # Flatten the list of lists and concatenate topic words
        lda_words_flattened = [word for sublist in lda_word_list for word in sublist]
        lda_words = " ".join(lda_words_flattened)

        combined_text = f"{text} [SEP] {lda_words}"

        encoding = self.tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path):
    """
    Load preprocessed data from a pickle file.
    """
    logger.info(f"Loading data from {data_path}")
    return joblib.load(data_path)

def load_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from a checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from {checkpoint_path}")
    return model

def evaluate(args):
    """
    Evaluates the trained model on the test dataset and reports performance metrics.
    """
    # Enable logging
    logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger(__name__)

    # Device configuration
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")

    # Load best hyperparameters
    if os.path.exists(args.params_path):
        best_params = joblib.load(args.params_path)
        logger.info(f"Loaded best hyperparameters from {args.params_path}")
    else:
        logger.error(f"Best hyperparameters file not found at {args.params_path}")
        return

    # Compute hidden_dim
    hidden_dim = best_params['num_heads'] * best_params['hidden_dim_multiplier']
    logger.info(f"Computed hidden_dim: {hidden_dim} (num_heads: {best_params['num_heads']}, hidden_dim_multiplier: {best_params['hidden_dim_multiplier']})")

    # Load data
    data = load_data(args.data_path)
    texts = data['test_texts']
    lda_words = data['test_lda_words']
    labels = data['test_labels']
    target_names = data['target_names']

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logger.info("Tokenizer initialized")

    # Create dataset and dataloader
    dataset = ThemeDataset(texts, lda_words, labels, tokenizer, args.max_len)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize model with best hyperparameters
    model = get_model(
        vocab_size=tokenizer.vocab_size,
        embed_dim=best_params['embed_dim'] if 'embed_dim' in best_params else 100,
        hidden_dim=hidden_dim,
        output_dim=len(target_names),
        n_layers=best_params['n_layers'],
        bidirectional=best_params['bidirectional'],
        dropout=best_params['dropout'],
        pretrained_embeddings=None,  # Assuming embeddings are frozen or handled during training
        max_len=args.max_len,
        num_heads=best_params['num_heads']
    )

    # Load model checkpoint
    model = load_checkpoint(model, args.model_path, device)
    model = model.to(device)
    model.eval()

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for evaluation")
        model = nn.DataParallel(model)

    all_preds = []
    all_labels = []

    # Collect predictions
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")

    # Classification report
    logger.info("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print(report)

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(args.save_confusion_matrix)
    logger.info(f"Confusion matrix saved to {args.save_confusion_matrix}")
    plt.show()

    return accuracy, precision, recall, f1

def main():
    args = parse_args()
    evaluate(args)

if __name__ == '__main__':
    main()
