import os
import argparse
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from model import get_model
import matplotlib.pyplot as plt
import logging
import numpy as np
import random
import optuna
from optuna.trial import TrialState
from sklearn.metrics import f1_score
from torch.optim import AdamW  


def parse_args():
    parser = argparse.ArgumentParser(description="Train Custom Theme Classifier SLM with Enhancements")
    parser.add_argument('--data_path', type=str, default='data/20newsgroups_with_lda_words.pkl', help='Path to the preprocessed data')
    parser.add_argument('--model_save_path', type=str, default='models/', help='Directory to save models and checkpoints')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    # removed '--hidden_dim' since it's handled by Optuna
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')  # overridden by Optuna
    parser.add_argument('--embed_dim', type=int, default=100, help='Embedding dimension')
    # removed '--n_layers', '--bidirectional', '--dropout' since it's handled by Optuna
    parser.add_argument('--n_layers', type=int, default=2, help='Number of LSTM layers')  # overridden by Optuna
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')  # overridden by Optuna
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')  # overridden by Optuna
    parser.add_argument('--max_len', type=int, default=128, help='Maximum input sequence length')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--glove_path', type=str, default='glove.6B.100d.txt', help='Path to GloVe embeddings')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')  
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--optuna_trials', type=int, default=50, help='Number of Optuna trials for hyperparameter optimization')
    return parser.parse_args()


def set_seed(seed):
    """
    Sets seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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

        # flatten the lists and concatenate topic words
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


def save_checkpoint(model, epoch, loss, optimizer, scheduler, save_path):
    """
    Save the model checkpoint.
    """
    os.makedirs(save_path, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    checkpoint_file = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_file)
    logger.info(f"Checkpoint saved at {checkpoint_file}")


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

    vocab_size = tokenizer.vocab_size
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim))

    for word, idx in tokenizer.get_vocab().items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]

    logger.info(f"Loaded {len(embeddings_index)} word vectors from GloVe")
    return torch.tensor(embedding_matrix, dtype=torch.float32)


def create_data_loaders(train_dataset, val_dataset, batch_size, num_workers):
    """
    Create DataLoaders for training and validation datasets.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers
    )

    return train_loader, val_loader


def evaluate(model, data_loader, device):
    """
    Evaluates the model on the validation set.
    Returns:
        Accuracy and F1-score.
    """
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            _, preds_batch = torch.max(outputs, 1)
            preds.extend(preds_batch.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = (np.array(preds) == np.array(true_labels)).mean()
    f1 = f1_score(true_labels, preds, average='weighted')
    return accuracy, f1


def objective(trial, args, train_loader, val_loader, device, num_classes):
    """
    Objective function for Optuna hyperparameter optimization.
    Ensures that hidden_dim is divisible by num_heads.
    """
    # hyperparameters for optimization
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    num_heads = trial.suggest_int('num_heads', 4, 12, step=2)  # Even numbers only
    hidden_dim_multiplier = trial.suggest_int('hidden_dim_multiplier', 16, 64, step=8)
    hidden_dim = num_heads * hidden_dim_multiplier  # Ensures divisibility
    n_layers = trial.suggest_int('n_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.3, 0.7)
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])

    # initializing the model
    model = get_model(
        vocab_size=train_loader.dataset.tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        n_layers=n_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        pretrained_embeddings=None,  # loaded in main
        max_len=args.max_len,
        num_heads=num_heads
    )
    model = model.to(device)

    # DataParallel for parallel processing
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    # optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-2)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # loss function
    criterion = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Trial {trial.number} - Training Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)

        # validation
        val_accuracy, val_f1 = evaluate(model, val_loader, device)
        trial.report(val_f1, epoch)

        # handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_f1


def train(args):
    """
    Main training function incorporating hyperparameter optimization and enhanced training techniques.
    """
    logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger(__name__)

    # setting seed
    set_seed(args.seed)

    # device configuration
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")

    # loading the data
    data = load_data(args.data_path)
    texts = data['train_texts']
    lda_words = data['train_lda_words']
    labels = data['train_labels']
    target_names = data['target_names']
    num_classes = len(target_names)

    # initializing Bertokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logger.info("Tokenizer initialized")

    # training and validation sets
    train_texts, val_texts, train_lda_words, val_lda_words, train_labels, val_labels = train_test_split(
        texts, lda_words, labels, test_size=0.15, random_state=args.seed, stratify=labels
    )
    logger.info(f"Data split into training and validation sets: {len(train_texts)} train, {len(val_texts)} val")

    # creating datasets 
    train_dataset = ThemeDataset(train_texts, train_lda_words, train_labels, tokenizer, args.max_len)
    val_dataset = ThemeDataset(val_texts, val_lda_words, val_labels, tokenizer, args.max_len)

    # attaching the tokenizer to dataset for access in Optuna
    train_dataset.tokenizer = tokenizer

    # creating data loaders
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, args.batch_size, args.num_workers)

    # initialize model
    # loading the glove embeddings
    if os.path.exists(args.glove_path):
        pretrained_embeddings = load_glove_embeddings(args.glove_path, tokenizer, args.embed_dim)
    else:
        logger.warning(f"GloVe path {args.glove_path} does not exist. Using random embeddings.")
        pretrained_embeddings = None

    # hyperparameter optimization with Optuna
    def optuna_objective(trial):
        return objective(trial, args, train_loader, val_loader, device, num_classes)

    study = optuna.create_study(direction='maximize')
    study.optimize(optuna_objective, n_trials=args.optuna_trials, timeout=None)

    logger.info("Number of finished trials: {}".format(len(study.trials)))
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # saving the best hyperparameters
    best_params = trial.params
    os.makedirs(args.model_save_path, exist_ok=True)
    joblib.dump(best_params, os.path.join(args.model_save_path, 'best_params.pkl'))
    logger.info(f"Best hyperparameters saved at {os.path.join(args.model_save_path, 'best_params.pkl')}")

    # based on the best params the hidden_dim are computed
    hidden_dim = best_params['num_heads'] * best_params['hidden_dim_multiplier']
    logger.info(f"Computed hidden_dim: {hidden_dim} (num_heads: {best_params['num_heads']}, hidden_dim_multiplier: {best_params['hidden_dim_multiplier']})")

    # initializing the final model for training
    final_model = get_model(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        n_layers=best_params['n_layers'],
        bidirectional=best_params['bidirectional'],
        dropout=best_params['dropout'],
        pretrained_embeddings=pretrained_embeddings,
        max_len=args.max_len,
        num_heads=best_params['num_heads']
    )
    final_model = final_model.to(device)

    # using DataParallel for parallel processing
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        final_model = nn.DataParallel(final_model)

    # defining the optimizer and scheduler again based on best params
    optimizer = AdamW(filter(lambda p: p.requires_grad, final_model.parameters()), lr=best_params['learning_rate'], weight_decay=1e-2)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # loss function
    criterion = nn.CrossEntropyLoss()

    # capturing the metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0.0
    patience_counter = 0
    patience = args.patience

    # training loop
    for epoch in range(1, args.epochs + 1):
        final_model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = final_model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch {epoch}/{args.epochs}, Training Loss: {avg_train_loss:.4f}")

        # validation
        final_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = final_model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        logger.info(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

        # early stopping for resource saving
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            save_checkpoint(final_model, epoch, avg_val_loss, optimizer, scheduler, args.model_save_path)
            logger.info("Best model updated.")
        else:
            patience_counter += 1
            logger.info(f"Early Stopping Counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered!")
                break

        # condition to save checkpoints every N epochs :-> lookup args.epochs and args.save_every
        if epoch % args.save_every == 0 and epoch != args.epochs:
            save_checkpoint(final_model, epoch, avg_val_loss, optimizer, scheduler, args.model_save_path)

    logger.info("\nTraining complete!")

    # ploting the loss and accuracy graphs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_save_path, 'loss_graph.png'))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_save_path, 'accuracy_graph.png'))
    plt.show()

    # saving the final model
    final_model_path = os.path.join(args.model_save_path, 'final_model.pth')
    if isinstance(final_model, nn.DataParallel):
        torch.save(final_model.module.state_dict(), final_model_path)
    else:
        torch.save(final_model.state_dict(), final_model_path)
    logger.info(f"Final model saved at {final_model_path}")


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
