import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

LEARNING_RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 64
EMBEDDING_DIM = 50
KERNEL_WIDTH = 3
OUT_CHANNELS = 8
POOL_SIZE = 8
SEQ_LENGTH = 100 
DENSE_IN_FEATURES = 12 * OUT_CHANNELS
FILE_PATH = './data/imdb_dataset.csv'

class IMDB_CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_channels, kernel_width, pool_size, dense_in_features):
        super(IMDB_CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=out_channels,
            kernel_size=kernel_width
        )
        self.relu = nn.ReLU()
        self.maxpool1d = nn.MaxPool1d(kernel_size=pool_size)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(dense_in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool1d(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

class IMDBDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.LongTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].unsqueeze(0)

def load_and_preprocess_data(path, seq_len):
    df = pd.read_csv(path)
    tokenized_reviews = [review.lower().split() for review in df['review']]
    word_counts = Counter(word for review in tokenized_reviews for word in review)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    word_to_idx = {word: i+2 for i, word in enumerate(vocab)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    numerical_reviews = [[word_to_idx.get(word, 1) for word in review] for review in tokenized_reviews]
    padded_reviews = np.zeros((len(numerical_reviews), seq_len), dtype=int)
    for i, review in enumerate(numerical_reviews):
        if len(review) > seq_len:
            padded_reviews[i] = review[:seq_len]
        else:
            padded_reviews[i, :len(review)] = review
    labels = (df['sentiment'] == 'positive').astype(np.float32).values
    return padded_reviews, labels, word_to_idx

features, labels, word_to_idx = load_and_preprocess_data(FILE_PATH, SEQ_LENGTH)
VOCAB_SIZE = len(word_to_idx)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
train_dataset = IMDBDataset(X_train, y_train)
test_dataset = IMDBDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = IMDB_CNN(VOCAB_SIZE, EMBEDDING_DIM, OUT_CHANNELS, KERNEL_WIDTH, POOL_SIZE, DENSE_IN_FEATURES)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    print(f"--- Epoch {epoch+1}/{EPOCHS} ---")

    for i, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_function(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 50 == 0:
            print(f"  Batch {i+1}/{len(train_loader)}")

    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_pred = model(x_batch)
            preds = (y_pred > 0.5).float()
            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

    accuracy = (total_correct / total_samples) * 100
    elapsed_time = time.time() - start_time
    avg_train_loss = total_loss / len(train_loader)
    
    print(f"\nEpoch {epoch+1} ended in: {elapsed_time:.2f}s. Avg loss: {avg_train_loss:.4f}, accuracy od test set: {accuracy:.2f}%\n")