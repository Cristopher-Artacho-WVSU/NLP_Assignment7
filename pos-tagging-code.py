import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import treebank
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict, Counter
import time
import random


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


nltk.download('treebank')
nltk.download('universal_tagset')

print("Loading the Penn Treebank dataset...")


tagged_sentences = list(treebank.tagged_sents(tagset='universal'))


print(f"Number of tagged sentences: {len(tagged_sentences)}")
print(f"Example of a tagged sentence: {tagged_sentences[0]}")

all_tags = set()
for sentence in tagged_sentences:
    for _, tag in sentence:
        all_tags.add(tag)
print(f"Unique POS tags: {all_tags}")
print(f"Number of unique tags: {len(all_tags)}")

train_sentences, test_sentences = train_test_split(tagged_sentences, test_size=0.2, random_state=42)

print(f"Training set size: {len(train_sentences)} sentences")
print(f"Test set size: {len(test_sentences)} sentences")

# -------------------- MODEL 1: HIDDEN MARKOV MODEL --------------------- #
print("\n--- Training Hidden Markov Model ---")

class HiddenMarkovModel:
    def __init__(self):
        self.tags = set()
        self.words = set()
        self.word_counts = Counter()
        self.tag_counts = Counter()
        self.emission_counts = defaultdict(Counter)
        self.transition_counts = defaultdict(Counter)
        self.initial_counts = Counter()
        
        self.emission_smoothing = 1e-5
        self.transition_smoothing = 1e-5
        
    def train(self, train_data):
        for sentence in train_data:
            prev_tag = None
            for i, (word, tag) in enumerate(sentence):
                self.words.add(word)
                self.tags.add(tag)
                self.word_counts[word] += 1
                self.tag_counts[tag] += 1
                self.emission_counts[tag][word] += 1
                
                if i == 0:
                    self.initial_counts[tag] += 1
                else:
                    self.transition_counts[prev_tag][tag] += 1
                prev_tag = tag

        self.calc_probabilities()
    
    def calc_probabilities(self):
        self.initial_probs = {tag: count / len(self.initial_counts) 
                               for tag, count in self.initial_counts.items()}
        
        self.transition_probs = {}
        for prev_tag in self.tags:
            self.transition_probs[prev_tag] = {}
            total = sum(self.transition_counts[prev_tag].values()) + self.transition_smoothing * len(self.tags)
            for tag in self.tags:
                self.transition_probs[prev_tag][tag] = (self.transition_counts[prev_tag][tag] + 
                                                        self.transition_smoothing) / total
        
        self.emission_probs = {}
        for tag in self.tags:
            self.emission_probs[tag] = {}
            total = sum(self.emission_counts[tag].values()) + self.emission_smoothing * len(self.words)
            for word in self.words:
                self.emission_probs[tag][word] = (self.emission_counts[tag][word] + 
                                                   self.emission_smoothing) / total
    
    def viterbi_algorithm(self, sentence):
        T = len(sentence)
        N = len(self.tags)
        tags_list = list(self.tags)
        
        viterbi = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        
        unknown_word_prob = 1e-10
        
        for i, tag in enumerate(tags_list):
            initial_prob = self.initial_probs.get(tag, 1e-10)
            word = sentence[0]
            if word in self.words:
                emission_prob = self.emission_probs[tag].get(word, unknown_word_prob)
            else:
                emission_prob = unknown_word_prob
            viterbi[i, 0] = np.log(initial_prob) + np.log(emission_prob)
        
        for t in range(1, T):
            word = sentence[t]
            word_in_vocab = word in self.words
            
            for i, tag in enumerate(tags_list):
                max_prob = float('-inf')
                max_idx = 0
                
                if word_in_vocab:
                    emission_prob = self.emission_probs[tag].get(word, unknown_word_prob)
                else:
                    emission_prob = unknown_word_prob
                
                for j, prev_tag in enumerate(tags_list):
                    transition_prob = self.transition_probs[prev_tag].get(tag, self.transition_smoothing)
                    
                    prob = viterbi[j, t-1] + np.log(transition_prob)
                    
                    if prob > max_prob:
                        max_prob = prob
                        max_idx = j
                
                viterbi[i, t] = max_prob + np.log(emission_prob)
                backpointer[i, t] = max_idx
        
        best_path = [0] * T
        best_path_prob = float('-inf')
        
        for i, tag in enumerate(tags_list):
            if viterbi[i, T-1] > best_path_prob:
                best_path_prob = viterbi[i, T-1]
                best_path[T-1] = i
        
        for t in range(T-1, 0, -1):
            best_path[t-1] = backpointer[best_path[t], t]
        
        predicted_tags = [tags_list[idx] for idx in best_path]
        
        return predicted_tags
    
    def predict(self, sentences):
        predictions = []
        for sentence in sentences:
            words = [word for word, _ in sentence]
            predicted_tags = self.viterbi_algorithm(words)
            predictions.append(predicted_tags)
        return predictions

start_time = time.time()
hmm = HiddenMarkovModel()
hmm.train(train_sentences)
hmm_training_time = time.time() - start_time
print(f"HMM training completed in {hmm_training_time:.2f} seconds")

start_time = time.time()
hmm_predictions = hmm.predict(test_sentences)
hmm_inference_time = time.time() - start_time
print(f"HMM inference completed in {hmm_inference_time:.2f} seconds")

hmm_true_tags = []
hmm_pred_tags = []
for i, sentence in enumerate(test_sentences):
    true_tags = [tag for _, tag in sentence]
    pred_tags = hmm_predictions[i]
    
    min_len = min(len(true_tags), len(pred_tags))
    hmm_true_tags.extend(true_tags[:min_len])
    hmm_pred_tags.extend(pred_tags[:min_len])

hmm_accuracy = accuracy_score(hmm_true_tags, hmm_pred_tags)
print(f"HMM Accuracy: {hmm_accuracy:.4f}")
print("\nHMM Classification Report:")
print(classification_report(hmm_true_tags, hmm_pred_tags))

# -------------------- MODEL 2: BIDIRECTIONAL LSTM NETWORK --------------------- #
print("\n--- Training Bidirectional LSTM Network ---")

# Create vocabularies and mappings
def build_vocab(sentences):
    word_counts = Counter()
    tag_counts = Counter()
    
    for sentence in sentences:
        for word, tag in sentence:
            word_counts[word] += 1
            tag_counts[tag] += 1

    word_to_idx = {word: i+1 for i, (word, _) in enumerate(word_counts.most_common())}
    word_to_idx['<PAD>'] = 0  
    word_to_idx['<UNK>'] = len(word_to_idx) 
    
    tag_to_idx = {tag: i for i, (tag, _) in enumerate(tag_counts.most_common())}
    
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    
    return word_to_idx, tag_to_idx, idx_to_word, idx_to_tag

class POSDataset(Dataset):
    def __init__(self, sentences, word_to_idx, tag_to_idx):
        self.sentences = sentences
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        words = [word for word, _ in sentence]
        tags = [tag for _, tag in sentence]
        
        word_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        tag_indices = [self.tag_to_idx[tag] for tag in tags]
        
        return torch.tensor(word_indices), torch.tensor(tag_indices)

def collate_fn(batch):
    words, tags = zip(*batch)
    
    words_padded = pad_sequence(words, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=-1)
    
    return words_padded, tags_padded

class BiLSTM_POS_Tagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim // 2, 
                           num_layers=1,
                           bidirectional=True,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.lstm(embedded)
        predictions = self.fc(outputs)
        
        return predictions

word_to_idx, tag_to_idx, idx_to_word, idx_to_tag = build_vocab(train_sentences)
vocab_size = len(word_to_idx)
num_tags = len(tag_to_idx)

print(f"Vocabulary size: {vocab_size}")
print(f"Number of tags: {num_tags}")

train_dataset = POSDataset(train_sentences, word_to_idx, tag_to_idx)
test_dataset = POSDataset(test_sentences, word_to_idx, tag_to_idx)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

embedding_dim = 100
hidden_dim = 128
model = BiLSTM_POS_Tagger(vocab_size, embedding_dim, hidden_dim, num_tags, 0)
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-1) 
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for words, tags in dataloader:
        words = words.to(device)
        tags = tags.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(words)
        
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        
        loss = criterion(predictions, tags)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    all_predictions = []
    all_tags = []
    
    with torch.no_grad():
        for words, tags in dataloader:
            words = words.to(device)
            tags = tags.to(device)
            
            predictions = model(words)
            
            predictions_flat = predictions.view(-1, predictions.shape[-1])
            tags_flat = tags.view(-1)
            
            loss = criterion(predictions_flat, tags_flat)
            epoch_loss += loss.item()
            
            _, predicted = torch.max(predictions, dim=2)
            
            for i in range(len(tags)):
                length = (tags[i] != -1).sum().item()
                all_predictions.extend(predicted[i, :length].cpu().numpy())
                all_tags.extend(tags[i, :length].cpu().numpy())
    
    accuracy = accuracy_score(all_tags, all_predictions)
    
    true_tags = [idx_to_tag[idx] for idx in all_tags]
    pred_tags = [idx_to_tag[idx] for idx in all_predictions]
    
    return epoch_loss / len(dataloader), accuracy, true_tags, pred_tags

train_losses = []

def train_bilstm(num_epochs = 5):
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_dataloader, optimizer, criterion, device)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

    bilstm_training_time = time.time() - start_time
    print(f"BiLSTM training completed in {bilstm_training_time:.2f} seconds")

    start_time = time.time()
    test_loss, bilstm_accuracy, true_tags, pred_tags = evaluate_model(model, test_dataloader, criterion, device)
    bilstm_inference_time = time.time() - start_time
    print(f"BiLSTM inference completed in {bilstm_inference_time:.2f} seconds")

    print(f"BiLSTM Test Loss: {test_loss:.4f}")
    print(f"BiLSTM Accuracy: {bilstm_accuracy:.4f}")
    print("\nBiLSTM Classification Report:")
    print(classification_report(true_tags, pred_tags))

    return [bilstm_accuracy, bilstm_inference_time, bilstm_training_time]

bilstm_perf_5 = train_bilstm(num_epochs=5)
print(bilstm_perf_5)
bilstm_perf_10 = train_bilstm(num_epochs=10)

# -------------------- MODEL COMPARISON --------------------- #
print("\n--- Model Comparison ---")

def compare_bilstm(bilstm_perf, epoch):
    print(f"---------------EPOCH {epoch} ----------------")
    # Compare accuracies
    print(f"HMM Accuracy: {hmm_accuracy:.4f}")
    print(f"BiLSTM Accuracy: {bilstm_perf[0]}")

    # Compare training times
    print(f"HMM Training Time: {hmm_training_time:.2f} seconds")
    print(f"BiLSTM Training Time: {bilstm_perf[2]} seconds")

    # Compare inference times
    print(f"HMM Inference Time: {hmm_inference_time:.2f} seconds")
    print(f"BiLSTM Inference Time: {bilstm_perf[1]} seconds")

    return bilstm_perf[0]

bilstm5 = compare_bilstm(bilstm_perf_5, epoch=5)
bilstm10 = compare_bilstm(bilstm_perf_10, epoch=10)

num_epochs = 5 if bilstm5 > bilstm10 else 10 
bilstm_accuracy = compare_bilstm(bilstm10, epoch=num_epochs)

hmm_cm = confusion_matrix(hmm_true_tags, hmm_pred_tags, labels=list(all_tags))
bilstm_cm = confusion_matrix(true_tags, pred_tags, labels=list(all_tags))

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_losses, marker='o', linestyle='-')
plt.title('BiLSTM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('bilstm_training_loss.png')
plt.close()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(hmm_cm, annot=False, cmap='Blues', fmt='d', xticklabels=list(all_tags), yticklabels=list(all_tags))
plt.title('HMM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(bilstm_cm, annot=False, cmap='Blues', fmt='d', xticklabels=list(all_tags), yticklabels=list(all_tags))
plt.title('BiLSTM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

plt.figure(figsize=(8, 6))
models = ['HMM', 'BiLSTM']
accuracies = [hmm_accuracy, bilstm_accuracy]
plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
plt.grid(axis='y', alpha=0.3)
plt.savefig('accuracy_comparison.png')
plt.close()

best_model = "BiLSTM" if bilstm_accuracy > hmm_accuracy else "HMM"
print(f"\nBest performing model: {best_model}")

print("\nConclusion:")
if best_model == "BiLSTM":
    print("The BiLSTM model outperformed the HMM for POS tagging, demonstrating the advantage of neural approaches for sequential linguistic tasks.")
else:
    print("Interestingly, the traditional HMM model performed better than the BiLSTM for this POS tagging task, which might be due to the dataset size or complexity.")
