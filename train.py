import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import tokenize, stem, bag_of_words
from model import NeuralNet

with open("intents.json",'r') as file:
    intents = json.load(file)

all_words = []
tags = []
xy = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
    
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0) # increase number of workes according to your wish

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(len(X_train[0]), 8, len(tags)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device).long()

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# some hyperparameters that can be used to tune the model

# data = {
#     "model_state": model.state_dict(),
#     "input_size": len(X_train[0]),
#     "output_size": len(tags),
#     "hidden_size": 8,
#     "all_words": all_words,
#     "tags": tags
# }

FILE = "data.pth"
torch.save(model, FILE)

print('Training Completed')