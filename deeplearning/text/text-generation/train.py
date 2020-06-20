import torch
import torch.nn as nn
import torch.optim as optim

from data import download_data, url, vocab, index_lists, word_to_tensor, shuffle_words
from model import TextGeneration
from lm import generate_word

n_epochs = 1000
lr = 0.01
print_every = 100
embedding_dim = 30
hidden_size = 50
batch_size = 32
seq_length = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up data 
dino_names = download_data(url)
max_len, (char_to_idx, idx_to_char) = vocab(dino_names)

words = dino_names.split()
start_indices, end_indices = index_lists(words, char_to_idx)
n_chars = len(start_indices)
n_batches = n_chars // (batch_size * seq_length)

# set up model 
model = TextGeneration(max_len, hidden_size, max_len)
loss_function = nn.CrossEntropyLoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=lr)
model.to(device)

# model training 

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    with torch.set_grad_enabled(True):
        for word in words:
            model.init_hidden(len(word)+1)
            s, e = word_to_tensor(word, max_len, char_to_idx)
            
            s = s.to(device)
            e = e.to(device)
        
            out = model(s)
            # print(out.shape, e.shape)
            loss = loss_function(out.squeeze(), e.squeeze())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
        
        epoch_loss /= len(words)

        generated_word = generate_word(model, max_len, char_to_idx, idx_to_char)
        print(f"Epoch: {epoch+1:4,}, loss: {epoch_loss:.4f} - {generated_word}")
