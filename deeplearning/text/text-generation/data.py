import requests 
import random
import string
import torch

from torch.utils.data import DataLoader, Dataset

url = "https://raw.githubusercontent.com/AdalbertoCq/Deep-Learning-Specialization-Coursera/master/Sequence%20Models/week1/Character-level%20language%20model/dinos.txt"

def download_data(url):
    response = requests.get(url)
    text = response.text
    dino_names = [name.lower().strip() for name in text.split("\n")]

    unique_characters = len(set("".join(dino_names)))
    total_characters = sum(len(name) for name in dino_names)

    print(f"Number of dino names = {len(dino_names):,.0f}")
    print(f"Total chars = {total_characters:,}, Unique chars = {unique_characters:,}")
    
    sample_names = [dino_names[random.choice(range(len(dino_names)))] for i in range(5)]
    print(f"Sample names: {' '.join(sample_names)}")

    return " ".join(dino_names)

def vocab(corpus):
    characters = set(corpus)

    max_len = len(characters) + 1
    idx_to_char = dict([(i, c) for (i, c) in enumerate(characters)])
    char_to_idx = dict([(c,i) for (i,c) in idx_to_char.items()])

    char_to_idx["!"] = max_len - 1
    idx_to_char[max_len-1] = "!"

    return max_len, (char_to_idx, idx_to_char)

def onehot(char, max_len, char_to_idx):
    index_tensor = torch.zeros(max_len, dtype=torch.float32)

    if char in char_to_idx:
        index = char_to_idx[char]
        index_tensor[index] = 1
    
    return index_tensor.unsqueeze(0).unsqueeze(0)

def shuffle_words(corpus):
    words = corpus.split()
    random.shuffle(words)
    return words

def word_to_tensor(word, max_len, char_to_idx):
    start_tensor = torch.zeros(size=(1, len(word)+1, max_len))
    end_tensor = torch.zeros(size=(1, len(word) + 1), dtype=torch.long)
    for i in range(len(word)+1):
        if i == 0:
            start_tensor[0,i,char_to_idx["!"]] = 1
            end_tensor[0, i] = char_to_idx[word[i]]
        elif i == len(word):
            start_tensor[0,i,char_to_idx[word[i-1]]] = 1
            end_tensor[0, i] = char_to_idx[" "]
        else:
            start_tensor[0,i,char_to_idx[word[i-1]]] = 1
            end_tensor[0, i] = char_to_idx[word[i]]
        
    return start_tensor, end_tensor

def word_to_indices(word, char_to_idx):
    start, end = [], []
    for i, letter in enumerate(word):
        if i == 0:
            start += [char_to_idx["!"]]
            end += [char_to_idx[letter]]
        elif i == len(word) - 1:
            start += [char_to_idx[letter]]
            end += [char_to_idx[" "]]
        else:
            start += [char_to_idx[letter]]
            end += [char_to_idx[word[i+1]]]
    return start, end

def index_lists(words, char_to_idx):
    start_indices, end_indices = [], []
    for word in words:
        start, end = word_to_indices(word, char_to_idx)
        start_indices += start
        end_indices += end

    return start_indices, end_indices

if __name__ == "__main__":    
    dino_names = download_data(url)
    max_len, (char_to_idx, idx_to_char) = vocab(dino_names)

    # letters = ['a', None]
    # for letter in letters:
    #     print(f"{letter}: {onehot(letter, max_len, char_to_idx)}")

    word_tensor("ashok")


