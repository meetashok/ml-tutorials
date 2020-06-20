import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from data import onehot, word_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_letter(start_tensor, model, idx_to_char):
    
    model.eval()
    with torch.set_grad_enabled(False):
        out = model(start_tensor)
        probs = F.softmax(out.squeeze(), dim=-1).cpu().detach().numpy()

        topk = np.argsort(probs)[::-1][:10]
        probs_top = probs[topk] / sum(probs[topk])

        index = np.random.choice(topk, 1, p=probs_top)[0]
        # index = torch.argmax(out.squeeze()).item()
        letter = idx_to_char[index]
    return index, letter

def generate_word(model, max_len, char_to_idx, idx_to_char, max_length=20):
    model.init_hidden(1)
    letters = ""
    
    start_tensor = torch.zeros(size=(1,1,max_len), dtype=torch.float32)
    start_tensor[0,0,char_to_idx["!"]] = 1
    start_tensor = start_tensor.to(device)
    
    index, letter = generate_letter(start_tensor, model, idx_to_char)
    letters += letter
    while (letter != " " or len(letters.strip()) < 1):
        start_tensor = torch.zeros(size=(1,1,max_len), dtype=torch.float32)
        start_tensor[0,0,char_to_idx[letter]] = 1
        start_tensor = start_tensor.to(device)
        index, letter = generate_letter(start_tensor, model, idx_to_char)
        letters += letter
        if len(letters) > max_length:
            break

    return letters.strip()


