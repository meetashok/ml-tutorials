import torch
import torch.nn as nn
from torch.autograd import Variable

class TextGeneration(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TextGeneration, self).__init__()
        # self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, output_size, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(hidden_size, output_size)
        self.n_layers = 1
        self.hidden_size = hidden_size

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        return out

    def init_hidden(self, n_seqs):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.n_layers, n_seqs, self.hidden_size).zero_()),
                Variable(weight.new(self.n_layers, n_seqs, self.hidden_size).zero_()))

if __name__ == "__main__":
    tensor = torch.randn(size=(1, 3, 27))
    # tensor = torch.tensor(tensor, dtype=torch.float32)
    print(tensor.shape, tensor.dtype)
    model = TextGeneration(27, 10, 27)
    out = model(tensor)
    print(out.shape)