import torch.nn as nn
import torch
import torch.nn.functional as f


#C  
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms




class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
                
        # Forward propagate RNN
    
        out, _ = self.lstm(x, (h0,c0))  # or x, h
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
     
        # Decode the hidden state of the last time step
        out = out[:, -1, :]   
         
        out = self.fc(out)
        # out: (n, 10)
        return out