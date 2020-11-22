import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as f


#Create fully connected network / model

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size * sequence_length, num_classes)





	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)


		#forward

		out, _ = self.rnn(x, h0)
		out = out.reshape(out.shape[0], -1)

		out = self.fc(out)

		return out



