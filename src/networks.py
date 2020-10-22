import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Model(nn.Module):
	"""[Model for predicting Hand location using LSTM]
	
	Args:
		INPUT_DIM (int): [Input Dimension of the Encoder]
		HIDDEN_DIM (int): [Number of the states in the hidden layers]
	"""

	def __init__(self, INPUT_DIM, HIDDEN_DIM=64):
		super(Model, self).__init__()
		self.HIDDEN_DIM = HIDDEN_DIM
		self._hidden0 = nn.LSTM(INPUT_DIM, HIDDEN_DIM, 1, batch_first=True)
		self._hidden1 = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, 1, batch_first=True)
		self._output = nn.Linear(HIDDEN_DIM, 3)

	def forward(self, x):
		"""[Forward pass of the model.]

		Args:
			x (Tensor [batchsize, INPUT_DIM]): [Input sequence]
		
		Returns:
			pred (Tensor [batchsize, 3]): [Predicted Final Hand Location]
		"""
		# encode 
		encoded0, state0 = self._hidden0(x, self.init_hidden(int(x.shape[0])))
		# encoded0, input_sizes = nn.utils.rnn.pad_packed_sequence(encoded0, batch_first=True)
		
		encoded0 = nn.Dropout(p=0.2)(nn.Tanh()(encoded0))
		
		# encoded0 = torch.nn.utils.rnn.pack_padded_sequence(encoded0, input_sizes, batch_first=True)
		
		encoded1, state1 = self._hidden1(encoded0, state0)#self.init_hidden(int(encoded0.shape[0])))
		# encoded1, input_sizes = nn.utils.rnn.pad_packed_sequence(encoded1, batch_first=True)

		encoded1 = nn.Dropout(p=0.2)(nn.Tanh()(encoded1))
		
		pred = self._output(encoded1)
		
		return pred
	
	def init_hidden(self, batch_size):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		hidden = (torch.randn(1, batch_size, self.HIDDEN_DIM).to(device), torch.randn(1, batch_size, self.HIDDEN_DIM).to(device))
		return hidden