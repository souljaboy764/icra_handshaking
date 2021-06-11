import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

from util import SkeletonSequenceDataset

import json
import datetime
import os
import pickle
import argparse

from networks import *

import sys

parser = argparse.ArgumentParser(description='LSTM Hand prediction')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='Input batch size for training (default: 128).')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='LR',
                    help='Learning rate for the optimizer (default: 1e-3).')
parser.add_argument('--epochs', type=int, default=200, metavar='E',
                    help='Number of epochs to train (default: 25).')
parser.add_argument('--hidden-dim', type=int, default=64, metavar='H',
                    help='Dimensions of the hidden layers (default: 100).')
parser.add_argument('--seed', type=int, default=89514, metavar='S',
                    help='Random seed (default: 89514).')
parser.add_argument('--log-interval', type=int, default=10, metavar='L',
                    help='How many batches to wait before logging training status.')
parser.add_argument('--src-file', type=str, metavar='SRC', required=True,
                    help='Directory to the npy files where the skeletons are stored.')
parser.add_argument('--model-dir', type=str, default="models/"+datetime.datetime.now().strftime("%m%d%H%M"),metavar='MODEL_DIR',
                    help='Directory where the model files should be saved to.')
parser.add_argument('--checkpoint', type=str, default=None, metavar='CKPT',
                    help='Path to saved checkpoint from which training should resume. (default: None)')
args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)


print('Loading Dataset')
train_dataset = SkeletonSequenceDataset(args.src_file)
train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
print("Training Dataset Size:",len(train_dataset))
test_dataset = SkeletonSequenceDataset(args.src_file, args.seq_len, False)
test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
print("Testing Dataset Size:",len(test_dataset))
print(test_dataset[0][0][0][:-4].shape)
INPUT_DIM = np.prod(train_dataset[0][0][0][:-4].shape[0]* train_dataset[0][0][0].shape[1])

print('Creating Model')
model = Model(INPUT_DIM, args.hidden_dim).to(device)


# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
params = list(model.parameters())

completed_epochs = 0
if args.checkpoint is not None and os.path.exists(args.checkpoint):
	try:
		checkpoint = torch.load(args.checkpoint)
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		completed_epochs = checkpoint['epoch']
	except Exception as e:
		print('Exception occurred while loading checkpoint:')
		print(e)
		print('Continuing without loading checkpoint')

if not os.path.exists(args.model_dir):
	os.mkdir(args.model_dir)
logfile_name = os.path.join(args.model_dir,'output.log')

print('Models shall be saved to:', args.model_dir)
print('Log shall be written to:', logfile_name)

logfile = open(logfile_name, 'w')
logfile.write('Args ' + str(args))
print(args)

def run(epoch, dataset_iterator):
	total_loss = 0
	total_mse_losses = 0
	total_kl_losses = 0
	torch.autograd.set_detect_anomaly(True)

	for i, x in enumerate(dataset_iterator):
		#flatten input
		x[1], perm_idx = x[1].sort(0, descending=True)
		x[0] = x[0][perm_idx]
		x[0] = x[0].to(device)
		x[0] = x[0][:,:,:-4]
		optimizer.zero_grad()
		batch, seq, joints, values = x[0].shape
		x[0] = x[0].view(batch, seq, joints * values)
		# pack = torch.nn.utils.rnn.pack_padded_sequence(Variable(x[0]), x[1], batch_first=True)
		# forward pass
		pred = model(x[0])
		if torch.any(torch.isinf(pred)):
			print("PRED INF")
		label = x[0][:,-1,11*3:12*3].unsqueeze(1).repeat(1,pred.shape[1],1)
		# losses = torch.zeros(pred.shape[0]).to(device)
		# for j in range(pred.shape[0]):
		# 	losses[j] = F.mse_loss(pred[j,:x[1][j]-1], x[0][j,x[1][j]-1,11*3:12*3].repeat(x[1][j]-1,1))
			# losses[j] = F.mse_loss(pred[j], x[0][j,-1,7*3:8*3].repeat(x[1][j]-1,1))
		# loss = torch.mean(losses)
		loss = F.mse_loss(pred, label)
		if torch.any(torch.isinf(loss)):
			print("LOSS INF")
		
		if model.training:
			loss.backward()
			# grads = []
			# for p in params:
			# 	grads.append(p.grad.cpu().detach().numpy().mean())
			# print('GRADS:', np.mean(grads))
			# update the weights
			optimizer.step()
	return loss

		

for e in range(args.epochs):
	
	train_loss = 0
	
	test_loss = 0
	
	model.train()
	train_loss = run(e+completed_epochs+1, train_iterator)
		
	model.eval()
	test_loss = run(e+completed_epochs+1, test_iterator)

	if (e+1)%10==0:
		# print(f'Epoch {e+completed_epochs+1}, Train Loss: {train_loss}, Test Loss: {test_loss}')
		print('Epoch', e+completed_epochs+1, 'Train Loss:', train_loss.cpu().detach().numpy(), 'Test Loss:', test_loss.cpu().detach().numpy())
		logfile.write('Epoch ' + str(e+completed_epochs+1) + 'Train Loss: ' + str(train_loss.cpu().detach().numpy()) + 'Test Loss: ' + str(test_loss.cpu().detach().numpy()))
		checkpoint_file = os.path.join(args.model_dir, '%0.4d.pth'%(e+completed_epochs+1))
		torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': e+completed_epochs+1}, checkpoint_file)
checkpoint_file = os.path.join(args.model_dir, 'final.pth')
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': e+completed_epochs+1}, checkpoint_file)


model.eval()
preds = []
inputs = []
for i, x in enumerate(test_iterator):
	x[1], perm_idx = x[1].sort(0, descending=True)
	x[0] = x[0][perm_idx]
	x[0] = x[0].to(device)
	x[0] = x[0][:,:,:-4]
	batch, seq, joints, values = x[0].shape
	x[0] = x[0].view(batch, seq, joints * values)
	# pack = torch.nn.utils.rnn.pack_padded_sequence(Variable(x[0]), x[1], batch_first=True)
	# forward pass
	pred = model(x[0])
	for j in range(pred.shape[0]):
		inputs.append(x[0][j,:x[1][j]-1].cpu().numpy())
		preds.append(pred[j,:x[1][j]-1].cpu().detach().numpy())

np.savez_compressed(os.path.join(args.model_dir,'handreach_skeletons_pred.npz'), inputs=inputs, preds=preds)