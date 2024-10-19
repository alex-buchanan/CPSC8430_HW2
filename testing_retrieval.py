import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import img_classifier as d
import os
import sys
import json
import numpy as np

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def Get_All_CNN_Data(order) :
	direc = "training_data//feat"
	dir_list = os.listdir(direc)
	data_set_CNN = torch.tensor(np.array([np.load(direc + '/' + f + '.npy') for f in list(order.keys())[:25]]), dtype=torch.float32)
	return data_set_CNN

def training_labels() :
    test = json.load(open('training_label.json','r'))
    dict_list = {}
    for item in test:
        captions = [x.rstrip('.') for x in item['caption']]
        dict_list[item['id']] = captions
    return dict_list

def load_dict():
	dList = {}
	i = 1
	f = open("dict.txt", "r")
	for l in f:
		dList[l[:-1]] = i
		i+=1
	f.close()
	return dList

def convert_to_num_array(dictionary, dict_list):
	num_list = []
	row_list = []
	j = 0
	for v in dictionary.values():
		c = v[0]
		col_list = np.zeros((1,len(dict_list)))
		col_list[0,0] = 1
		c = c.split(' ')
		for i in c:
			i = i.replace('.', '')
			i = i.replace(',', '')
			if i != '':
				ans = np.zeros((1,len(dict_list)))
				ans[0,dict_list[i]-1] = 1
				col_list = np.concatenate((col_list, ans), axis=0)
		for i in range(col_list.shape[0],49):
			ans = np.zeros((1,len(dict_list)))
			ans[0,3] = 1
			col_list = np.concatenate((col_list, ans), axis=0)
		ans = np.zeros((1,len(dict_list)))
		ans[0,1] = 1
		col_list = np.concatenate((col_list, ans), axis=0)
		row_list.append(col_list)
		j+=1

	ret_tensor = torch.tensor(row_list, dtype=torch.float32)

	return ret_tensor

captioner = d.NNModel(2039, 4096, 2039, 1)

criterion = nn.CrossEntropyLoss()

Adam = torch.optim.Adam(params=captioner.parameters())

t_labels = training_labels()
CNN_data = Get_All_CNN_Data(t_labels)
diclist = load_dict()
q = list(t_labels.values())
num_vocab = convert_to_num_array(t_labels, diclist)

def train(epoch, data, vocab, model, loss_fcn, optimizer, loss_, accy_):
	i = 0
	for conv_image in data:
		model.train()
		train_loss = 0.0

		# Calculate Probabilities
		probs = model(conv_image, 50)

		# Calculate loss, and backpropagate
		loss = loss_fcn(probs, vocab[i,:,:])
		train_loss += loss.item()

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		i+=1

		loss, current = loss.item(), i
		print(f"loss: {loss:>7f}  [{current:>3d}/{len(data):>3d}]")


train_loss = []
train_accy = []

tgs = list(t_labels.keys())

for epoch in range(0, 20) :
	print(f"Epoch {epoch+1}\n-------------------------------")
	train(epoch, CNN_data, num_vocab, captioner, criterion, Adam, train_loss, train_accy)

torch.save(captioner.state_dict(), './model_state.pth')
