import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import img_classifier as d
import testing_retrieval as t
import os
import sys
import json
import numpy as np

def Get_All_CNN_Data(order, data_dir) :
	dir_list = os.listdir(data_dir)
	data_set_CNN = torch.tensor(np.array([np.load(direc + '/' + f + '.npy') for f in list(order.keys())]), dtype=torch.float32)
	# print(data_set_CNN.shape)
	return data_set_CNN

def training_labels() :
    test = json.load(open(data_dir + 'testing_label.json','r'))
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
		# print(v)
		
		# for c in v:
		c = v[0]
		col_list = np.zeros((1,len(dict_list)))
		col_list[0,0] = 1
		c = c.split(' ')
		# print(c)
		for i in c:
			i = i.replace('.', '')
			i = i.replace(',', '')
			# print(i)
			if i != '':
				# print(i)
				ans = np.zeros((1,len(dict_list)))
				ans[0,dict_list[i]-1] = 1
				col_list = np.concatenate((col_list, ans), axis=0)
		# print(col_list)
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
	#ret_tensor = num_list[0]
	#for n in num_list[1:]:
	#	ret_tensor = torch.stack((ret_tensor,n), dim=-1)
	# print(ret_tensor.shape)
	# print(torch.transpose(ret_tensor,0,1))
	return ret_tensor

captioner = d.NNModel(2039, 4096, 2039, 1)
captioner.load_state_dict(torch.load('../model_state.pth'), weights_only=True)

criterion = nn.CrossEntropyLoss()

Adam = torch.optim.Adam(params=captioner.parameters())

def test(epoch, data, vocab, model, loss_fcn, optimizer, dict_, output_file):
	model.eval()
	test_loss, correct = 0,0
	with torch.no_grad():
		i = 0
		for conv_image in data:
			# print(conv_image.shape)
			probs = model(conv_image, 50)
			test_loss += loss_fcn(probs, vocab[i,:,:]).item()
			phrase = []
			for p in probs:
				# print(p)
				i = p.argmax(0).item()
				# print(i)
				w = list(dict_.keys())[list(dict_.values()).index(i+1)]
				# print(w)
				phrase.append(w)

			print("image: " + str(i) + " phrase: " + ' '.join(phrase))
			i+=1

def main():
	args = sys.argv[1:]
	t_labels = training_labels()
	CNN_data = Get_All_CNN_Data(t_labels, args[1])
	diclist = load_dict()
	q = list(t_labels.values())
	num_vocab = convert_to_num_array(t_labels, diclist)

	test(epoch, CNN_data, num_vocab, captioner, criterion, Adam, diclist, args[2])

