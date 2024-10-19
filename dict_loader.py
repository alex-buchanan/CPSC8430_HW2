import math
import operator
import sys
import json
from functools import reduce 

# Create a dictionary list of words such that no word is repeated and the dictionary draws from the training and testing JSON files
if __name__ == "__main__" :
    f = open("dict.txt", "a")
    test = json.load(open('training_label.json','r'))
    dict_list = []
    for item in test:
    	desc = item['caption']
    	a = desc[0].split(' ')
    	for i in a:
    		i = i.replace('.', '')
    		i = i.replace(',', '')
    		dict_list.append(i)
    dict_list = list(set(dict_list))
    for k in dict_list:
    	f.write(k)
    	f.write("\n")
    f.close()