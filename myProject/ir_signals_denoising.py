import json
import os
import optparse
import sys
import csv
import random
import numpy as np
from collections import defaultdict
import pickle


sys.path.append("../")

from snorkel import SnorkelSession
from snorkel.models import candidate_subclass
from snorkel.models import Context, Candidate
from snorkel.contrib.models.text import RawText
from snorkel.annotations import LabelAnnotator
from snorkel.learning.gen_learning import GenerativeModel


session = SnorkelSession()

#loading data


parser = optparse.OptionParser("%prog [options]")

parser.add_option("-i", "--input-pair",   dest="input_pair", help="the pair file [default: %default]")
parser.add_option("-s", "--saved-dir",  dest="saved_dir", help="directory to save the rank scores [default: %default]")
parser.add_option("-d", "--data-split",  dest="data_split", type="int", help="percent data going to training gen model [default: %default]")


parser.set_defaults(
        input_pair       = "/Users/datienguyen/Desktop/coding/data-search/exp-data/dataSEARCH/pair-store/train.csv" #3_signals.top20doc.csv" #
        ,saved_dir      = "../../data-search/exp-data/dataSearch/pair-store/"
        ,data_split        = 10
)

opts,args = parser.parse_args(sys.argv)
input_pair =opts.input_pair
data_split = opts.data_split

session.query(Context).delete()
session.query(Candidate).delete()

values = ['positive', 'negative']
Tweet = candidate_subclass('Tweet', ['tweet'], values=values)


#item_id,worker_id,query_id,doc1,doc2,annotation
cand_dict = {}

with open(input_pair, "r") as myFile:  
    reader = csv.reader(myFile)
    for row in reader:
        #print(row)
        item_id = row[0]
        worker_id = row[1]
        anno = row[5]

        if item_id not in cand_dict:
        	cand_dict[item_id] = {}
        	cand_dict[item_id][worker_id] = anno
        else:
        	cand_dict[item_id][worker_id] = anno

print("Loading pair file done...")




#doing statistic here and sample for training the generative model

cand_list = cand_dict.keys()

count_dict = {}

for i, cand in enumerate(cand_list):
	n = len(cand_dict[cand])
	if n not in count_dict:
		count_dict[n] = [cand]
	else:
		count_dict[n].append(cand)

n_signals = len(count_dict)


train_cand_list = []

for i in range(1, n+1):
	cand_list = count_dict[i]
	random.shuffle(cand_list)
	#take 10 percent from here
	train_cand_list += cand_list[0:int(len(cand_list)*data_split/100)]


print(" -number of pairs:", len(cand_dict))
print(" -number of signals:", n)
print(" -percent of train data:", data_split)
print(" -number of pair to train GEN model", len(train_cand_list))



for i, cand in enumerate(cand_list):
	split = 0 if cand in train_cand_list else 1

	raw_text = RawText(stable_id=cand, name=cand, text=cand)
	tweet = Tweet(tweet=raw_text, split=split)
	session.add(tweet)

session.commit()

print("Commit to snorkel database done...")

#writing label generator 
def worker_label_generator(t):
	for worker_id in cand_dict[t.tweet.stable_id]:
		yield worker_id, cand_dict[t.tweet.stable_id][worker_id]

np.random.seed(1701)
labeler = LabelAnnotator(label_generator=worker_label_generator)
L_train = labeler.apply(split=0)

print(L_train.lf_stats(session))

print("Creat training data done...")
print(" -train data shape", (L_train.shape))


print("Start to train a generative model")
gen_model = GenerativeModel(lf_propensity=True)
gen_model.train(
  	L_train,
    reg_type=2,
    reg_param=0.1,
    epochs=30
)


#doing statistics
print(gen_model.learned_lf_stats())

print("Train a genetive model done...!")
train_marginals = gen_model.marginals(L_train)
print("Number of examples:", len(train_marginals))
print(train_marginals)







