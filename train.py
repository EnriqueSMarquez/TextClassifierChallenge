#SVM + bag of words
from data_processing import FullOneHot, RawDataYielder
import argparse
from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from scipy import sparse
import os
import pickle
import numpy as np
import joblib
from tqdm import tqdm
import json
# from attention_models import SimpleSelfAttention
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import TransformerTextClassifier

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def pad_collate_with_value(token_value):
    """Collate fn to pad sentences given the maximum number of regions on the batch"""
    def f(batch):
        length = np.max([len(x[0]) for x in batch])
        padded_sequence = token_value*np.ones((len(batch),length)).astype(int)
        padded_targets = np.zeros((len(batch),length)).astype(int)
        padded_positions_x = np.zeros((len(batch),length,batch[0][2].shape[-1])).astype(int)
        padded_positions_y = np.zeros((len(batch),length,batch[0][3].shape[-1])).astype(int)
        padded_positions_page = np.zeros((len(batch),length,batch[0][4].shape[-1])).astype(int)
        for i,(sequence,target,positions_x,positions_y,positions_page) in enumerate(batch):
            padded_sequence[i,0:len(sequence)] = sequence
            padded_targets[i,0:len(sequence)] = target
            padded_positions_x[i,0:len(sequence),:] = positions_x
            padded_positions_y[i,0:len(sequence),:] = positions_y
            padded_positions_page[i,0:len(sequence),:] = positions_page
        return [torch.LongTensor(padded_sequence),
               torch.FloatTensor(padded_targets),
               torch.FloatTensor(padded_positions_x),
               torch.FloatTensor(padded_positions_y),
               torch.FloatTensor(padded_positions_page)]
    return f

def train_epoch(model, train_iter, epoch,loss_fn):
    total_epoch_loss = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    model.train()
    pbar = tqdm(enumerate(train_iter),total=int(np.floor(len(train_iter.dataset)/train_iter.batch_size)))
    for idx, batch in pbar:
        sequence = batch[0]
        target = batch[1]
        positions_x = batch[2]
        positions_y = batch[3]
        positions_page = batch[4]
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            sequence = sequence.cuda()
            target = target.cuda()
            positions_x = positions_x.cuda().float()
            positions_y = positions_y.cuda().float()
            positions_page = positions_page.cuda().float()
        optim.zero_grad()
        prediction = model(sequence,positions_x,positions_y,positions_page)
        loss = loss_fn(prediction, target.float())
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        total_epoch_loss += loss.item()
        pbar.set_description('Mean loss : {:.3f}'.format(total_epoch_loss/(idx+1)))
    return total_epoch_loss/len(train_iter)

parser = argparse.ArgumentParser(description='Train model on hypatos data')
parser.add_argument('--datapath', type=str, help='Path to train json')
parser.add_argument('--model',type=str,default='svm',help='Model type: svm or selfattention')
parser.add_argument('--seed',type=int,default=7)
parser.add_argument('--saving-path',dest='saving_path',type=str,default='',help='Path to save models')
parser.add_argument('--feature-compression',dest='feature_compression',type=str,default='bow',help='Type of compression: bow or raw')
parser.add_argument('--verbose',type=int,default=1)
parser.add_argument('--nb-jobs',dest='nb_jobs',type=int,default=6)
parser.add_argument('--dataset-chopped-portion',dest='dataset_chopped_portion',type=float,default=0.,help='Portion of data to ignore')
parser.add_argument('--min-repetitions',dest='min_repetitions',type=int,default=10,help='Min repetitions to be in dictionary')
parser.add_argument('--epochs',type=int,default=20,help='Number of training epochs')
parser.add_argument('--batch-size',dest='batch_size',default=4,type=int,help='Training batch size')
args = parser.parse_args()

if len(args.saving_path) > 0 and not os.path.isdir(args.saving_path):
    os.mkdir(args.saving_path)
if args.feature_compression is 'bow':
    if len(args.saving_path) > 0 and os.path.isfile(os.path.join(args.saving_path,'dictionary.json')):
        print('Loading dictionary at {} '.format(os.path.join(args.saving_path,'dictionary.json')),end='')
        print(u'\u2713')
        dictionary = json.load(open(os.path.join(args.saving_path,'dictionary.json'),'r'))
    else:
        dictionary = None
    data = FullOneHot(json_path=args.datapath,
                      seed=args.seed,
                      chop_dataset_th=args.dataset_chopped_portion,
                      dictionary=dictionary,
                      min_repetitions=args.min_repetitions)
    data.build_dataset()
    if len(args.saving_path) > 0 and not os.path.isfile(os.path.join(args.saving_path,'dictionary.json')):
        print('Saving dictionary at {} '.format(os.path.join(args.saving_path,'dictionary.json')),end='')
        print(u'\u2713')
        json.dump({'dictionary' : data.dictionary,
                   'max-pages' : data.max_pages},open(os.path.join(args.saving_path,'dictionary.json'),'w'))
    if len(args.saving_path) > 0 and os.path.isfile(os.path.join(args.saving_path,'bow.pkl')):
        print('Loading BagOfWords at {} '.format(os.path.join(args.saving_path,'bow.pkl')),end='')
        print(u'\u2713')
        kmeans_model = joblib.load(open(os.path.join(args.saving_path,'bow.pkl'),'rb'))
    elif len(args.saving_path) > 0:
        train_x = sparse.vstack(data.X)
        print('Running KMeans')
        kmeans_model = KMeans(n_clusters=128,verbose=1).fit(train_x)
        print('Saving BagOfWords at {} '.format(os.path.join(args.saving_path,'bow.pkl')),end='')
        print(u'\u2713')
        joblib.dump(kmeans_model,open(os.path.join(args.saving_path,'bow.pkl'),'wb'))
    else:
        train_x = sparse.vstack(data.X)
        print('Running KMeans')
        kmeans_model = KMeans(n_clusters=128,verbose=args.verbose).fit(train_x)
elif 'raw':
    if len(args.saving_path) > 0 and os.path.isfile(os.path.join(args.saving_path,'dictionary.json')):
        print('Loading dictionary at {} '.format(os.path.join(args.saving_path,'dictionary.json')),end='')
        print(u'\u2713')
        dictionary = json.load(open(os.path.join(args.saving_path,'dictionary.json'),'r'))
    else:
        dictionary = None
    data = RawDataYielder(json_path=args.datapath,
                          seed=args.seed,
                          chop_dataset_th=args.dataset_chopped_portion,
                          dictionary=dictionary,
                          min_repetitions=args.min_repetitions,
                          pad_sequences=False)
    if len(args.saving_path) > 0 and not os.path.isfile(os.path.join(args.saving_path,'dictionary.json')):
        print('Saving dictionary at {} '.format(os.path.join(args.saving_path,'dictionary.json')),end='')
        print(u'\u2713')
        json.dump({'dictionary' : data.dictionary,
                   'max-pages' : data.max_pages,
                   'max-regions' : data.max_regions},open(os.path.join(args.saving_path,'dictionary.json'),'w'))
if 'svm' == args.model:
    if len(args.saving_path) > 0 and os.path.isfile(os.path.join(args.saving_path,'svm.pkl')):
        print('Loading SVM weights at {} '.format(os.path.join(args.saving_path,'svm.pkl')),end='')
        print(u'\u2713')
        svm = joblib.load(open(os.path.join(args.saving_path,'svm.pkl'),'rb'))
    else:
        train_x = []
        train_y = []
        print('Transforming document feature vectors & preparing entities as targets')
        for words_fv,entities in tqdm(zip(data.X,data.Y)):
            clustered_words_fv = kmeans_model.transform(words_fv)
            entities_vector = np.zeros((len(clustered_words_fv),)).astype(int)
            for entity in entities:
                entities_vector[entity] = 1
            train_x += [clustered_words_fv]
            train_y += [entities_vector]
        train_x = np.vstack(train_x)
        train_y = np.hstack(train_y)
        svm = BaggingClassifier(SVC(kernel='rbf', probability=True,verbose=args.verbose != 0), max_samples=0.1, n_estimators=10,verbose=args.verbose != 0,n_jobs=args.nb_jobs)
        # svm = SVC(verbose=args.verbose != 0)
        print('Training SupportVectorMachine')
        print('Input shape : {}'.format(train_x.shape))
        print('Target shape : {}'.format(train_y.shape))
        svm.fit(train_x,train_y)
        if len(args.saving_path) > 0:
            print('Saving SupportVectorMachine at {} '.format(os.path.join(args.saving_path,'svm.pkl')),end='')
            print(u'\u2713')
            joblib.dump(svm,open(os.path.join(args.saving_path,'svm.pkl'),'wb'))
elif 'selfattention' == args.model:
    padding_fn = pad_collate_with_value(data.dictionary['PAD'])
    data_batcher = DataLoader(data,collate_fn=padding_fn,batch_size=args.batch_size,num_workers=args.nb_jobs,drop_last=True) #Change data loader
    selfattention = TransformerTextClassifier(len(data.dictionary.keys()), data.dictionary['PAD'],
                                              n_position=data.max_regions,positional_encoding='bbox',max_pages=data.max_pages)
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.BCELoss()
    results = []
    print('Training self-attention model for Text classification with epoch {}'.format(args.epochs))
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch+1,args.epochs))
        results += [train_epoch(selfattention,data_batcher,epoch,loss_fn)]
        if len(args.saving_path) > 0:
            print('Saving checkpoint at {}'.format(os.path.join(args.saving_path,'self-attention-epoch_{}.pt '.format(epoch))),end='')
            torch.save(selfattention,os.path.join(args.saving_path,'self-attention-epoch_{}.pt'.format(epoch)))
            print(u'\u2713')
#USE EASY ATTENTION MODEL FIRST. ENCODE THE POSITION USING THE OCR INDEX AT FIRST, TRY USING THE COORDINATES LATER ON.
#LAST MODEL USES QUERIES AS INPUTS. EVALUATE ASKING ALL THE QUESTIONS
