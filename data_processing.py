import json
import numpy as np
from gensim.models import Word2Vec
import random
from scipy import sparse
from tqdm import tqdm
from joblib import Parallel,delayed

class FullOneHot():
    def __init__(self,json_path,
                      dictionary=None,
                      chop_dataset_th=0.,
                      seed=7,
                      min_repetitions=20,
                      nb_jobs=6):
        """
        Encodes dataset as multi-one-hot vectors. The Feature vectors include the positional information of the word.
        Kmeans model runs these sparse content+position vectors.
        json_path : path to json data
        dictionary : dictionary object. If None, the dictionary is computed
        chop_dataset_th : crop dataset using this threshold
        min_repetitions : min repetitions to belong to the dictionary
        nb_jobs : number of parallel workers to use when generating feature vectors
        """
        self.data = json.load(open(json_path,'r'))
        random.seed(seed)
        random.shuffle(self.data)
        self.data = self.data[int(chop_dataset_th*len(self.data))::]
        self.nb_jobs = nb_jobs
        if dictionary == None:
            print('Building dictionary')
            self.build_dictionary(min_repetitions=min_repetitions)
        else:
            self.dictionary = dictionary['dictionary']
            self.dict_len = len(self.dictionary)
            self.max_pages = dictionary['max-pages']
    def build_dictionary(self,min_repetitions=20):
        dictionary = []
        i = 0
        max_pages = 0
        for document in self.data:
            for word_json in document['words']:
                dictionary += [str(word_json['value'])]
                if int(word_json['region']['page']) > max_pages:
                    max_pages = int(word_json['region']['page'])
            i += 1
            if i == 1000:
                dictionary = np.unique(dictionary).tolist()
                i = 0
        dictionary = np.unique(dictionary).tolist()
        dict_count = {word : 0 for word in dictionary}
        for document in self.data:
            for word_json in document['words']:
                dict_count[word_json['value']] += 1
        nb_words = 0.
        nb_othered_words = 0.
        for word in dictionary:
            nb_words += dict_count[word]
            if dict_count[word] < min_repetitions:
                nb_othered_words += dict_count[word]
                del dict_count[word]
        othered_nb_words = len(dictionary) - len(dict_count.keys())
        dictionary = list(dict_count.keys()) + ['OTHER']
        self.dictionary = {word : i for i,word in enumerate(dictionary)}
        self.dict_len = len(dictionary)
        self.max_pages = max_pages
        print('Dictionary size: {}\nMinimum number of repetition: {}'.format(self.dict_len,min_repetitions))
        print('Excluded words by number of repetition (%): {}'.format(np.round(nb_othered_words/nb_words,2)))
        print('Removed number of words: {}'.format(othered_nb_words))
    def build_dataset(self,document_shape=(1245,862)):
        """Encodes the data as multi one hot vector"""
        dictionary_words = list(self.dictionary.keys())
        def vectorize_document(document):
            words_fv = []
            for word_json in document['words']:
                # if self.
                word_fv = np.zeros((self.dict_len,))
                if word_json['value'] in dictionary_words:
                    word_fv[self.dictionary[word_json['value']]] = 1
                else:
                    word_fv[self.dictionary['OTHER']] = 1
                position_fv = [np.zeros((document_shape[0],),dtype=int),np.zeros((document_shape[1],),dtype=int)]
                page_fv = np.zeros((self.max_pages,))
                centre_y = np.abs(word_json['region']['top']+word_json['region']['height'] / 2.)
                centre_x = np.abs(word_json['region']['left']+np.abs(word_json['region']['width']) / 2.)
                if centre_y > 1:
                    centre_y = 1
                if centre_x > 1:
                    centre_x = 1
                position_fv[0][int((document_shape[0]-1)*centre_y)] = 1
                position_fv[1][int((document_shape[1]-1)*centre_x)] = 1
                page_fv[int(word_json['region']['page'])-1] = 1
                words_fv += [np.hstack([word_fv] + [page_fv] + position_fv).astype(int)]
            return [sparse.csr_matrix(np.asarray(words_fv)),[np.hstack([x['indices'] for x in document['entities']])]]
        print('Building feature vectors. n_jobs={}'.format(self.nb_jobs))
        XY = Parallel(n_jobs=self.nb_jobs,prefer='threads')(delayed(vectorize_document)(document) for document in tqdm(self.data,desc='Processed documents'))
        self.X,self.Y = [],[]
        for x,y in XY:
            self.X += [x]
            self.Y += y

class RawDataYielder():
    def __init__(self,json_path,
                      dictionary=None,
                      chop_dataset_th=0.,
                      # val_split=0.1,
                      seed=7,
                      min_repetitions=10,
                      nb_jobs=6,
                      positional_grid_shape=(1245,862),
                      pad_sequences=False):
        """Yield raw words (dictionary index only) with their associated positions
           dictionary : dictionary object, if None dictionary is computed
           chop_dataset_th : crop dataset using this threshold
           min_repetitions : min repetitions to belong to the dictionary
           nb_jobs : number of parallel workers to use when generating feature vectors
           positional_grid_shape : grid to use to discretize positions of bboxes
           pad_sequences: wheter to pad sequence using max number of regions in dataset"""
        self.data = json.load(open(json_path,'r'))
        random.seed(seed)
        random.shuffle(self.data)
        self.data = self.data[int(chop_dataset_th*len(self.data))::]
        self.nb_jobs = nb_jobs
        self.positional_grid_shape = positional_grid_shape
        self.pad_sequences = pad_sequences
        if dictionary == None:
            print('Building dictionary')
            self.build_dictionary(min_repetitions=min_repetitions)
        else:
            self.dictionary = dictionary['dictionary']
            self.dict_len = len(self.dictionary)
            self.max_pages = dictionary['max-pages']
            self.max_regions = dictionary['max-regions']
            self.dictionary_words = list(self.dictionary.keys())
    def build_dictionary(self,min_repetitions=20):
        dictionary = []
        i = 0
        max_pages = 0
        max_regions = 0
        for document in self.data:
            for word_json in document['words']:
                dictionary += [str(word_json['value'])]
                if int(word_json['region']['page']) > max_pages:
                    max_pages = int(word_json['region']['page'])
            if len(document['words']) > max_regions:
                max_regions = len(document['words'])
            i += 1
            if i == 1000:
                dictionary = np.unique(dictionary).tolist()
                i = 0
        dictionary = np.unique(dictionary).tolist()
        dict_count = {word : 0 for word in dictionary}
        for document in self.data:
            for word_json in document['words']:
                dict_count[word_json['value']] += 1
        nb_words = 0.
        nb_othered_words = 0.
        for word in dictionary:
            nb_words += dict_count[word]
            if dict_count[word] < min_repetitions:
                nb_othered_words += dict_count[word]
                del dict_count[word]
        othered_nb_words = len(dictionary) - len(dict_count.keys())
        dictionary = list(dict_count.keys()) + ['OTHER'] + ['PAD']
        self.dictionary = {word : i for i,word in enumerate(dictionary)}
        self.dictionary_words = dictionary
        self.dict_len = len(dictionary)
        self.max_pages = max_pages
        self.max_regions = max_regions
        print('Dictionary size: {}\nMinimum number of repetition: {}'.format(self.dict_len,min_repetitions))
        print('Excluded words by number of repetition (%): {}'.format(np.round(nb_othered_words/nb_words,2)))
        print('Removed number of words: {}'.format(othered_nb_words))
    def __getitem__(self,index):
        document_data = self.data[index]
        if self.pad_sequences:
            length = self.max_regions
        else:
            length = len(document_data['words'])
        x = int(self.dictionary['PAD'])*np.ones((length,)).astype(int)
        y = np.zeros((length,)).astype(int)
        positions_x = np.zeros((length,self.positional_grid_shape[1])).astype(int)
        positions_y = np.zeros((length,self.positional_grid_shape[0])).astype(int)
        positions_page = np.zeros((length,self.max_pages)).astype(int)
        for i,word_json in enumerate(document_data['words']):
            if word_json['value'] in self.dictionary_words:
                x[i] = self.dictionary[word_json['value']]
            else:
                x[i] = self.dictionary['OTHER']
            centre_y = np.abs(word_json['region']['top']+word_json['region']['height'] / 2.)
            centre_x = np.abs(word_json['region']['left']+np.abs(word_json['region']['width']) / 2.)
            if centre_y > 1:
                centre_y = 1
            if centre_x > 1:
                centre_x = 1
            positions_x[i,int(self.positional_grid_shape[1]*centre_x)-1] = 1
            positions_y[i,int(self.positional_grid_shape[0]*centre_y)-1] = 1
            positions_page[i,int(word_json['region']['page'])-1] = 1

            # positions += [(centre_x,centre_y,int(word_json['region']['page']))]
        # x = [self.dictionary[word_json['value']] for word_json in document_data['words']]
        for entity in document_data['entities']:
            for e in entity['indices']:
                y[int(e)] = 1
        # y = np.hstack([x['indices'] for x in document_data['entities']]).astype(int)
        return x,y,positions_x,positions_y,positions_page
    def __len__(self):
        return len(self.data)
