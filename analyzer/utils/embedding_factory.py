"""This class  will be an interface for all the Embeddings encapsulating
classes that will be created later on.
"""
from collections import defaultdict, Counter
import json
import os
from time import time

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
import numpy as np
from sklearn.utils.extmath import randomized_svd


class EmbeddingFactory:

    def __init__(self, corpus):
        self.corpus = corpus

    def get_params(self, param_file):
        raise NotImplementeError()

    def create_embeddings(self):
        raise NotImplementeError()

    def get_most_similar(self, word , k):
        raise NotImplementedError()

    def create_vocabulary():
        raise NotImplementeError()

    def start(self):
        self.get_params()
        self.create_vocabulary()
        self.create_embeddings()


class W2vEmbedding(EmbeddingFactory):

    def get_params(self, param_file='params/params_w2v.json'):
        with open(param_file, 'r') as f:
            params = json.load(f)
        self.dimensionality, self.window, self.iter = params['dimensionality'], params['window'], params['iter']

    def create_embeddings(self):
        if not os.path.isfile('embeddings/w2v_'+str(self.dimensionality)+'_'+str(self.window)+'_'+str(self.iter)+'.kw'):
            print('Start training W2V embeddings:')
            tic = time()
            model = Word2Vec(sentences=self.corpus, size=self.dimensionality, window=self.window,
            min_count=1, sg=1, iter=10)
            toc = time()
            print('Finished training. Elapsed time --> {}'.format(toc-tic))
            path = 'embeddings/w2v_'+str(self.dimensionality)+'_'+str(self.window)+'_'+str(self.iter)+'.kw'
            self.embeddings = model.wv
            model.wv.save(path)
        else:
            self.embeddings = KeyedVectors.load('embeddings/w2v_'+str(self.dimensionality)+'_'
            +str(self.window)+'_'+str(self.iter)+'.kw', mmap='r')
        #string that will be taken by the query class to take care of saving process
        self.save_file = 'results/w2v_'+str(self.dimensionality)+'_'+str(self.window)+'_'+str(self.iter)+'.pkl'

    def create_vocabulary(self):
        self.vocab = {word for sentence in self.corpus for word in sentence}

    def get_most_similar(self, word, k=5):
        most_similar = self.embeddings.most_similar(word, topn=k)
        #most_similar = [x[i][0] for i,x in enumerate(most_similar)]
        to_return = []
        for i in range(len(most_similar)):
            to_return.append(most_similar[i][0])

        return to_return

class LsaEmbedding(EmbeddingFactory):

    def get_params(self, param_file='params/params_lsa.json'):
        with open(param_file, 'r') as f:
            params = json.load(f)
        self.dimensionality, self.window = params['dimensionality'], params['window']

    def create_embeddings(self):
        """This methid will create a cooccurrance matrix with self.corpus
        at the end will be applied a singular value decomposition to obtain the embeddings
        it needs a mapping beetween embeddings and words back and forth.
        """
        window = self.window
        #No need to use the vocabulary size parameter
        #vocabulary_size = self.vocabulary_size
        cooccurance_count = defaultdict(Counter)
        for sentence in self.numbered_corpus:
            for idx, center_word_id in enumerate(sentence):
                for i in range(max(idx - window - 1, 0), min(idx + window + 1, len(sentence))):
                    cooccurance_count[center_word_id][sentence[i]] += 1
                cooccurance_count[center_word_id][center_word_id] -= 1
        self.cocount = cooccurance_count
        #Here we build the matrix and perform the singular value decomposition according
        #to the dimension specified by the user
        Nij = np.zeros([self.vocab_size, self.vocab_size])
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                Nij[i,j] += self.cocount[i][j]
        self.n = Nij



        U, Sigma, VT = randomized_svd(Nij, n_components=self.dimensionality, n_iter=5, random_state=None)
        unnormalized_embeddings = U * Sigma
        #normalization of embeddings to have faster cosine distance measure
        norms = np.linalg.norm(unnormalized_embeddings, axis=1)
        self.embeddings = unnormalized_embeddings / norms[:,None]

    def create_vocabulary(self):
        #Different from the previous one.
        corpus_listed = [word for sentence in self.corpus for word in sentence]
        self.vocab = dict()
        index = 0
        for word in corpus_listed:
            if not word in self.vocab:
                self.vocab[word] = index
                index += 1
        self.retro_vocab = {v:k for k,v in self.vocab.items()}
        #Need to create a numbered corpus to not have to deal with numbers everytime
        self.numbered_corpus = [[self.vocab[word] for word in sentence] for sentence in self.corpus]
        self.vocab_size = len(self.vocab)

    def get_most_similar(self, word, k=2):
        #to be implemented
        index = self.vocab[word]
        #most_similar = [x[i][0] for i,x in enumerate(most_similar)]
        word_indexed = self.embeddings[index]
        scores = self.embeddings.dot(word_indexed)
        #stupidata per evitare che mi venga restituita la stessa parola
        scores[index] = -10000
        k_index = np.argsort(scores)[-k:][::-1]

        to_return = []
        for i in range(len(k_index)):
            to_return.append(self.retro_vocab[k_index[i]])

        return to_return

class HpcaEmbedding(EmbeddingFactory):

    def get_params(self, param_file='params/params_hpca.json'):
        with open(param_file, 'r') as f:
            params = json.load(f)
        self.dimensionality, self.window = params['dimensionality'], params['window']

    def create_embeddings(self):
        """This methid will create a cooccurrance matrix with self.corpus
        at the end will be applied a singular value decomposition to obtain the embeddings
        it needs a mapping beetween embeddings and words back and forth.
        """
        window = self.window
        #No need to use the vocabulary size parameter
        #vocabulary_size = self.vocabulary_size
        cooccurance_count = defaultdict(Counter)
        for sentence in self.numbered_corpus:
            for idx, center_word_id in enumerate(sentence):
                for i in range(max(idx - window - 1, 0), min(idx + window + 1, len(sentence))):
                    cooccurance_count[center_word_id][sentence[i]] += 1
                cooccurance_count[center_word_id][center_word_id] -= 1
        self.cocount = cooccurance_count
        #Here we build the matrix and perform the singular value decomposition according
        #to the dimension specified by the user
        Nij = np.zeros([self.vocab_size, self.vocab_size])
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                Nij[i,j] += self.cocount[i][j]
        self.n = Nij

        summed_rows = np.sum(Nij,axis=1)
        #Here we have to normalize every row to be interpreted as a
        #distribution and then take the square root.
        Nij = np.sqrt(Nij / summed_rows[:,None])

        U, Sigma, VT = randomized_svd(Nij, n_components=self.dimensionality, n_iter=5, random_state=None)
        unnormalized_embeddings = U * Sigma
        #normalization of embeddings to have faster cosine distance measure
        norms = np.linalg.norm(unnormalized_embeddings, axis=1)
        self.embeddings = unnormalized_embeddings / norms[:,None]

    def create_vocabulary(self):
        #Different from the previous one.
        corpus_listed = [word for sentence in self.corpus for word in sentence]
        self.vocab = dict()
        index = 0
        for word in corpus_listed:
            if not word in self.vocab:
                self.vocab[word] = index
                index += 1
        self.retro_vocab = {v:k for k,v in self.vocab.items()}
        #Need to create a numbered corpus to not have to deal with numbers everytime
        self.numbered_corpus = [[self.vocab[word] for word in sentence] for sentence in self.corpus]
        self.vocab_size = len(self.vocab)

    def get_most_similar(self, word, k=2):
        #to be implemented
        index = self.vocab[word]
        #most_similar = [x[i][0] for i,x in enumerate(most_similar)]
        word_indexed = self.embeddings[index]
        scores = self.embeddings.dot(word_indexed)
        #stupidata per evitare che mi venga restituita la stessa parola
        scores[index] = -10000
        k_index = np.argsort(scores)[-k:][::-1]

        to_return = []
        for i in range(len(k_index)):
            to_return.append(self.retro_vocab[k_index[i]])

        return to_return

class GloveEmbedding(EmbeddingFactory):

    def get_params(self, param_file='params/params_glove.json'):
        """This method will also generate a file from the corpus as a
        .txt file in order to make the backend glove algorithm usable
        """
        with open(param_file, 'r') as f:
            params = json.load(f)
        self.dimensionality, self.window, self.iter = params['dimensionality'], params['window'], params['iter']
        stringed_corpus = ' '.join([word for sentence in self.corpus for word in sentence])
        if not os.path.isfile('data/stringed_corpus.txt'):
            with open ('data/stringed_corpus.txt','w') as f:
                f.write(stringed_corpus)
        corpus_path = os.path.abspath('.') + '/data/stringed_corpus.txt'
        #Parameter injection in the demo.sh file
        print('Modifing sh file')
        file = 'utils/glove/demo.sh'
        with open(file, 'r') as f:
            text = f.readlines()
        with open(file, 'w') as f:
            for i, line in enumerate(text,1):
                if i==8:
                    f.writelines('CORPUS={}\n'.format(corpus_path))
                else:
                    f.writelines(line)
        with open(file, 'r') as f:
            text = f.readlines()
        with open(file, 'w') as f:
            for i, line in enumerate(text,1):
                if i==17:
                    f.writelines('VECTOR_SIZE={}\n'.format(self.dimensionality))
                else:
                    f.writelines(line)
        with open(file, 'r') as f:
            text = f.readlines()
        with open(file, 'w') as f:
            for i, line in enumerate(text,1):
                if i==18:
                    f.writelines('MAX_ITER={}\n'.format(self.iter))
                else:
                    f.writelines(line)
        with open(file, 'r') as f:
            text = f.readlines()
        with open(file, 'w') as f:
            for i, line in enumerate(text,1):
                if i==19:
                    f.writelines('WINDOW_SIZE={}\n'.format(self.window))
                else:
                    f.writelines(line)
