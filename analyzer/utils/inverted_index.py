import re
from collections import defaultdict, Counter
from time import time
from math import log
import os
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import *
from numpy import unique

class DataPolisher:
    """General utility class. Provide methods to clean the data and preprocessing it
    """

    def __init__(self, dataset='cran.all.1400',cran=True ,stem=True):
        self.dataset = dataset
        self.cran = cran
        self.stem = stem
        self.cleaned_dataset = None

    def prepare_dataset(self):
        """Preprocess the dataset: lower everything, keep only alphabetical, delete
        stopwords, and store in cleaned_dataset dictionary with entry as {key:cleaned_document}
        The document will be the concat of title and the abstract body.
        After this, document number 471 and 995 will be empty"""
        if not os.path.isfile('data/cleaned_dataset.pkl'):
            r = r'(?sm)\.T\n(.*)\s{0,1}\.\n\.A\n(.*?)\n*\.B\n(.*?)\n*\.W\n(.*) \.'

            with open('data/'+ self.dataset,'r') as f:
                data = f.read()

            self.splitted_data = re.split(r'.I \d+\n',data)
            del data

            self.cleaned_dataset = dict()
            #First element of the list is an empty string
            index = 1

            #Not using eumerate because of the mismatched indexes
            for el in self.splitted_data[1:]:
                matched  = re.match(r,el)
                if matched == None:
                    self.cleaned_dataset[index] = ''
                    index = index + 1
                    continue
                doc = matched.group(1) + matched.group(4)

                #Eliminates all non alpha chars, eliminates stopwords and stem it with porter stemmer?
                cleaned_doc = self.clean_doc(doc)
                self.cleaned_dataset[index] = cleaned_doc
                index = index + 1

            f = open('data/cleaned_dataset.pkl','wb')
            pickle.dump(self.cleaned_dataset,f)
            f.close()

        else:
            self.cleaned_dataset =  pickle.load(open('data/cleaned_dataset.pkl','rb'))

    def clean_supporter(self):
        """Same as prepare_dataset, but returns a list insted of dict and doesn't save anything"""

        r = r'(?sm)\.T\n(.*)\s{0,1}\.\n\.A\n(.*?)\n*\.B\n(.*?)\n*\.W\n(.*) \.'

        with open('data/'+ self.dataset,'r') as f:
            data = f.read()

        splitted_data = re.split(r'.I \d+\n',data)
        del data

        cleaned_corpus = list()
        #First element of the list is an empty string

        index = 1
        #Not using eumerate because of the mismatched indexes
        for el in splitted_data[1:]:
            matched  = re.match(r,el)
            if matched == None:
                cleaned_corpus.append('')
                index = index + 1
                continue
            doc = matched.group(1) + matched.group(4)

            #Eliminates all non alpha chars, eliminates stopwords and stem it with porter stemmer?
            cleaned_doc = self.clean_doc_no_stem(doc)
            cleaned_corpus.append(cleaned_doc)
            index = index + 1
        return cleaned_corpus

    def clean_doc(self, dirty_doc):
        """Utiliy method for prepare_dataset that does the actual job
        """
        sw = set(stopwords.words('english'))
        splitted = dirty_doc.split()

        #Delete all non alpha charachters, eliminates stopwords and stem it
        #using the Porter Stemmer in NLTK
        stemmer = PorterStemmer()
        pol = []

        for word in splitted:
            #Need to take in account the compound hypheneted terms (like self-explanatory)
            new_word = [c if c.isalpha() else ' ' for c in word]
            joined = ''.join(new_word)
            if ' ' not in joined:
                joined = joined.lower()
                if joined in sw:
                    continue
                else:
                    final_word = stemmer.stem(joined)
                    pol.append(final_word)

            else:
                splitted_word = joined.split()
                for term in splitted_word:
                    term = term.lower()
                    if term in sw:
                        continue
                    else:
                        fw = stemmer.stem(term)
                        pol.append(fw)

        document = [s for s in pol if s!='']
        return document

    def clean_doc_no_stem(self, dirty_doc):
        """Same as the previous one wothout stemming"""

        sw = set(stopwords.words('english'))
        splitted = dirty_doc.split()

        pol = []

        for word in splitted:
            #Need to take in account the compound hypheneted terms (like self-explanatory)
            new_word = [c if c.isalpha() else ' ' for c in word]
            joined = ''.join(new_word)
            if ' ' not in joined:
                joined = joined.lower()
                if joined in sw:
                    continue
                else:
                    pol.append(joined)

            else:
                splitted_word = joined.split()
                for term in splitted_word:
                    term = term.lower()
                    if term in sw:
                        continue
                    else:
                        pol.append(term)

        document = [s for s in pol if s!='']
        return document

    def clean_text(self, data_path='data/cran.all.1400',algorithm='w2v'):
        """Read only titles and abstract body oh each document.
        Builds and save a list of iterable depending on the algorithm
        chosen (w2v or else)
        For the embeddings models used here is enough to use w2v. Though
        could be useful for the future.
        """
        if algorithm == 'w2v':
            if not os.path.isfile('data/cleaned_corpus_w2v.pkl'):
                cleaned_corpus_w2v = self.clean_supporter()

                f = open('data/cleaned_corpus_w2v.pkl','wb')
                pickle.dump(cleaned_corpus_w2v,f)
                f.close()
                #Will be list of list of strings as individual words
            with open('data/cleaned_corpus_w2v.pkl','rb')  as f:
                cleaned_corpus_w2v = pickle.load(f)

            return cleaned_corpus_w2v

        elif algorithm == 'else':
            if not os.path.isfile('data/cleaned_corpus_else.pkl'):
                cleaned_corpus_else = self.clean_supporter()
                cleaned_corpus_else = [' '.join(x) for x in cleaned_corpus_else]
                cleaned_corpus_else = ' '.join(cleaned_corpus_else)
                #(now the text is a plain string)

                f = open('data/cleaned_corpus_else.pkl','wb')
                pickle.dump(cleaned_corpus_else,f)
                f.close()
            with open('data/cleaned_corpus_else.pkl','rb')  as f:
                cleaned_corpus_else = pickle.load(f)
                #Will be a string representing the whole corpus
            return cleaned_corpus_else
        else:
            print('Not yet implemented embedding model')

    def get_dataset(self):
        self.prepare_dataset()
        return self.cleaned_dataset


class InvertedIndex:

    def __init__(self):
        print('Cleaning the Cranfield dataset')
        tic = time()
        self.d = DataPolisher()
        self.dataset = self.d.get_dataset()
        toc = time()
        print('Finished cleaning, elapsed time {}'.format(toc-tic))

        self.ii = defaultdict(list)
        self.inverted_index = defaultdict(dict)
        self.word_to_num = defaultdict(list)
        self.num_docs = len(self.dataset)

        #This will hold documents as embedded in terms tf-idf space
        self.vsm = defaultdict(dict)

    def build_inverted_index(self):
        """Two passes per each document: in one I store the term
        frequency in the doc, in the second I build tuples of (#doc,tf_term_doc)
        """

        ind = 0

        for num in range(1,self.num_docs+1):
            tf_dict = defaultdict(int)

            for word in self.dataset[num]:
                if word not in self.word_to_num:
                    self.word_to_num[word].append(ind)
                    ind += 1

                tf_dict[word] += 1

            for word in unique(self.dataset[num]):
                self.ii[word].append((num,tf_dict[word]))

        #Calculating the idf weight for each word
        #and multiply it by the term frequency

        #This dictionary will be used during the query evaluation phase
        #to detect the terms with the lowest idf in order to expanded
        self.idf_dict = dict()

        for word in self.ii:
            df_word = len(self.ii[word])
            idf_word = log(self.num_docs/df_word)
            self.idf_dict[word] = idf_word

            for doc,tf in self.ii[word]:
                self.inverted_index[word][doc] = tf*idf_word

            self.word_to_num[word].append(idf_word)

    def query(self, query ,k=10):
        """Given a query returns the top k documents scoring"""

        #Query returned as a cleaned list
        if not type(query) == list:
            cleaned_query = self.d.clean_doc(query)
        else:
            cleaned_query = query
        #Filter the unknown ones:
        filtered = [term for term in cleaned_query if term in self.word_to_num]

        if filtered == []:
            print('Empty query!')
            return None

        #Calculcates query tf-idf compressed vector
        counted_words = Counter(filtered)
        query_vec = {word:counted_words[word]*self.word_to_num[word][1] for word in counted_words}

        #Compute it against all documents tf-idf vectors and take the top scoring k
        scores = []
        for doc in self.vsm:
            s = self.compute_score(query_vec,self.vsm[doc])
            scores.append((doc,s))
        results = sorted(scores,key=lambda x:x[1],reverse=True)
        return results[:k]


    def compute_score(self, vec1, vec2):
        """Fastened intersection and multiplication thanks to pyhon set
        """

        inter = set(vec1).intersection(vec2)

        if len(inter) == 0:
            return 0

        score = 0

        for w in inter:
            score += vec1[w] * vec2[w]

        return score

    def build_vsm(self):
        """Self vsm will be formed by 1398 items, since 2 docuemnts are empty.
        Nevertheless the indexing will be fine.
        """
        for doc in self.dataset:
            for word in self.dataset[doc]:
                self.vsm[doc][word] = self.inverted_index[word][doc]
