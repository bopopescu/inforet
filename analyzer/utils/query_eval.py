"""Utility class to create a numpy dump of the relevant and non
relevant items in the collection.
The file cran.qry contains (bad ordered) 225 queryes, and cranqrel
contains relevance judgement (here the cutoff is decided to be 3).
"""
from collections import defaultdict
import pickle
import re
import os
from itertools import product

import matplotlib.pyplot as plt
from nltk.stem.porter import *
import pandas as pd
import seaborn as sn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from time import time

from utils.inverted_index import DataPolisher

class QueryBuilder:
    """This class will have some repeated code, nned to be refactored
    """

    def __init__(self, path_qrel='data/cranqrel', path_qry='data/cran.qry'):
        self.path_qrel = path_qrel
        self.path_qry = path_qry
        self.polisher = DataPolisher()

    def number_manager(self):
        """Takes the cranqrel file and transform it in a dictionary
        of binary list representing rel/non relevance of documents gven
        the queries. It is made up and saved as a dictionary pickle dump
        Dictionary will have the shape {query: '...', rel_list: [...]}.
        """
        if not os.path.isfile('data/ground_truth_dict.pkl'):
            with open(self.path_qrel,'r') as a, open(self.path_qry,'r') as b:
                rel = a.readlines()
                qry = b.read()

            rel_iter = iter(rel)
            query_re = r'(?sm)\.W\n(.+?)\s{0,1}\.'
            query_iter = re.finditer(query_re,qry)

            rel = next(rel_iter).split()
            current = 1

            to_save = dict()

            for q in query_iter:
                cleaned_q = self.polisher.clean_doc(q.group(1))
                cleaned_q_no_stem = self.polisher.clean_doc_no_stem(q.group(1))
                el = dict()
                el['query'] = cleaned_q
                el['query_no_stem'] = cleaned_q_no_stem
                relevants = []
                while int(rel[0]) == current:
                    if int(rel[2]) <= 3:
                        relevants.append(int(rel[1]))
                    try :
                        rel = next(rel_iter).split()
                    except:
                        break
                to_fill = [0]*1400
                for i in relevants:
                    #List are indexed from 0 to 1399, remember to cope up with this later on
                    to_fill[i-1] = 1
                el['rel_list'] = to_fill
                to_save[current] = el
                current += 1
            stacked_truth = []
            for i in range(1,226):
                stacked_truth.extend(to_save[i]['rel_list'])
            to_save['stacked_truth'] = stacked_truth
            f = open('data/ground_truth_dict.pkl','wb')
            pickle.dump(to_save,f)
            f.close()
        ground_truth = pickle.load(open('data/ground_truth_dict.pkl','rb'))
        return ground_truth

    def evualuate_baseline(self, inv_index, k=[5,10,15,20,25,30,35,40,45,50]):
        #Evaluate precision, recall, F1 measure, ROC curve and confusion matrix
        #with k setted as wanted by the user
        #Save and returns a tuple with precision, recall, F1, confusion matrix and ROC curve
        if not os.path.isfile('results/baseline_results.pkl'):
            ground_truth = self.number_manager()
            stacked_truth = ground_truth['stacked_truth']
            num_queries = 225
            results = dict()
            for number in k:
                stacked_pred = []
                for i in range(1,num_queries+1):
                    query = ground_truth[i]['query']
                    relevants = inv_index.query(query,number)
                    rel_indexes = [x[0] for x in relevants]
                    relevants_zeros = [0]*1400
                    for ind in rel_indexes:
                        relevants_zeros[ind-1] = 1
                    stacked_pred.extend(relevants_zeros)
                precision = precision_score(stacked_truth,stacked_pred)
                recall = recall_score(stacked_truth,stacked_pred)
                f1 = f1_score(stacked_truth,stacked_pred)
                cm = confusion_matrix(stacked_truth,stacked_pred)
                results[number] = (precision,recall,f1,cm)
            f = open('results/baseline_results.pkl','wb')
            pickle.dump(results,f)
            f.close()
        results = pickle.load(open('results/baseline_results.pkl','rb'))
        return results

    def evaluate_word_embedding_scores(self,  embeddings,inv_index, k=[5,10,15,20,25,30,35,40,45,50], exp_number=3):
        #METTERE A POSTO STA COSA DEL K IN MANIERA TALE DA NON FARE TUTTO IL LAVORO AD OGNI ITERAZIONE
        #Evaluate precision, recall, F1 measure, ROC curve and confusion matrix
        #with k setted as wanted by the user
        #Save and returns a tuple with precision, recall, F1, confusion matrix and ROC curve.
        #Everything will be expanded thanks to word embeddings
        save_file = embeddings.save_file
        if not os.path.isfile(save_file):
            #Qui la logica deve essere cambiata completamente
            #Ho bisogno anche di un dizionario che contenga le query senza
            #lo stemming ma che poi applichi lo stemming.
            #Ho modificato la funzione nuber manager per avere una cosa in più nel dizionario
            ground_truth = self.number_manager()
            stacked_truth = ground_truth['stacked_truth']
            num_queries = 225
            results = defaultdict(list)

            for i in range(1,num_queries+1):
                """Cambio di logica rispetto a prima: per ogni diversa query
                genero la sua espansione in accordo con l'implementazione precedente.
                Il metodo get_relevant_indexes deve restituire una lista composta da
                coppie del tipo (k,most_relevant_index) dove most relevant index  è
                ancora una lista di dimensine k.
                Costruirò il dizionario dei risultati incrementalmente, ovvero per ogni i,
                appenderò al k la sua lista man mano che viene generata. Results dovrà
                quindi essere un defaultdict(list)
                k deve essere sempre una lista!
                """
                print('Query number {}'.format(i))
                query = ground_truth[i]['query_no_stem']
                tic = time()
                #rel_indexed adesso è un dizionario
                rel_indexes = self.get_relevant_indexes(query, inv_index, embeddings, exp_number, k)
                toc = time()
                print('Elapsed time for query number {} ---> {}'.format(i,toc-tic))
                for key,l in rel_indexes.items():
                    print(key)
                    print(l)
                    relevant_zeros = [0]*1400
                    for ind in l:
                        relevant_zeros[ind-1] = 1
                    results[key].extend(relevant_zeros)
            results_returned = dict()
            for key in k:
                stacked_pred = results[key]
                precision = precision_score(stacked_truth,stacked_pred)
                recall = recall_score(stacked_truth,stacked_pred)
                f1 = f1_score(stacked_truth,stacked_pred)
                cm = confusion_matrix(stacked_truth,stacked_pred)
                results_returned[key] = (precision, recall, f1, cm)
            f = open(save_file,'wb')
            pickle.dump(results_returned,f)
            f.close()
        results = pickle.load(open(save_file,'rb'))
        return results

    def get_relevant_indexes(self, query, inv_index, embeddings, ex_number, top_number):
        #just to debug later on
        #RICORDATI DI STEMMARE QUI ALTRIMENTI È TUTTO UN CASINO
        #print(query)
        s = PorterStemmer()
        #NON È DETTO CHE LE PAROLE CHE CI SONO NEL CORPUS CI SIANO ANCHE NELLE QUERY
        #expanded_list = [embeddings.get_most_similar(word, ex_number) for word in query]
        expanded_list = []
        for word in query:
            if word in embeddings.vocab:
                ms = embeddings.get_most_similar(word, ex_number)
                expanded_list.append(ms)
            else:
                expanded_list.append([word])
        stemmed_list = [[s.stem(x) for x in y] for y in expanded_list]
        combined_list = list(product(*stemmed_list))
        combined_list = [list(x) for x in combined_list]
        #Retrive for each top number, sort them by score and returns list of most relevant ones
        to_return = []

        """for tern in combined_list:
            ms = inv_index.query(tern, top_number)
            to_return.extend(ms)
        sorted_results = sorted(to_return, key=lambda x:x[1], reverse=True)
        retrived = []
        for el in sorted_results:
            if len(retrived) < 5 and not el[0] in retrived:
                retrived.append(el[0])
                print(retrived)"""
        k_max = max(top_number)
        support_dict=defaultdict(list)
        for tern in combined_list:
            ms = inv_index.query(tern, k_max)
            for k in top_number:
                support_dict[k].extend(ms[:k])
        #Questa lista contiene i k_max migliori per questa 'terna'
        #Adesso devo ordinare in maniera ascnedente ognuna delle enteate del dizionario secondo la seconda chiave
        to_return_dict = dict()
        for k,l in support_dict.items():
            ordered = sorted(l,key=lambda x:x[1],reverse=True)
            retrived = []
            for el in ordered:
                if len(retrived)<k and not el[0] in retrived:
                    print(el)
                    retrived.append(el[0])
            to_return_dict[k]=retrived
            print('Finished retriving the {} best'.format(k))
        return to_return_dict




    def draw_conf_matrix_and_save(self,cf_matrix,file_name):
        #per qualche motivo pyplot non mi centra bene i numeri, da risolvere prima di fare sta cosa
        pass

    def draw_roc_curve_and_save(self,cf_matrix,file_name):
        #per qualche motivo pyplot non mi centra bene i numeri, da risolvere prima di fare sta cosa
        pass
