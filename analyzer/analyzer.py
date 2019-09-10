"""
Script that performs a performance analysis about the
usage og query expansion through different model model of builfding
word embeddings in a simple informtion retrival system buildt
on the Cranfield dataset. The baseline is a plain tf-idf based
vector space model. Precision, recall and F1 measure will be measured.
The baseline will be compared against expansion query system on the
performance gained recived through the usage of different word embeddings (models),
both trained on the dataset and pretrained.
The hypothesis to test is that query expansion methods through word embeddings
stems in a performance gain both in accuracy and recall for the entire system.
"""
import os

from utils.downloader import Downloader
from utils.inverted_index import *
from utils.query_eval import *
from utils.embedding_factory import *
import pickle


def test_glove():
    with open('data/cleaned_corpus_w2v.pkl','rb') as f:
        data = pickle.load(f)
    g = GloveEmbedding(data)
    g.get_params()
        

if __name__ == '__main__':

    existing_files = set(os.listdir('data'))
    cran_files = {'cranqrel', 'cran.qry', 'cran.all.1400'}

    if not cran_files <= existing_files:
        d = Downloader()
        d.download()
        d.untar()

    i = InvertedIndex()
    i.build_inverted_index()
    i.build_vsm()

    d = DataPolisher()
    cleaned_corpus = d.clean_text(algorithm='w2v')

    w = W2vEmbedding(cleaned_corpus)
    w.start()
    q = QueryBuilder()
    res = q.evaluate_word_embedding_scores(inv_index=i, embeddings=w)
    print(res)
    #k_top = i.query('simple shear flow past a flat plate in an incompressible fluid of small \
#viscosity')
    #print(k_top)
    #q = QueryBuilder()
    #q.number_manager()
    #scores = q.evualuate_baseline(i)
    #print(scores)
    #From here baseline results will be calculated
    #Every query is evaluated and the resulting vector is taken
    #with different k : 5,10,15,20
