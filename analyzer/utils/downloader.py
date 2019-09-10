import os
import tarfile

import requests

class Downloader:
    """Class to manage download of the cranfield dataset
    here and to save them in the proper directory
    """
    def __init__(self,url='http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz',folder='data'):
        self.url = url
        self.folder = folder

    def download(self):
        req = requests.get(self.url)
        with open('data/cranfield.tar.gz','wb') as f:
            f.write(req.content)

    def untar(self):
        """Untar the tar file and delete the tar"""
        tf = tarfile.open('data/cranfield.tar.gz')
        tf.extractall('data')
        if os.path.exists('data/cranfield.tar.gz'):
            os.remove('data/cranfield.tar.gz')
