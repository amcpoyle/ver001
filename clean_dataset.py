import numpy as np
import pandas as pd

class Clean(object):

    def __init__(self, verbose=0):
        self.verbose = verbose

    def encode_labels(self, column, dataframe):
        self.column = column
        self.dataframe = dataframe

    def preprocess(self, path):
        self.path = path
        df = pd.read_csv(self.path)
        id_dict = encode_labels()
