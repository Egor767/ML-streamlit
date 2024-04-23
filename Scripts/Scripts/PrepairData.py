import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import scipy.stats as stats
from pandas.api.types import is_string_dtype

class PrepairData():
    def __init__(self):
        self.data = None
        self.encoder = None

    def init_data(self, data):
        self.data = data

    def clean_trash(self):
        self.data = self.data.drop_duplicates()
        self.data = self.data.dropna()

    def delete_predicts(self, predicts):
        self.data = self.data.drop(predicts, axis = 1)

    def standardization(self, X_test):
        return StandardScaler().fit_transform(X_test)
    
    def data_show(self):
        st.write(self.data.head(10))

    def data_return(self):
        return self.data
    
    def get_columns(self):
        return self.data.columns.to_list()
    
    def data_encoder(self, columns):
        self.columns = columns
        for i in self.data.columns.tolist():
            if is_string_dtype(self.data[i]):
                self.columns.append(i)

        label_encoder = preprocessing.LabelEncoder()

        for i in self.columns:
            self.data[i] = label_encoder.fit_transform(self.data[i])

    def split(self, predict, test_size = 0.3):
        y = self.data[predict].to_numpy()
        X = self.standardization(self.data.drop([predict], axis=1).to_numpy())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
        return X_train, X_test, y_train, y_test

    