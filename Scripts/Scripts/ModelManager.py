import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import MultiTaskLasso, ElasticNet
from sklearn.linear_model import ElasticNet

class ModelManager():
    def __init__(self, models) -> None:
        self.models = models
        
    def model_init(self, actual_model):
        self.actual_model = self.models[actual_model[0]][0]
        self.model_params = self.models[actual_model[0]][1]
        if actual_model[1] == 'classification':
            self.report = self.report_class
        else:
            self.report = self.report_reg

    def report_class(self, y_test, y_pred):
        return {'Classification report': classification_report(y_test, y_pred), 'F1 score': f1_score(y_test, y_pred)}

    def report_reg(self, y_test, y_pred):
        MAE = mean_absolute_error(y_test, y_pred)
        MAPE = mean_absolute_percentage_error(y_test, y_pred)
        MSE = mean_squared_error(y_test, y_pred)
        R2 = r2_score(y_test, y_pred)
        return {'MAE': MAE, 'MSE': MSE, 'MAPE': MAPE, 'R2': R2}
    
    def fit(self, X_test, y_test):
        st.write("fit 1")
        st.write(X_test, y_test)
        result = ElasticNet(self.model_params).fit(X_test, y_test)  
        st.write("fit 2")
        self.actual_model = result

    def predict(self, X_test, y_test):
        st.write("predict 1")
        st.write(self.actual_model)
        y_pred = self.actual_model.predict(X_test)
        st.write("predict 2")
        return y_pred, self.report(X_test, y_pred)
    
    