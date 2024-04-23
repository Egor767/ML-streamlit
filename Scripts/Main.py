import streamlit as st
import pandas as pd
import numpy as np

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import MultiTaskLasso, Lasso

import sys

sys.path.append('..')
from Scripts.ModelManager import ModelManager
from Scripts.Visual import Visual
from Scripts.PrepairData import PrepairData

visual = Visual()
prepair = PrepairData()
model_manager = ModelManager({
    'GaussianClassifier' : [GaussianProcessClassifier(), {
        'kernel': 1.0 * RBF(1.0),
        'random_state': 0}],
    'MultiTaskLasso' : [MultiTaskLasso(), {
        'alpha': np.arange(0.1, 1, 0.1)}]
    })

def loader():
    data = st.file_uploader("Загрузите датасет в формате *.csv", type="csv")
    data = pd.read_csv(data, sep = ',')
    prepair.init_data(data)

def convertor(data):
    return data.to_csv().encode('utf-8')

def model_choiser():
    st.markdown("Датасет:")
    prepair.data_show()
         
    choices = ["Классификация", "Регрессия"]   
    choice = st.selectbox("Выберите вид задачи", choices, index=None)
               
    if choice is not None:
        if choice == choices[0]:
            model_manager.model_init(['GaussianClassifier', 'classification'])
        else:
            model_manager.model_init(['MultiTaskLasso', 'regression'])

        prepairer()

def predictor():
    choice = st.text_input("Введите целевой признак")
    check = st.checkbox('Обучить модель')
    
    if (choice != None and check):
        X_train, X_test, y_train, y_test = prepair.split(choice)
        model_manager.fit(X_train, y_train)
        result, report = model_manager.predict(X_test, y_test)
        st.write("final")
        st.json(report)
        st.write(result)

def deleter():
    check = st.checkbox('Нужно ли удалять некоторые столбцы?')
    prepair.data_show()
    if (check):   
        columns= st.multiselect('Выбор предикторов для удаления', prepair.get_columns(),
            max_selections=len(prepair.get_columns()))

        check2 = st.checkbox('Удалить')
        if (len(columns)!= 0 and check2):
            prepair.delete_predicts(columns)
            prepair.data_show()

def encoder():
    check = st.checkbox('Есть ли в датасете столбцы-предикторы с нечисловыми значениями?')
    prepair.data_show()
    if (check):   
        columns = st.multiselect('Выбор предиктов с нечисловыми значениями', prepair.get_columns(),
            max_selections=len(prepair.get_columns()))
            
        check2 = st.checkbox('Закодировать выбранные признаки')
        if (len(columns)!= 0 and check2):
            prepair.data_encoder(columns)
            prepair.data_show()

def cleaner():
    st.write("Пропуски автоматически удалены")
    prepair.clean_trash()
    prepair.data_show()

def prepairer():
    choices = ["Да", "Нет"]   
    choice = st.selectbox("Был ли ваш датасет подготовлен для решения задачи машинного обучения?", choices, index=None)

    if choice is not None:
        if choice == choices[0]:
            predictor()

        else:
            cleaner()
            deleter()
            encoder()
            check2 = st.checkbox('Перейти к получению результатов')  
            if (check2):
                predictor()
                
st.title("Обработка данных и получение результата")
loader()
check = st.checkbox('Переход к моделям')
if(check):
    model_choiser()

