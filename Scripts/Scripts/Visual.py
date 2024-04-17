import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

class Visual():
    def __init__(self):
        self.data = None
        self.params = None
    
    def init_data(self, data):
        self.data = data
        self.params = data.columns.to_list()

    def displot(self, column):
        st.title("Диаграмма распределения:")    
        plt.figure(figsize = (12, 6))
        sns.displot(data = self.data, x = column, kind = "kde")
        st.pyplot(plt)

    def boxplot(self, column):
        st.title("Диаграмма 'boxblot':")
        plt.figure(figsize=(12, 6))
        sns.boxplot(x = self.data[column], color="orange", linewidth = 1.5)
        st.pyplot(plt)

    def show_graphs(self):
        column = st.selectbox("Признаки: ", self.params)
        if column is not None:
            self.displot(column)
            self.boxplot(column)


    