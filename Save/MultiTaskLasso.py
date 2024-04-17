import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import MultiTaskLasso, Lasso
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

data = pd.read_csv("Regression.csv")

y = data['MEDV']
X = data.drop(['MEDV'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, shuffle = False)
mlt = MultiTaskLasso(alpha=1.0)

with open('MultiTaskLasso.pkl','wb') as file:
    pickle.dump(mlt, file)