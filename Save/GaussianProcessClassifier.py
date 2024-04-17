import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from pandas.api.types import is_string_dtype

data = pd.read_csv("Classification.csv")

columns = []
for i in data.columns.tolist():
    if is_string_dtype(data[i]):
        columns.append(i)

columns.append('Alcohol Consumption')
columns.append('Medical Conditions')
columns.append('Medications')

#Кодирование категориальных прихнаков
label_encoder = preprocessing.LabelEncoder()

for i in columns:
  data[i] = label_encoder.fit_transform(data[i])

#Обучение
#data = data.drop(['Unnamed: 0'], axis=1)
y = data["Osteoporosis"]
x = data.drop(["Osteoporosis"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X_train, y_train)

with open('GaussianProcessClassifier.pkl','wb') as file:
    pickle.dump(gpc, file)
