import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv("salary v1.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

# menghilangkan missing value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 2:3])
x[:, 2:3] = imputer.transform(x[:, 2:3])

print(x)

# Encoding data kategori(atribut)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)

# Encoding data kategori(class/label)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

# Membagi dataset ke dalam training set dan test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size= 0.2, random_state=1)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 2:3] = sc.fit_transform(x_train[:, 2:3])
x_test[:, 2:3] = sc.transform(x_test[:, 2:3])

print(x_train)
print(x_test)