'''Realizando Pré-preocessamento em uma base composta por variaveis Categóricas'''

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.express as px

# Exploração dos dados
df = pd.read_csv('census.csv')
print(df.head(10))
print(df.describe())
print(df.isnull().sum())

# Visualização dos dados
np.unique(df['income'], return_counts=True)
sn.countplot(x=df['income'])
plt.show()

plt.hist(x=df['age'])
plt.show()

plt.hist(x=df['education-num'])
plt.show()

plt.hist(x=df['hour-per-week'])
plt.show()