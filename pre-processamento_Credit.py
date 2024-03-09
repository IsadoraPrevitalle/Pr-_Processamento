'''Realizando Pré-preocessamento em uma base composta por variaveis Numéricas
Essa base representa uma associação entre renda, idade e divida de empréstimos realizados em comparação a variavel de pagamento dessa divida'''

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.express as px

#ID --> numeral discreta
#Income --> renda anual --> numérica continua
#Age --> numeral discreta
#Divida --> numerica continua
#Default --> pagamento da divida --> numerico discreto

df = pd.read_csv('credit_data.csv')

#Dados descritivos
print(df.describe())

#Diferentes valores dessa classe
#return_counts - respectivas quantidades 
print(np.unique(df['default'], return_counts = True))

#Visualização dos dados
#Histogramas de algumas das classes principais
sn.countplot(x=df['default'])
plt.show()
plt.hist(x=df['age'])
plt.show()
plt.hist(x=df['income'])
plt.show()
# grafico = px.scatter_matrix(df, dimensions=['age', 'income','loan'], color='default')
# grafico.show()

# Tratamento de idades negativas
print(df[df['age']<0])

# 1º maneira - apagar a coluna
df2 = df.drop('age', axis=1)
print(df2.head(8))

# 2º maneira - apagar somente registros com valores incosistentes
df3 = df.drop(df[df['age']<0].index)
print(df3.loc[df3['age']<0])

# 3º maneira - preencher os valores inconsistentes manualmente
# 4º maneira - preencher os valores com a média das idades 

print(df['age'][df['age']>0].mean())
print(df.loc[df['age']< 0])
df.loc[df['age']<0, 'age'] = 40.92
print(df.loc[df['age']< 0])
print(df.head(27))

# Tratamento de valores faltantes
# Somatória nulos
print(df.isnull().sum())

# Busca de clientes com valores idade nula
print(df.loc[pd.isnull(df['age'])])

# Função fillna - preencher os valores nulos com a média
print(df['age'].fillna(df['age'].mean(), inplace=True))
# Verificar se existe null
print(df.loc[pd.isnull(df['age'])])

# Verificando os mesmos Ids
print(df.loc[df['clientid'].isin([29,31,92])])

# Divisão entre previsores e classe meta
X_credit = df.iloc[:,1:4].values #Previsores 
y_credit = df.iloc[:, 4].values #Meta

#Escalonamento dos valores - renda e idade
''' Erros podem ocorrer pelo fato de o algoritmo considerar a renda mais importante
É necessário normalizar ou padronizar os valores, especialmente quando o algotirmo é sensível aos dados'''

print("renda / idade / divida ")
print(X_credit[:,0].min(), X_credit[:,1].min(), X_credit[:,2].min())
print(X_credit[:,0].max(), X_credit[:,1].max(), X_credit[:,2].max()) 

# StandardScaler = padronização
# KNN é um algoritmo extremamente sensivel, ele precisa dessa padronização

from sklearn.preprocessing import StandardScaler
scaler_credit = StandardScaler()

X_credit = scaler_credit.fit_transform(X_credit)

# Aqui é comum dados negativos
print("renda / idade / divida ")
print(X_credit[:,0].min(), X_credit[:,1].min(), X_credit[:,2].min())
print(X_credit[:,0].max(), X_credit[:,1].max(), X_credit[:,2].max()) 

# Divisãode dados de treino e teste
from sklearn.model_selection import train_test_split

# Por padrão se utiliza em torno de 70% a 80% da base, com exeções de base de dados pequenas, onde  aumentanmos os dados de treino
X_treino, X_teste, y_treino, y_teste = train_test_split(X_credit, y_credit, test_size=0.25, random_state=0)
# Random_state é para definir uma config padrão
# Shape exibe qtd de linhas e col
print(X_treino.shape) 
print(X_teste.shape)
print(y_treino.shape)
print(y_teste.shape)

# Salvar variaveis em formato binário com o módulo pickle para serem utilizadas posteriormente
import pickle
with open ('df.pkl', mode='wb') as f:
    pickle.dump([X_treino, y_treino, X_teste, y_teste], f)