from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv('kmeans.txt', sep=';', header=None)

X.columns = ['x', 'y', 'z']

X = X.loc[:, ['x', 'y']]

K = 4

# Gráfico da dispersão dos dados
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('Visualização geral')
ax.scatter(X['x'], X['y'])
plt.show()

kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(X)
print('Agrupamento: ', kmeans.labels_)
print('Centroides: ', kmeans.cluster_centers_)

fig2, ax = plt.subplots(figsize=(16, 8))
ax.set_title(f'Gráfico de dispersão com {K} clusters')
# background color
ax.set_facecolor('grey')

ax.scatter(X['x'], X['y'], c=kmeans.labels_, cmap='rainbow')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black')
plt.show()

# metodo do cotovelo

valores_k = []
inercias = []

for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(X)
    valores_k.append(i)
    inercias.append(kmeans.inertia_)
    print(f'k: {i} - inercia: {kmeans.inertia_}')

# observando o grafico, o melhor valor de k é 4
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_title('Cotovelo')
# background color
ax.set_facecolor('grey')
ax.plot(valores_k, inercias, '-o')
plt.show()
