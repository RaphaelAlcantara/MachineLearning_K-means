from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Leitura dos dados e definição das colunas
X = pd.read_csv('kmeans.txt', sep=';', header=None)
X.columns = ['x', 'y', 'z']

# Definindo os dados que serão utilizados para clustering (colunas 'x', 'y' e 'z')
X = X.loc[:, ['x', 'y', 'z']]

# Número de clusters (K)
K = 6

# Gráfico da dispersão dos dados em 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Visualização geral')
ax.scatter(X['x'], X['y'], X['z'])
plt.show()

# Aplicando K-means
kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(X)
print('Agrupamento: ', kmeans.labels_)
print('Centroides: ', kmeans.cluster_centers_)

# Gráfico de dispersão dos dados em 3D com cores representando os clusters e centróides em preto
fig2 = plt.figure(figsize=(12, 8))
ax = fig2.add_subplot(111, projection='3d')
ax.set_title(f'Gráfico de dispersão com {K} clusters')
# Background color
ax.set_facecolor('grey')
fig2.set_facecolor('grey')

ax.scatter(X['x'], X['y'], X['z'], c=kmeans.labels_, cmap='rainbow')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], c='black')


plt.show()

# Método do cotovelo para escolher o melhor valor de K
valores_k = []
inercias = []

for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(X)
    valores_k.append(i)
    inercias.append(kmeans.inertia_)
    print(f'k: {i} - inércia: {kmeans.inertia_}')

# Observando o gráfico do método do cotovelo, o melhor valor de K parece ser 4
fig3 = plt.figure(figsize=(16, 8))
ax = fig3.add_subplot(111)
ax.set_title('Cotovelo')
# Background color
ax.set_facecolor('grey')
ax.plot(valores_k, inercias, '-o')
plt.show()
