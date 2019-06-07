'''
Wholesale customers Data Set
Klaszterezés
forrás: UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/Wholesale+customers

Attribútumok:
    1) FRESH: annual spending (m.u.) on fresh products (Continuous);
    2) MILK: annual spending (m.u.) on milk products (Continuous);
    3) GROCERY: annual spending (m.u.)on grocery products (Continuous);
    4) FROZEN: annual spending (m.u.)on frozen products (Continuous)
    5) DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)
    6) DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
    7) CHANNEL: Channel - Horeca (Hotel/Restaurant/Café) or Retail channel (Nominal) (Horeca=1, Retail=2)
    8) REGION: Region - Lisbon, Oporto or Other (Nominal) (Lisbon=1, Oporto=2, Other=3)
'''

#Dataset beolvasása
import pandas as pd

dataset = pd.read_csv('data/Wholesale customers data.csv',
                      sep=',',
                      names=['Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'])

#Oszlopneveket tartalmazó sor törlése
dataset.drop(dataset.index[0], inplace=True)

#Régiókból az "Other" eltávoltítása
dataset = dataset[dataset.Region != '3']

dataset = dataset.reset_index()

#Típuskonverzió
for i in dataset:
    dataset[i] = dataset[i].astype('float64')

#Attribútumok szétválasztása
from sklearn.preprocessing import StandardScaler

features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = dataset.loc[:, features].values

#Sztenderdizáció
X = StandardScaler().fit_transform(X)

# region PCA (Region)

#Dimenzió redukció
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDF = pd.DataFrame(data=principalComponents, columns=['Principal_Component_1', 'Principal_Component_2'])
finalDF = pd.concat([principalDF, dataset[['Region']]], axis=1)

finalDF['Region'] = dataset['Region'].astype('int64')

#Vizualizáció
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('PCA a "Wholesale customers Data Set"-en (Region)', fontsize=20)

targets = [1, 2]
colors = ['r', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDF['Region'] == target
    ax.scatter(finalDF.loc[indicesToKeep, 'Principal_Component_1'],
               finalDF.loc[indicesToKeep, 'Principal_Component_2'],
               c=color,
               s=50)
ax.legend(['Lisbon', 'Oporto'])
ax.grid()
fig.savefig('data/pcaRegion.png')
fig.show()
fig.clf()
# endregion

# region PCA (Channel)

#Dimenzió redukció

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDF = pd.DataFrame(data=principalComponents, columns=['Principal_Component_1', 'Principal_Component_2'])
finalDF = pd.concat([principalDF, dataset[['Channel']]], axis=1)

finalDF['Channel'] = dataset['Channel'].astype('int64')

#Vizualizáció

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('PCA a "Wholesale customers Data Set"-en (Channel)', fontsize=20)

targets = [1, 2]
colors = ['r', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDF['Channel'] == target
    ax.scatter(finalDF.loc[indicesToKeep, 'Principal_Component_1'],
               finalDF.loc[indicesToKeep, 'Principal_Component_2'],
               c=color,
               s=50)
ax.legend(['Horeca', 'Retail'])
ax.grid()
fig.savefig('data/pcaChannel.png')
fig.show()
fig.clf()
# endregion

# region K-Means

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2).fit(X)

plt.figure(figsize=(8, 8))
for i in range(0, 2):
    plt.scatter(principalComponents[kmeans.labels_ == i][:, 0],
                principalComponents[kmeans.labels_ == i][:, 1])
plt.title('K-Means eredmények', fontsize=20)
plt.grid()
plt.savefig('data/kmeans.png')
plt.show()
plt.clf()
# endregion

# region DBSCAN

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=1, min_samples=15).fit(X)

plt.figure(figsize=(8, 8))
for i in set(dbscan.labels_):
    plt.scatter(principalComponents[dbscan.labels_ == i][:, 0],
                principalComponents[dbscan.labels_ == i][:, 1])
plt.title('DBSCAN eredmények', fontsize=20)
plt.grid()
plt.savefig('data/dbscan.png')
plt.show()
plt.clf()
# endregion

# region Hierarchikus klaszterezés (Agglomeratív)

from sklearn.cluster import AgglomerativeClustering

agglomerative = AgglomerativeClustering(n_clusters=2).fit(X)

plt.figure(figsize=(8, 8))
for i in range(0, 2):
    plt.scatter(principalComponents[agglomerative.labels_ == i][:, 0],
                principalComponents[agglomerative.labels_ == i][:, 1])
plt.title('Agglomeratív klaszterezési eredmények', fontsize=20)
plt.grid()
plt.savefig('data/agglomerative.png')
plt.show()
plt.clf()
# endregion

# region Adjusted Rand Score
from sklearn import metrics

print('Region összehasonlítása a K-means-el:\n', metrics.adjusted_rand_score(dataset['Region'].astype('int32'),kmeans.labels_))
print('Region összehasonlítása a DBSCAN-el:\n', metrics.adjusted_rand_score(dataset['Region'].astype('int32'),dbscan.labels_))
print('Region összehasonlítása az Agglomerative-el:\n', metrics.adjusted_rand_score(dataset['Region'].astype('int32'),agglomerative.labels_))
print('Channel összehasonlítása a K-means-el:\n',metrics.adjusted_rand_score(dataset['Channel'].astype('int32'),kmeans.labels_))
print('Channel összehasonlítása a DBSCAN-el:\n', metrics.adjusted_rand_score(dataset['Channel'].astype('int32'),dbscan.labels_))
print('Channel összehasonlítása a Agglomerative-el:\n', metrics.adjusted_rand_score(dataset['Channel'].astype('int32'),agglomerative.labels_))
# endregion

