from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dx,dy = make_blobs( n_samples = 500 , n_features = 2 , centers = 10 , random_state = 42 )

model = KMeans( n_clusters = 10 ) # > 分幾組
model.fit( dx )
new_dy = model.predict( dx )

plt.figure( figsize = ( 10 , 10 ) )

plt.subplot( 121 )
plt.title( 'k-Means = 10 groups' )
plt.scatter( dx.T[0] , dx.T[1] ,  c = new_dy , cmap = plt.cm.Set1 )

plt.tight_layout()
plt.show()