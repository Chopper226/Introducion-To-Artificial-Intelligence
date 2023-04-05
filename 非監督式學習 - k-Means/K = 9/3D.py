from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dx , dy = make_blobs( n_samples = 500 , n_features = 3 , centers = 10 , random_state = 42 )

model = KMeans( n_clusters = 9 ) # > 分幾組
model.fit( dx )
new_dy = model.predict( dx )

fig = plt.figure( figsize = ( 16 , 8 ) )

ax = fig.add_subplot( 121 , projection = '3d' )
plt.title( 'KMeans =  9 groups' )
ax.scatter( dx.T[0] , dx.T[1] , dx.T[2] , c = new_dy, cmap = plt.cm.Set1 )

plt.tight_layout()
plt.show()