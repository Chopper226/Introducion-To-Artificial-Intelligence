from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
model = KMeans( n_clusters = 7 ) 
model.fit( digits.data )       

class_0 = np.where( model.labels_ == 0 )[0]
class_1 = np.where( model.labels_ == 1 )[0]
class_2 = np.where( model.labels_ == 2 )[0]

print( class_0[0:10] )
print( class_1[0:10] )
print( class_2[0:10] )

plt.figure( figsize = ( 20 , 20 ) )
for i in range( 1 , 4 ) :
    for j in range( 1 , 11 ) :
        plt.subplot( 3 , 10 , (i-1)*10+j )
        if i == 1 :
            plt.imshow( digits.images[ class_0[j] ], cmap = plt.cm.binary , interpolation = 'sinc' )
        elif i == 2 :
            plt.imshow( digits.images[ class_1[j] ], cmap = plt.cm.binary , interpolation = 'sinc' )
        elif i == 3 :
            plt.imshow( digits.images[ class_2[j] ], cmap = plt.cm.binary , interpolation = 'sinc' )

plt.tight_layout()
plt.show()