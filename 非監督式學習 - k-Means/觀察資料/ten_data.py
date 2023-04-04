# 觀察前 10 筆資料

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
plt.figure( figsize = ( 5 , 5 ) )

for i in range( 1 , 11 ) :
  plt.subplot( 2 , 5 , i )
  plt.imshow( digits.images[i] , cmap = plt.cm.binary )

plt.show()