# 觀察第一筆資料

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

print( "First_data : " , digits.data[0] , "\n" )
print( "First_image : " , digits.images[0] , "\n" )
print( "First_target : " , digits.target[0] , "\n" )

# 將資料視覺化
plt.imshow( digits.images[0] , cmap = plt.cm.binary )
plt.show()