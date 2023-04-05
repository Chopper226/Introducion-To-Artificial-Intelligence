# -> Knn

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = load_wine()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split( x , y , test_size=0.2 )

Knn_model=KNeighborsClassifier( n_neighbors = 11 )
Knn_model.fit( x_train, y_train )
predict_Knn = Knn_model.predict( x_test )

print( "Knn :\n")

Knn_test_score = Knn_model.score( x_test , y_test )

print( 'predict :',predict_Knn )
print( 'real : ',y_test ,"\n")
print( Knn_test_score )