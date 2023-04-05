# -> Decision Tree

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_wine()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split( x , y , test_size=0.2 )

DTree_model = DecisionTreeClassifier( criterion = "entropy" )
DTree_model.fit( x_train , y_train )
predict_DTree = DTree_model.predict( x_test )

print( "Decision Tree :\n")

DTree_test_score = DTree_model.score( x_test , y_test )

print( 'predict : ', predict_DTree )
print( 'real :',y_test ,'\n')
print( DTree_test_score )


'''
Predict :

X_test1 = [[1.51, 1.73, 1.98, 20.15, 85, 2.2, 1.92, .32, 1.48, 2.94, 1, 3.57, 172]]
X_test2 = [[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, .28, 2.29, 5.64, 1.04, 3.92, 1065]]
X_test3 = [[13.71, 5.65, 2.45, 20.5, 95, 1.68, .61, .52, 1.06, 7.7, .64, 1.74, 720]]

print( "Decition Tree predict: ")
print( "x_test1 : ", DTree_model.predict( X_test1 ))
print( "x_test2 : ", DTree_model.predict( X_test2 ))
print( "x_test3 : ", DTree_model.predict( X_test3 ))

'''