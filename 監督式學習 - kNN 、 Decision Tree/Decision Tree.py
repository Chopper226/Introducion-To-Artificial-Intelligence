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