# Run this program on your local python 
# interpreter, provided you have installed 
# the required libraries. 

# Importing the required packages 
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# Function importing Dataset 
def importdata(): 
	balance_data = pd.read_csv( 
'https://archive.ics.uci.edu/ml/machine-learning-'+
'databases/balance-scale/balance-scale.data', 
	sep= ',', header = None) 
	
	# Printing the dataswet shape 
	print ("Dataset Lenght: ", len(balance_data)) 
	print ("Dataset Shape: ", balance_data.shape) 
	
	# Printing the dataset obseravtions 
	print ("Dataset: ",balance_data.head()) 
	return balance_data 

# Function to split the dataset 
def splitdataset(balance_data): 

	# Seperating the target variable 
	X = balance_data.values[:, 1:5] 
	Y = balance_data.values[:, 0] 

	# Spliting the dataset into train and test 
	X_train, X_test, y_train, y_test = train_test_split( 
	X, Y, test_size = 0.25, random_state=100) 
	
	return X, Y, X_train, X_test, y_train, y_test 
	
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 

	# Creating the classifier object 
	clf_gini = DecisionTreeClassifier(criterion = "gini", 
			random_state = 100, max_depth=8, min_samples_leaf=5) 

	# Performing training 
	clf_gini.fit(X_train, y_train) 
	return clf_gini 
	
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 

	# Decision tree with entropy 
	clf_entropy = DecisionTreeClassifier( criterion = "entropy",
                        random_state = 100, max_depth = 8, min_samples_leaf = 5) 

	# Performing training 
	clf_entropy.fit(X_train, y_train) 
	return clf_entropy 


# Function to make predictions 
def prediction(X_test, clf_object): 

	# Predicton on test with giniIndex 
	y_pred = clf_object.predict(X_test) 
	print("Predicted values:") 
	print(y_pred) 
	return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
	
	print("Confusion Matrix: ", 
		confusion_matrix(y_test, y_pred)) 
	
	print ("Accuracy : ", 
	accuracy_score(y_test,y_pred)*100) 
	
	print("Report : ", 
	classification_report(y_test, y_pred)) 




# Driver code 
def main(): 
	
	# Building Phase 
	data = importdata() 
	X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
	clf_gini = train_using_gini(X_train, X_test, y_train) 
	clf_entropy = tarin_using_entropy(X_train, X_test, y_train)

        #Visualizing tree using Gini Index
	dot_data = StringIO()
	export_graphviz(clf_gini, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_png('gini_graph.png')
	Image(graph.create_png())

	
	print('\n')
        # Operational Phase 
	print("Results Using Gini Index:") 
	print ("\n")
	
	# Prediction using gini 
	y_pred_gini = prediction(X_test, clf_gini)
        #Test instance prdictions
	print("\n")
	
	test1_set =[1,1,1,1]
	print ("Test instance 1:  ", test1_set) 
	test1 = clf_gini.predict([test1_set])
	print("Predicted label: ", test1)
	print("Actual label: B")
	print('\n')
        
	test2_set =[1,3,2,3]
	print ("Test instance 2:  ", test2_set) 
	test2 = clf_gini.predict([test2_set])
	print("Predicted label: ", test2)
	print("Actual label: R")
	print('\n')

	test3_set = [5,4,5,1]
	print ("Test instance 3:  ", test3_set) 
	test3 = clf_gini.predict([test3_set])
	print("Predicted label: ", test3)
	print("Actual label: L")
	print('\n')

	test7_set = [1,4,1,4]
	print ("Test instance 4:  ", test7_set) 
	test7 = clf_gini.predict([test7_set])
	print("Predicted label: ", test7)
	print("Actual label: B")
	print('\n')
        
	
	cal_accuracy(y_test, y_pred_gini)

        #Visualizing tree using Entropy
	dot_data = StringIO()
	export_graphviz(clf_entropy, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_png('entropy_graph.png')
	Image(graph.create_png())
        
	print("Results Using Entropy:")

	print('\n')
	# Prediction using entropy 
	y_pred_entropy = prediction(X_test, clf_entropy)
	print('\n')

	
	test4_set =[1,1,1,1]
	print ("Test instance 1:  ", test4_set) 
	test4 = clf_gini.predict([test4_set])
	print("Predicted label: ", test4)
	print("Actual label: B")
	print('\n')
        
	test5_set =[1,3,2,3]
	print ("Test instance 2:  ", test5_set) 
	test5 = clf_gini.predict([test5_set])
	print("Predicted label: ", test5)
	print("Actual label: R")
	print('\n')

	test6_set = [5,4,5,1]
	print ("Test instance 3:  ", test6_set) 
	test6 = clf_gini.predict([test6_set])
	print("Predicted label: ", test6)
	print("Actual label: L")
	print('\n')

	test8_set = [1,4,1,4]
	print ("Test instance 4:  ", test8_set) 
	test8 = clf_gini.predict([test8_set])
	print("Predicted label: ", test8)
	print("Actual label: B")
	print('\n')
	
	
	cal_accuracy(y_test, y_pred_entropy)


	
	
	
# Calling main function 
if __name__=="__main__": 
	main() 
