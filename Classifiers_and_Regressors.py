import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error,r2_score

# TODO:
# - Implement K-fold validation
# - Replace Linear Regression with Linear SVM (SVC)
# - Implement Regressors
# - Perform tests for Multi & Final datasets
# - Implement Confusion matrices
# - Implement Command-line game
# - Implement Extra Credit?
# - Finish project tbh

def runClassifier(x,y,m,hl=2,k=5,k_fold=10):    
    # Split test
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # Normalize Data
    x_train=preprocessing.normalize(x_train)
    x_test=preprocessing.normalize(x_test)
    # sc = StandardScaler()
    # sc.fit(x_train)
    # StandardScaler(copy=True, with_mean=True, with_std=True)
    # x_train = sc.transform(X_train)
    # x_test = sc.transform(X_test)

    if m == 0:
        print("Plz implement me")
    elif m == 1:
        # Train MLP classifier
        layer_size=[]
        for i in range(0,hl):
            layer_size.append(11)
        layer_size = tuple(layer_size)
        print("DEBUG::"+str(layer_size))
        cl = MLPClassifier(hidden_layer_sizes=layer_size,max_iter=1000)
        cl.fit(x_train,y_train)
        predictions = cl.predict(x_test)
        
        # Get metrics
        cm = confusion_matrix(y_test,predictions,normalize="true")
        cm = cm.round(decimals=4)
        print("Multi-Layer Perceptron Metrics ("+ str(hl) +" hidden layers):")
        print("Confusion Matrix:")
        print(cm)
        print()
        # for ea in cm:
        #    print(np.sum(ea).round(decimals=2))
        # print()
        print("Classification Report:\n")
        print(classification_report(y_test,predictions))
        print("\n")

    elif m == 2:
        # Train K-Nearest Neighbor Classifier
        cl = KNeighborsClassifier(n_neighbors=k)
        cl.fit(x_train, y_train)
        predictions = cl.predict(x_test)
        
        # Get metrics
        cm = confusion_matrix(y_test,predictions,normalize="true")
        cm = cm.round(decimals=4)
        print("K-Nearest Neighbors (K = "+ str(k) +"):")
        print("Confusion Matrix:")
        print(cm)
        print()
        print("Classification Report:\n")
        print(classification_report(y_test,predictions))
        print()
        print("Score:")
        print(cl.score(x_test, y_test))
        print("\n")
    else:
        print("ERR: Bad Model ID")

def runRegressor():
    m = 0
    if m == 0:
        # Train Linear Regression Classifier
        cl = linear_model.LinearRegression()
        cl.fit(x_train, y_train)

        # Make predictions
        y_predict = cl.predict(x_test)
        
        print("Linear Regression Metrics:")
        print('MSE: %.3f' % mean_squared_error(y_test, y_predict))
        print('CoD: %.3f' % r2_score(y_test, y_predict))
        print("\n")

        # The below code is broken; confusion matrices cannot work
        # on continuous data!
        #
        # cm = confusion_matrix(y_test,y_predict,normalize="true")
        # print(cm)

def main():
    single = np.loadtxt('tictac_single.txt')
    multi = np.loadtxt('tictac_multi.txt')
    final = np.loadtxt('tictac_final.txt')

    x_single = single[:,:9]
    y_single = np.hstack(single[:,9:])

    # Single LR
    runClassifier(x_single, y_single, 0)

    # Single MLP
    # runClassifier(x_single, y_single, 1, hl=1)
    # runClassifier(x_single, y_single, 1)
    # runClassifier(x_single, y_single, 1, hl=3)
    # runClassifier(x_single, y_single, 1, hl=4)
    # runClassifier(x_single, y_single, 1, hl=5)
    # runClassifier(x_single, y_single, 1, hl=6)
    
    runClassifier(x_single, y_single, 2, k=4)
    runClassifier(x_single, y_single, 2)
    runClassifier(x_single, y_single, 2, k=6)
    runClassifier(x_single, y_single, 2, k=7)
    runClassifier(x_single, y_single, 2, k=8)
    runClassifier(x_single, y_single, 2, k=9)
    runClassifier(x_single, y_single, 2, k=10)
    runClassifier(x_single, y_single, 2, k=11)

main()

    

    
        

