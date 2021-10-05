import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error,r2_score

# TODO:
# - Implement K-fold validation
# - Fix Linear SVM (SVC)
# - Implement Regressors & their metrics
# - Perform tests for Multi & Final datasets
# - Implement Command-line game
# - Implement Extra Credit?
# - Finish project tbh

#
# runClassifier:
# x/y_train/test -> Training/Testing data for x and y
# c -> classifier_id
# reg -> regularizer for LinearSVC; default 1.0
# hl -> number of hidden layers; default 3
# k -> number of nearest neighbors; default 5
# k_fold -> number of folds for k-fold validation
# debug -> load debug prints; default 0
#
def runClassifier(x_train,x_test,y_train,y_test,c,reg=1.0,hl=3,k=5,k_fold=10,debug=0):    
    # Normalize Data
    x_train=preprocessing.normalize(x_train)
    x_test=preprocessing.normalize(x_test)
    # sc = StandardScaler()
    # sc.fit(x_train)
    # StandardScaler(copy=True, with_mean=True, with_std=True)
    # x_train = sc.transform(X_train)
    # x_test = sc.transform(X_test)

    if c == 0:
        # LinearSVC (SVM)
        cl = SVC(kernel='linear',C=reg)
        cl.fit(x_train,y_train)
        predictions = cl.predict(x_test)
        
        # Get metrics
        cm = confusion_matrix(y_test,predictions,normalize="true")
        cm = cm.round(decimals=4)
        print("Linear SVM:")
        print("Confusion Matrix:")
        print(cm)
        print()
        if debug == 1:
            for ea in cm:
                print(np.sum(ea).round(decimals=2))
            print()
        print("Classification Report:\n")
        print(classification_report(y_test,predictions))
        print("\n")        
    elif c == 1:
        # Modify MLP structure
        layer_size=[]
        for i in range(0,hl):
            layer_size.append(11)
        layer_size = tuple(layer_size)
        
        # Train MLP classifier
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
        if debug == 1:
            for ea in cm:
                print(np.sum(ea).round(decimals=2))
            print()
        print("Classification Report:\n")
        print(classification_report(y_test,predictions))
        print("\n")
    elif c == 2:
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
        if debug == 1:
            for ea in cm:
                print(np.sum(ea).round(decimals=2))
            print()
        print("Classification Report:\n")
        print(classification_report(y_test,predictions))
        print()
        print("Score:")
        print(cl.score(x_test, y_test))
        print("\n")
    else:
        print("ERR: Bad Classifier ID")

def runRegressor(x_train,x_test,y_train,y_test,r,hl=3):
    if r == 0:
        # Train Linear Regression
        rg = linear_model.LinearRegression()
        rg.fit(x_train, y_train)

        # Make predictions
        y_predict = rg.predict(x_test)
        
        print("Linear Regression Metrics:")
        print('MSE: %.3f' % mean_squared_error(y_test, y_predict))
        print('CoD: %.3f' % r2_score(y_test, y_predict))
        print("\n")
    elif r == 1:
        # Modify MLP structure
        layer_size=[]
        for i in range(0,hl):
            layer_size.append(11)
        layer_size = tuple(layer_size)
        
        # Train MLP Regressor
        rg = MLPRegressor(hidden_layer_sizes=layer_size, max_iter=1000)
        rg.fit(x_train,y_train)

        # Make Predictions
        predictions = rg.predict(x_test)
        for each in predictions[0:10]:
            print(each)
        
        for each in y_test[0:10]:
            print(each)
        
        # Get Metrics
        print("Multi-Layered Perceptron Regressor ("+ str(hl) +" hidden layers):")
        print("Classification Report:\n")
        print(classification_report(y_test,predictions))
        print()
        print("Score:")
        print(cl.score(x_test, y_test))
        print("\n")
    elif r == 2:
        print("implement pls")
    else:
        print("ERR: Bad Regressor ID")
        

def main():
    # Load data
    single = np.loadtxt('tictac_single.txt')
    multi = np.loadtxt('tictac_multi.txt')
    final = np.loadtxt('tictac_final.txt')

    # Create datasets
    x_single = single[:,:9]
    y_single = np.hstack(single[:,9:])

    # Split train/test
    x_train, x_test, y_train, y_test = train_test_split(x_single, y_single)

    runRegressor(x_train, x_test, y_train, y_test, 1)

    # Single LR
    runClassifier(x_train, x_test, y_train, y_test, 0)

    # Single MLP
    runClassifier(x_train, x_test, y_train, y_test, 1, hl=1)
    runClassifier(x_train, x_test, y_train, y_test, 1, hl=2)
    runClassifier(x_train, x_test, y_train, y_test, 1)
    runClassifier(x_train, x_test, y_train, y_test, 1, hl=4)
    runClassifier(x_train, x_test, y_train, y_test, 1, hl=5)

    # Single 
    runClassifier(x_train, x_test, y_train, y_test, 2, k=4)
    runClassifier(x_train, x_test, y_train, y_test, 2)
    runClassifier(x_train, x_test, y_train, y_test, 2, k=6)
    runClassifier(x_train, x_test, y_train, y_test, 2, k=7)
    runClassifier(x_train, x_test, y_train, y_test, 2, k=8)

main()

    

    
        

