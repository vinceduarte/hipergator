import numpy as np
from math import ceil,floor
from numpy import mean
from numpy import absolute
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_squared_error,r2_score

# TODO:
# - Implement Extra Credit?
# - Finish video


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
def runClassifier(x_train,x_test,y_train,y_test,c,reg=1.0,hl=3,k=5,k_fold=10,debug=0,g=0,p=1):
    # Select Classifier
    if c == 0:
        # LinearSVC (SVM)
        cl = SVC(kernel='linear',C=reg)
        cl.fit(x_train,y_train)
        predictions = cl.predict(x_test)

        # Get metrics
        if p == 1:
            print("Linear SVM:")
            getMetrics(cl,x_train,y_train,x_test,y_test,predictions,k_fold=k_fold,debug=debug,graph=g)

        # Return classifier
        return cl
    
    elif c == 1:
        # Modify MLP structure
        layer_size=[]
        for i in range(0,hl):
            layer_size.append(9)
        layer_size = tuple(layer_size)
        
        # Train MLP classifier
        cl = MLPClassifier(hidden_layer_sizes=layer_size,activation='tanh')
        cl.fit(x_train,y_train)
        predictions = cl.predict(x_test)
        
        # Get metrics
        if p == 1:
            print("Multi-Layer Perceptron Metrics ("+ str(hl) +" hidden layers):")
            getMetrics(cl,x_train,y_train,x_test,y_test,predictions,k_fold=k_fold,debug=debug,graph=g)

        # Return classifier
        return cl
    
    elif c == 2:
        # Train K-Nearest Neighbor Classifier
        cl = KNeighborsClassifier(n_neighbors=k)
        cl.fit(x_train, y_train)
        predictions = cl.predict(x_test)
        
        # Get metrics
        if p == 1:
            print("K-Nearest Neighbors (K = "+ str(k) +"):")
            getMetrics(cl,x_train,y_train,x_test,y_test,predictions,k_fold=k_fold,debug=debug,graph=g)

        # Return classifier
        return cl
        
    else:
        print("ERR: Bad Classifier ID")


#
# runRegressor:
# x/y_train/test -> Training/Testing data for x and y
# r -> regressor_id
# hl -> number of hidden layers; default 3
# k -> number of nearest neighbors; default 5
# k_fold -> number of folds for k-fold validation
# debug -> load debug prints; default 0
#
def runRegressor(x_train,x_test,y_train,y_test,r,hl=3,k=5,k_fold=10,debug=0,g=0,p=1):
    # Select Regressor
    if r == 0:
        # Train Linear Regression
        rg = linear_model.LinearRegression()
        rg.fit(x_train, y_train)

        
        x_train = np.c_[ np.ones(len(x_train)), x_train]
        x_test = np.c_[ np.ones(len(x_test)), x_test]

        x_train = np.array(x_train)
        xt = np.transpose(x_train)
        xtx = np.dot(xt,x_train)
        xty = np.dot(xt,y_train)
        b_prime = np.linalg.solve(xtx,xty)


        predictions = []
        predictions = np.array(predictions)
        for x in x_test:
            prediction = []
            prediction = np.array(prediction)
            #print(len(x))
            for i in range(len(x)):
                # print(x[i])
                # print("~")
                # print(b_prime[i])
                #print(len(b_prime))
                for j in range(len(b_prime[i])):
                    val = x[i] * b_prime[i][j]
                prediction = np.append(prediction, np.abs(np.ceil(val)))
            predictions = np.append(predictions, prediction)
          
            #predictions.append(np.dot(x, b_prime))
        

        # Make predictions
        #predictions = rg.predict(x_test)
        prediction = prediction.reshape(-1,9)
        print(predictions[0:10])
        print()
        print(y_test[0:10])

        # Get Metrics
        if p == 1:
            print("Linear Regression Metrics:")
            print("Acc: %.3f" % getAccuracy(y_test, predictions))
            # getMetrics(rg,x_train,y_train,x_test,y_test,predictions,m_id=1,k_fold=k_fold,debug=debug,graph=g)

        # Return regressor
        return rg
    
    elif r == 1:
        # Modify MLP structure
        layer_size=[]
        for i in range(0,hl):
            layer_size.append(9)
        layer_size = tuple(layer_size)
        
        # Train MLP Regressor
        rg = MLPRegressor(hidden_layer_sizes=layer_size, batch_size=10, activation='tanh')
        rg.fit(x_train,y_train)

        # Make Predictions
        predictions = rg.predict(x_test).round()
        
        # Debug
        if debug == 1:       
            for each in predictions[0:10]:
                print(each)
            print()
            for each in y_test[0:10]:
                print(each)
        
        # Get Metrics
        if p == 1:
            print("Multi-Layered Perceptron Regression ("+ str(hl) +" hidden layers) Metrics:")
            print("Acc: %.3f" % getAccuracy(y_test, predictions))
            # getMetrics(rg,x_train,y_train,x_test,y_test,predictions,m_id=1,k_fold=k_fold,debug=debug,graph=g)

        # Return regressor
        return rg
    
    elif r == 2:
        # Train K-NN Regressor
        rg = KNeighborsRegressor(n_neighbors = k)
        rg.fit(x_train,y_train)

        # Make Predictions
        predictions = rg.predict(x_test).round()
                    
        # Get Metrics
        if p == 1:
            print("K-Nearest Neighbor Regression ("+ str(k) +" neighbors) Metrics:")
            print("Acc: %.3f" % getAccuracy(y_test, predictions))
            # getMetrics(rg,x_train,y_train,x_test,y_test,predictions,m_id=1,k_fold=k_fold,debug=debug,graph=g)

        # Return regressor
        return rg
    
    else:
        print("ERR: Bad Regressor ID")


def getMetrics(m,x_train,y_train,x_test,y_test,predictions,m_id=0,k_fold=10,debug=0,graph=0):
    if m_id == 0:
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test,predictions,normalize="true")
        print(cm.round(decimals=4))
        print()
        if debug == 1:
            for ea in cm:
                print(np.sum(ea).round(decimals=2))
            print()
        print("Classification Report:\n")
        print(classification_report(y_test,predictions))
        print()
        print("Score:")
        print(m.score(x_test, y_test))
        kf = KFold(n_splits=k_fold, random_state=1, shuffle=True)
        scores = cross_val_score(m, x_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
        print('K-fold (K = ' + str(k_fold) + ' folds) validation - Mean Absolute Error: %.3f' % mean(absolute(scores)))
        print("\n")
    elif m_id == 1:
        print('MSE: %.3f' % mean_squared_error(y_test, predictions))
        print('R2: %.3f' % r2_score(y_test, predictions))
        kf = KFold(n_splits=k_fold, random_state=1, shuffle=True)
        scores = cross_val_score(m, x_train, y_train, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
        print('K-fold (K = ' + str(k_fold) + ' folds) validation - Mean Absolute Error: %.3f' % mean(absolute(scores)))
        print("\n")
    
def normalizeInputs(x_train, x_test):
    x_train=(x_train - np.min(x_train))/(np.max(x_train) - np.min(x_train))
    x_test=(x_test - np.min(x_test))/(np.max(x_test) - np.min(x_test))
    return x_train, x_test

def getAccuracy(y_test, predictions):
    n_total = len(y_test)
    n_match = 0
    if n_total == len(predictions):
        for i in range(n_total):
            if len(predictions[i]) == 1:
                if predictions[i].round() == y_test[i]:
                    n_match += 1
            else:
                if (predictions[i].round() == y_test[i]).all():
                    n_match += 1
    return n_match / n_total

def main():
    # Load data
    single = np.loadtxt('tictac_single.txt')
    multi = np.loadtxt('tictac_multi.txt')
    final = np.loadtxt('tictac_final.txt')

    # Split datasets
    # optimal play single-label
    x_single = single[:, :9]
    y_single = np.hstack(single[:, 9:])

    # final score
    x_final = final[:, :9]
    y_final = np.hstack(final[:, 9:])

    # optimal play multi-label
    x_multi = multi[:, :9]
    y_multi = multi[:, 9:]

    rng = np.random.RandomState(0)
    # Split train/test
    x_train_single, x_test_single, y_train_single, y_test_single = train_test_split(
        x_single, y_single, test_size=0.2, random_state=rng)
    x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(
        x_final, y_final, test_size=0.2, random_state=rng)
    x_train_multi, x_test_multi, y_train_multi, y_test_multi = train_test_split(
        x_multi, y_multi, test_size=0.2, random_state=rng)

    x_train_final, x_test_final = normalizeInputs( x_train_final, x_test_final)
    x_train_single, x_test_single = normalizeInputs(x_train_single, x_test_single)
    # x_train_multi, x_test_multi = normalizeInputs(x_train_multi, x_test_multi)

    # -- REGRESSORS --
    # Run Single Linear Regression
    print(" ~~~~~~~ Running Linear Regression ~~~~~~~")
    runRegressor(x_train_multi, x_test_multi, y_train_multi, y_test_multi, 0)
    


    # Run Single MLP Regression
    print(" ~~~~~~~ Running MLP Regression ~~~~~~~")
    # for i in range(1, 6):
        # runRegressor(x_train_multi, x_test_multi,
                     # y_train_multi, y_test_multi, 1, hl=i)

    # Run Single K-NN Regression
    print(" ~~~~~~~ Running K-NN Regression ~~~~~~~")
    # for i in range(1, 10):
        # runRegressor(x_train_multi, x_test_multi,
                      # y_train_multi, y_test_multi, 2, k=i)

    # -- CLASSIFIERS --
    # Single LinearSVM
    print("~~~~~~~ Running LinearSVM Classifier ~~~~~~~")
    runClassifier(x_train_single, x_test_single,
                  y_train_single, y_test_single, 0)
    # runClassifier(x_train_final, x_test_final, y_train_final, y_test_final, 0)

    # Single MLP
    print("~~~~~~~ Running MLP Classifier ~~~~~~~")
    for i in range(1, 6):
        runClassifier(x_train_single, x_test_single,
                      y_train_single, y_test_single, 1, hl=i)
        # runClassifier(x_train_final, x_test_final,
                      # y_train_final, y_test_final, 1, hl=i)

    # Single
    print("~~~~~~~ Running K-NN Classifier ~~~~~~~")
    for i in range(1, 10):
        runClassifier(x_train_single, x_test_single,
                      y_train_single, y_test_single, 2, k=i)
        # runClassifier(x_train_final, x_test_final,
                      # y_train_final, y_test_final, 2, k=i)


main()

    

    
        

