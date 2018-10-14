### Averaged Perceptron

## Import libraries
import os
import sys
import struct
import numpy as np
from scipy.special import expit
from array import array as pyarray

#np.set_printoptions(threshold=np.nan)
#np.set_printoptions(suppress=True)


## Read MNIST dataset and reshape
def load_data(dataset, pathToDataFolder='./data/', nSamplesToUse=10000):
    #pathToDataFolder="./data/"
    #nSamplesToUse = 10000    
    if dataset == "training":
        fname_img = os.path.join(pathToDataFolder, 'train-images.idx3-ubyte')
        fname_labl = os.path.join(pathToDataFolder, 'train-labels.idx1-ubyte')
    elif dataset == "test":
        fname_img = os.path.join(pathToDataFolder, 't10k-images.idx3-ubyte')
        fname_labl = os.path.join(pathToDataFolder, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flabl = open(fname_labl, 'rb')
    magic_nr, size = struct.unpack(">II", flabl.read(8)) # size=60,000
    labl = pyarray("b", flabl.read()) # len(labl)=60,000
    flabl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, nRows, nCols = struct.unpack(">IIII", fimg.read(16)) # size=60,000, nRows=28, nCols=28
    img = pyarray("B", fimg.read()) # len(img)=47040000=60,000x784
    fimg.close()

    # Limit the number of samples to use for training
    if dataset == "training":
        labl = labl[:nSamplesToUse] # len(labl)=nSamplesToUse
        img = img[:nSamplesToUse*784] # len(img)=nSamplesToUse*784
    elif dataset == "test":
        nSamplesToUse = len(labl) # 10,000
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
        
    # Shuffle the dataset (img, labl)
    labl = np.array(labl).reshape(nSamplesToUse,1)
    img = np.array(img).reshape(nSamplesToUse,len(img)/nSamplesToUse)
    #print labl.shape, img.shape # DEBUG
    img_and_labl = np.hstack((img, labl))
    np.random.shuffle(img_and_labl)
    labl = img_and_labl[:,-1].reshape(nSamplesToUse,1)
    img = img_and_labl[:,:-1] #.reshape(-1,1)
    #print 'labels.shape:',labl.shape,'images.shape:',img.shape # (10000x1), (10000x784) # DEBUG
    
    return img, labl


# Averaged Perceptron
def averaged_perceptron(digit, img, labl, nEpoch=50, learnRate=0.001):
    # Input & Output formatting
    x_bias = np.ones((img.shape[0],1)) # bias term (nSamplesx1): pass the input value 1
    x = np.hstack((np.round(img / 255.0), x_bias)) # feature space: (nSamplesx785)
    y = 1*(labl==digit) + -1*(labl!=digit) # (nSamplesx1): +1 or -1
    
    # Initialize weights
    np.random.seed(1)
    w = 2 * np.random.random((x.shape[1],1)) - 1 # -1 < w < 1 (785x1)
    a = w # accumulated weight
    cnt = 1
    
    # Learn and update weights (Gradient descent)
    #nEpoch = 50, learnRate = 0.001
    for iter in xrange(nEpoch):
        y_hat = np.sign(np.dot(x, w)) # (nSamplesx785)x(785x1)=(nSamplesx1)        
        updateOrNot = (y_hat != y)
        w += learnRate * np.dot(x.T, np.multiply(y, updateOrNot).astype(float))
        ## DEBUG ##
        a = a + w
        #a = a + cnt * learnRate * np.dot(x.T, np.multiply(y, updateOrNot).astype(float))
        cnt += 1
    
    a = a/float(cnt)
    #a = w - a/float(nEpoch)
    
    #print 'a[300:305]:',a[300:305,-1]
    #print 'w[300:305]:',w[300:305,-1]
    return a # (nSamplesx1)


# Test the model
def test(img, w):
    x_bias = np.ones((img.shape[0],1)) # bias term (nSamplesx1): pass the input value 1
    x = np.hstack((np.round(img / 255.0), x_bias)) # feature space (nSamplesx785)
    
    xdotw = np.dot(x, w) # (nSamplesx10)
    y_pred= np.argmax(xdotw, axis=1) # outputs a vector of indices: (nSamplesx1)
    
    return y_pred # (nSamplesx1)


## Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


## Calculate misro and macro F1 scores
def f1_score(y_true, y_pred):
    microF1 = []
    for instance in set(y_true).union(set(y_pred)):
        #print 'Digit=',instance

        # Calculate TP, TN, FP, and FN
        ind_orig = [k for k in range(len(y_true)) if y_true[k]==instance]
        ind_pred = [k for k in range(len(y_pred)) if y_pred[k]==instance]
        TP = len(set(ind_orig).intersection(set(ind_pred)))
        #print 'TP:', ind_orig, ind_pred, TP # DEBUG

        ind_orig = [k for k in range(len(y_true)) if y_true[k]!=instance]
        ind_pred = [k for k in range(len(y_pred)) if y_pred[k]!=instance]
        TN = len(set(ind_orig).intersection(set(ind_pred)))
        #print 'TN:', ind_orig, ind_pred, TN # DEBUG

        ind_orig = [k for k in range(len(y_true)) if y_true[k]!=instance]
        ind_pred = [k for k in range(len(y_pred)) if y_pred[k]==instance]
        FP = len(set(ind_orig).intersection(set(ind_pred)))
        #print 'FP:', ind_orig, ind_pred, FP # DEBUG

        ind_orig = [k for k in range(len(y_true)) if y_true[k]==instance]
        ind_pred = [k for k in range(len(y_pred)) if y_pred[k]!=instance]
        FN = len(set(ind_orig).intersection(set(ind_pred)))
        #print 'FN:', ind_orig, ind_pred, FN # DEBUG

        # Calculate micro F1 score
        if TP+FP+FN==0:
            F1 = 0.0
            print 'There is no instance of class',digit,'in true or predicted labels.'
            print 'Thus, micro F1 for class',digit,'is not included to calculate macro F1.'
        else:
            F1 = 2*float(TP)/float(2*TP+FP+FN)
            microF1.append(F1)
        #print 'micro F1:',F1

    macroF1 = np.mean(microF1)
    #print '\nmicro F1 score for each class:',microF1    
    #print 'macro F1 score:',macroF1

    return microF1, macroF1


## Sigmoid function (if wanting to use Sigmoid instead of linear threshold units)
def sigmoid(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return expit(x)



## Main script ##
if __name__ == '__main__':
    
    # Hyperparameters 
    nTrainToUse = int(sys.argv[1])
    nEpoch = int(sys.argv[2])
    learnRate = float(sys.argv[3])
    pathToDataFolder = sys.argv[4]
    #nTrainToUse = 1000
    #nEpoch = 10
    #learnRate = 0.001
    #pathToDataFolder = 'data'
    
    # Train 10 Perceptrons for all digits (0~9)
    #print '1. Train Perceptrons for all digits (0~9).'
    img, labl = load_data("training", pathToDataFolder, nTrainToUse)
    weights = np.empty((img.shape[1]+1,0), float) # (785x0)
    for digit in range(10):
        weightForOneDigit = averaged_perceptron(digit, img, labl, nEpoch, learnRate) # (785x1)
        weights = np.append(weights, weightForOneDigit, axis=1) # (785x10)

    # Test the Perceptrons with training set
    #print '\n2. Test the Perceptrons with training set.'
    y_true = labl.reshape(-1,1)[:,-1] # just reshaping to be (nSamples,)
    y_pred = test(img, weights) # (nSamples,)
    microF1, macroF1 = f1_score(y_true, y_pred)
    print 'Training F1 score: %.2f' % macroF1

    # Test the Perceptrons with test set
    #print '\n3. Test the Perceptrons with test set.'
    img, labl = load_data("test", pathToDataFolder)
    y_true = labl.reshape(-1,1)[:,-1] # just reshaping to be (nSamples,)
    y_pred = test(img, weights) # (nSamples,)
    microF1, macroF1 = f1_score(y_true, y_pred)
    print 'Test F1 score: %.2f' % macroF1

