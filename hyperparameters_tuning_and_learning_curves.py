

## Hyperparameter tuning by plotting learning curves

## 1. Learning curve: nTrainToUse

# Varying the size of training (500:250:10000); nEpoch=50; learningRate=0.001

nEpoch=50
learnRate=0.001
pathToDataFolder='data'

paramBegin=500
paramStep=250
paramEnd=10000
listParam = []

listTrainF1 = []
listTestF1 = []

for param in np.linspace(paramBegin,paramEnd,(paramEnd-paramBegin)/paramStep+1):
    nTrainToUse = int(param)
    listParam.append(nTrainToUse)
    
    ## Main script ##
    # Train 10 Perceptrons for all digits (0~9)
    img, labl = load_data("training", pathToDataFolder, nTrainToUse)
    weights = np.empty((img.shape[1]+1,0), float) # (785x0)
    for digit in range(10):
        weightForOneDigit = perceptron(digit, img, labl, nEpoch, learnRate) # (785x1)
        weights = np.append(weights, weightForOneDigit, axis=1) # (785x10)

    # Test the Perceptrons with training set
    y_true = labl.reshape(-1,1)[:,-1] # just reshaping to be (nSamples,)
    y_pred = test(img, weights) # (nSamples,)
    microF1, macroF1 = f1_score(y_true, y_pred)
    #print 'Training F1 score: %.2f' % macroF1
    listTrainF1.append(macroF1)
    
    # Test the Perceptrons with test set
    img, labl = load_data("test", pathToDataFolder)
    y_true = labl.reshape(-1,1)[:,-1] # just reshaping to be (nSamples,)
    y_pred = test(img, weights) # (nSamples,)
    microF1, macroF1 = f1_score(y_true, y_pred)
    #print 'Test F1 score: %.2f' % macroF1
    listTestF1.append(macroF1)    

print 'Param=',listParam
print 'Training_F1=',listTrainF1
print 'Test_F1=',listTestF1



## 2. Learning curve: nEpoch

# Varying the number of epoch (10:5:100); nTrain=10k; learningRate=0.001

nTrainToUse=10000
learnRate=0.001
pathToDataFolder='data'

paramBegin=10
paramStep=5
paramEnd=100
listParam = []

listTrainF1 = []
listTestF1 = []

for param in np.linspace(paramBegin,paramEnd,(paramEnd-paramBegin)/paramStep+1):
    nEpoch = int(param)
    listParam.append(nEpoch)
    
    ## Main script ##
    # Train 10 Perceptrons for all digits (0~9)
    img, labl = load_data("training", pathToDataFolder, nTrainToUse)
    weights = np.empty((img.shape[1]+1,0), float) # (785x0)
    for digit in range(10):
        weightForOneDigit = perceptron(digit, img, labl, nEpoch, learnRate) # (785x1)
        weights = np.append(weights, weightForOneDigit, axis=1) # (785x10)

    # Test the Perceptrons with training set
    y_true = labl.reshape(-1,1)[:,-1] # just reshaping to be (nSamples,)
    y_pred = test(img, weights) # (nSamples,)
    microF1, macroF1 = f1_score(y_true, y_pred)
    #print 'Training F1 score: %.2f' % macroF1
    listTrainF1.append(macroF1)
    
    # Test the Perceptrons with test set
    img, labl = load_data("test", pathToDataFolder)
    y_true = labl.reshape(-1,1)[:,-1] # just reshaping to be (nSamples,)
    y_pred = test(img, weights) # (nSamples,)
    microF1, macroF1 = f1_score(y_true, y_pred)
    #print 'Test F1 score: %.2f' % macroF1
    listTestF1.append(macroF1)    

print 'Param=',listParam
print 'Training_F1=',listTrainF1
print 'Test_F1=',listTestF1



## 3. Learning curve: learnRate

# Varying the learning rate (.0001,.001,.01,.1,1); nEpoch=50; nTrain=10k

nTrainToUse=10000
nEpoch=50
pathToDataFolder='data'

listParam = []
listTrainF1 = []
listTestF1 = []

for param in np.array([.0001,.001,.01,.1,1]):
    learnRate = float(param)
    listParam.append(learnRate)
    
    ## Main script ##
    # Train 10 Perceptrons for all digits (0~9)
    img, labl = load_data("training", pathToDataFolder, nTrainToUse)
    weights = np.empty((img.shape[1]+1,0), float) # (785x0)
    for digit in range(10):
        weightForOneDigit = perceptron(digit, img, labl, nEpoch, learnRate) # (785x1)
        weights = np.append(weights, weightForOneDigit, axis=1) # (785x10)

    # Test the Perceptrons with training set
    y_true = labl.reshape(-1,1)[:,-1] # just reshaping to be (nSamples,)
    y_pred = test(img, weights) # (nSamples,)
    microF1, macroF1 = f1_score(y_true, y_pred)
    #print 'Training F1 score: %.2f' % macroF1
    listTrainF1.append(macroF1)
    
    # Test the Perceptrons with test set
    img, labl = load_data("test", pathToDataFolder)
    y_true = labl.reshape(-1,1)[:,-1] # just reshaping to be (nSamples,)
    y_pred = test(img, weights) # (nSamples,)
    microF1, macroF1 = f1_score(y_true, y_pred)
    #print 'Test F1 score: %.2f' % macroF1
    listTestF1.append(macroF1)    

print 'Param=',listParam
print 'Training_F1=',listTrainF1
print 'Test_F1=',listTestF1
