import numpy as nup
import matplotlib
import matplotlib.pyplot as plter
import pandas as pd
import operator
import math
from array import array

# TO SHOW / CALCULATE GRAPHS JUST RUN PYTHON SCRIPT AND WAIT FOR RESULTS, CALCULATION TAKES A WHILE
# TO SHOW / CALCULATE GRAPHS JUST RUN PYTHON SCRIPT AND WAIT FOR RESULTS, CALCULATION TAKES A WHILE
# TO SHOW / CALCULATE GRAPHS JUST RUN PYTHON SCRIPT AND WAIT FOR RESULTS, CALCULATION TAKES A WHILE
# TO SHOW / CALCULATE GRAPHS JUST RUN PYTHON SCRIPT AND WAIT FOR RESULTS, CALCULATION TAKES A WHILE
# TO SHOW / CALCULATE GRAPHS JUST RUN PYTHON SCRIPT AND WAIT FOR RESULTS, CALCULATION TAKES A WHILE
# TO SHOW / CALCULATE GRAPHS JUST RUN PYTHON SCRIPT AND WAIT FOR RESULTS, CALCULATION TAKES A WHILE
# TO SHOW / CALCULATE GRAPHS JUST RUN PYTHON SCRIPT AND WAIT FOR RESULTS, CALCULATION TAKES A WHILE
# TO SHOW / CALCULATE GRAPHS JUST RUN PYTHON SCRIPT AND WAIT FOR RESULTS, CALCULATION TAKES A WHILE

dataTrain = nup.loadtxt("MNIST-Train-cropped.txt")
labelsTrain = nup.loadtxt("MNIST-Train-Labels-cropped.txt")
dataT = nup.loadtxt("MNIST-Test-cropped.txt")
labelsT = nup.loadtxt("MNIST-Test-Labels-cropped.txt")

dataTrain_arr = nup.reshape(dataTrain,(10000,784))
labelsTrain_arr = nup.reshape(dataT, (2000, 784))


def read_image(n):
    images_array = nup.reshape(dataTrain_arr[n],(28,28))
    return nup.transpose(images_array)

def show_image(n, arr):
    images_array = nup.reshape(arr[n][0],(28,28))
    images_array2 = nup.transpose(images_array)
    plter.show(plter.imshow(images_array2))

def get_mat(ns):
    ims, labs = zip(*ns)
    return (nup.array(nup.transpose(ims)), nup.array(labs))

def get_images():
    imgs = []
    for i in range (0,10000):
        img = dataTrain_arr[i]
        label = labelsTrain[i]
        imgs.append([img,label])
    return imgs

im1 = nup.array(list(zip(dataTrain_arr, labelsTrain)))
im2 = nup.array(list(zip(labelsTrain_arr, labelsT)))


def show_pair(n, arr):
    print(arr[n][1])
    show_image(n, arr)
    return 0


def get_sorted_label_pairs(training_mat, training_lab, point1):
    dist_mat = get_dist_mat(point1, training_mat)
    zipped = nup.array(list(zip(dist_mat, training_lab )),dtype=[('dist', float),('label',nup.int8)])
    return nup.sort(zipped, order='dist')

def get_dist_mat(point1, training_mat):
    rows, columns = training_mat.shape
    point_mat = nup.transpose([point1,]*columns)
    diff_mat = training_mat - point_mat
    squared_mat = diff_mat**2
    unit_arr = nup.array([1]*rows)
    dist_mat = nup.dot(unit_arr,squared_mat)
    return nup.sqrt(dist_mat)

def kNearestClassifier(x,y,value_to_predict,k=1):

    ones = np.ones(x.shape[0])
    data = np.array(x)

    subtractor = np.outer(ones,value_to_predict)

    matrix = data - subtractor

    dist_before_sq = np.diag(np.inner(matrix,matrix))
    dist = np.sqrt(dist_before_sq)

    sorter = np.argsort(dist)

    sorted_y = y[sorter]
    counts = np.bincount(sorted_y[:k])
    return np.argmax(counts)

def k_NN_eval(dataTrain, dataT):
    rows, cols = dataT.shape
    training_mat, training_lab = get_mat(dataTrain)
    error = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    for i in range (0,rows):
        pairs = get_sorted_label_pairs(training_mat, training_lab, dataT[i][0])
        _, votes = zip(*pairs)
        for j in range(0,17):
            kcounts = nup.bincount(votes[0:j*2+1])
            num = nup.argmax(kcounts)
            if num != dataT[i][1]:
                error[j] = error[j]+1 
    for k in range(0,len(error)):
        error[k] = error[k]*100/rows
    return error

def strip(n, arr):
    vals = []
    for img in arr:
        if img[1] == n:
            vals.append(img)
    return nup.array(vals)

zeroOne = nup.concatenate((strip(0, im1), strip(1, im1)))
fiveSix = nup.concatenate((strip(5, im1), strip(6, im1)))
zeroEight = nup.concatenate((strip(0, im1), strip(8, im1)))

def checkDataprep(Data):
  numFeatures = Data.shape[1]-1
  targetvalues = Data[:,numFeatures]
  features = nup.delete(Data, numFeatures, axis=1)
  return(targetvalues,features)

def standRegres(xArr,yArr):
  xMat = nup.mat(xArr); yMat = nup.mat(yArr).T
  xTx = xMat.T*xMat
  ws = xTx.I * (xMat.T*yMat)
  return ws

def MSE(X,Y,weights):
  error = 0
  numdata = len(X)
  yHat = X*weights
  for i in range(numdata):
    error += (yHat[i]-Y[0][i])**2
  return error / numdata

def linearRegs(fname):
  Data = nup.loadtxt(fname)
  print(Data)
  (Y,X) = checkDataprep(Data)
  newData = nup.ones([len(X),1])
  X =  X.reshape((len(X), 1))
  X = nup.concatenate((newData,X),axis=1)
  Y = Y.reshape(1,len(Y))
  weights = standRegres(X,Y)


  error = MSE(X,Y,weights)
  print (error)
  return (weights,error)

def plotLinRegression(fname):
    Data = nup.loadtxt("DanWood.dt")
    fig = plter.figure()
    ax = fig.add_subplot(111)
    (weights, error) = linearRegs("DanWood.dt")
    x = Data[:, 0]
    y = Data[:, 1]
    x = x.reshape((6, 1))
    y = y.reshape((6, 1))
    p1, = plter.plot(x, y, '.', label='Data')
    p2, = plter.plot(x, (1 * weights[0] + x * weights[1]), '-', label='Regression')
    plter.title("Linear Regression - Predicted Values and Data Points", color='crimson')
    plter.ylabel('Radiated energy')
    plter.xlabel('Absolute Temperature')
    plter.legend(handles=[p1, p2], loc=2)
    plter.show()

def percentSplit(arr):
    rows, cols = arr.shape
    num = nup.floor(0.8*rows)
    return arr[0:int(num)], arr[int(num):rows]

def main():
  weights,error = linearRegs("DanWood.dt")
  print (str(weights[0]) + " " + str(weights[1]))
  print (str(error))
  plotLinRegression("DanWood.dt")

  train_0_1, val_0_1 = percentSplit(zeroOne)
  train_0_8, val_0_8 = percentSplit(zeroEight)
  train_5_6, val_5_6 = percentSplit(fiveSix)

  evaluateZeroEight = k_NN_eval(train_0_8, val_0_8)
  evaluateZeroOne = k_NN_eval(train_0_1, val_0_1)
  evaluateFiveSix = k_NN_eval(train_5_6, val_5_6)

  odd = [x for x in range(0, 34) if x % 2 == 1]

  test01 = nup.concatenate((strip(0, im2), strip(1, im2)))
  test08 = nup.concatenate((strip(0, im2), strip(8, im2)))
  test56 = nup.concatenate((strip(5, im2), strip(6, im2)))

  testZeroOne = k_NN_eval(zeroOne, test01)
  testZeroEight = k_NN_eval(zeroEight, test08)
  testFiveSix = k_NN_eval(fiveSix, test56)

  plter.figure(1)
  plter.axis([1, 33, -5, 9])
  plter.plot(odd, evaluateZeroOne, label="Validation error", color='indigo')
  plter.plot(odd, testZeroOne, label="Testing error", color='orange')
  plter.title("0 & 1", color='crimson')
  plter.xlabel("# of neighbours", color='indigo')
  plter.ylabel("% Error", color='orange')
  plter.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plter.gcf().transFigure)

  plter.figure(2)
  plter.axis([1, 33, -5, 9])
  plter.plot(odd, evaluateFiveSix, label="Validation error", color='indigo')
  plter.plot(odd, testFiveSix, label="Testing error", color='orange')
  plter.title("5 & 6", color='crimson')
  plter.xlabel("# of neighbours", color='indigo')
  plter.ylabel("% Error", color='orange')
  plter.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plter.gcf().transFigure)

  plter.figure(3)
  plter.axis([1, 33, -5, 9])
  plter.plot(odd, evaluateZeroEight, label="Validation error", color='indigo')
  plter.plot(odd, testZeroEight, label="Testing error", color='orange')
  plter.title("0 & 8", color='crimson')
  plter.xlabel("# of neighbours", color='indigo')
  plter.ylabel("% Error", color='orange')
  plter.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plter.gcf().transFigure)

  plter.show()

main()


