import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def sort_data(X, Y):
    sorted_data = []
    for j in range(max(Y) + 1):
        
        class_i = []
        for i in range(len(Y)):
            if (Y[i] == j):
                class_i.append(X[i])

        sorted_data.append(class_i)
            
    return np.array(sorted_data)


def means(data):
    means_i = []
    
    for class_i in data:
        means_i.append(np.average(class_i, axis=0))

    data = data.reshape(-1, data.shape[-1])
    total_mean = np.average(data, axis=0)
        
    return means_i, total_mean


def within_scatter(data, means):
    total = np.zeros([len(data[0][0]), len(data[0][0])])

    for i in range(len(data)):
        for x in data[i]:
            total = total + np.outer((x - means[i]),(x - means[i]))

    return total
    

def between_scatter(data, means, total_mean):
    total = np.zeros([len(data[0][0]), len(data[0][0])])

    for i in range(len(data)):
        total = total + len(data[i][0]) * np.outer((means[i] - total_mean),(means[i] - total_mean))

    return total


def project(X, w):
    y = []
    for x in X:
        y.append(np.dot(w, x))

    return np.array(y)


def generate_W(data, ws):
    W = []
    for i in range(len(data) - 1):
        W.append(ws[:,i])

    return W


fig = plt.figure(1)

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

#loading the dataset from sklearn
iris = datasets.load_iris()

features = iris.data
classes = iris.target

#sorting data classwise
data = sort_data(features, classes)

#finding means of individual classes and the total mean
class_means, total_mean = means(data)

#generate within scatter and between scatter matrices
sw = within_scatter(data, class_means)
sb = between_scatter(data, class_means, total_mean)

sw_inv_sb = np.dot(np.linalg.inv(sw), sb)

#get eigenvalues and eigenvectors of sw_inv_sw
lambdas, ws = np.linalg.eig(sw_inv_sb)

W = generate_W(data, ws.real)

#projection onto largest eigenvectors in descending order
y1 = project(features, W[0])
y2 = project(features, W[1])

ax1.set_title("1d projection")
ax2.set_title("2d projection")

ax1.scatter(y1, np.zeros_like(y1), s=10, c=classes)
ax2.scatter(y1, y2, s=10, c=classes)


plt.show()



