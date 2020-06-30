import os
import numpy as np
import theta as theta
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
# X, y = data[:, 0], data[:, 1]
# m = y.size
# X = np.stack([np.ones(m), X], axis=1)
#
#
# # print(X,X.size)
# def plotData(x, y):
#     pyplot.plot(x, y, 'ro', ms=10, mec='k')
#     pyplot.ylabel('Profit in $10,000')
#     pyplot.xlabel('Population of City in 10,000s')
#     # pyplot.show()
#
#
def computeCost(x, y, theta):
    m = y.size

    summationpart = 0
    for i in range(m):
        hypothesis = 0

        for j in range(x.shape[1]):  # this loops is to set linear hypothesis for any linear
            hypothesis = hypothesis + theta[j] * (x[i, j])**j

        summationpart = summationpart + (hypothesis - y[i]) ** 2

    J = summationpart / (2 * m)
    return J
#
#
# # J = computeCost(X, y, theta=np.array([-1, 2]))
# # print('With theta = [-1, 2]\\nCost computed = %.2f' % J)
# # print('Expected cost value (approximately) 54.24')
#
def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        alphabym = alpha / m
        sumofh0x = np.dot(X, theta)
        theta = theta - (alphabym * (np.dot(X.T, sumofh0x - y)))
        J_history.append(computeCost(X, y, theta))
    return theta, J_history
# theta = np.zeros(2)
# iterations = 1500
# alpha = 0.01
# theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
# print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
# print('Expected theta values (approximately): [-3.6303, 1.1664]')
# plotData(X[:, 1], y)
# pyplot.plot(X[:, 1], np.dot(X, theta), '-')
# pyplot.legend(['Training data', 'Linear regression']);
# pyplot.show()
# predict1 = np.dot([1, 3.5], theta)
# print('For population = 35,000, we predict a profit of {:.2f}\\n'.format(predict1*10000))
# predict2 = np.dot([1, 7], theta)
# print('For population = 70,000, we predict a profit of {:.2f}\\n'.format(predict2*10000))
# theta0_vals = np.linspace(-10, 10, 100)
# theta1_vals = np.linspace(-1, 4, 100)
# J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))
# for i, theta0 in enumerate(theta0_vals):
#     for j, theta1 in enumerate(theta1_vals):
#         J_vals[i, j] = computeCost(X, y, [theta0, theta1])
#
#     J_vals = J_vals.T
# fig = pyplot.figure(figsize=(12, 5))
# ax = fig.add_subplot(121, projection='3d')
# ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
# pyplot.xlabel('theta0')
# pyplot.ylabel('theta1')
# pyplot.title('Surface')
#
# ax = pyplot.subplot(122)
# pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
# pyplot.xlabel('theta0')
# pyplot.ylabel('theta1')
# pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
# pyplot.title('Contour, showing minimum')
# pyplot.show()



################################ Part2 ################################################
data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size

# print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
# print('-'*26)
# for i in range(10):
#     print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))





def  featureNormalize(x):
   X_norm = x.copy()
   mu = np.zeros(X.shape[1])
   sigma = np.zeros(X.shape[1])
   for j in range (x.shape[1]):
       meam= np.mean(x[:,j])
       mu[j]=meam

       standardev=np.std(x[:,j])
       sigma[j]=standardev
       for i in range (x.shape[0]):
           X_norm[i,j]= (x[i,j] - meam)/standardev
   return X_norm,mu,sigma
   return X_norm,mu,sigma
   return X_norm,mu,sigma

# X_norm, mu, sigma = featureNormalize(X)
# print('Computed mean:', mu)
# print('Computed standard deviation:', sigma)
# X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)


