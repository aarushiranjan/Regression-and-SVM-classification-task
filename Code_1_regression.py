import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import minimize

def read_data():
    #reading the data for regression
    data=np.loadtxt("regression data.txt")
    X=data[:,0]
    y=data[:,1]
    return X, y

def poly_transform(X, M):
    # adding polynomial terms in the input data
    N=np.size(X)
    X_poly=np.zeros([N, M+1])
    for k in range(M+1):
        X_poly[:,k]=np.power(X, k)
    return X_poly

def obj_sos(w, X, y):
    #objective function for sum of squares error minimization
    J=np.sum(np.power(y-np.dot(X, w),2))
    return J

def obj_abe(w, X, y):
    #objective function for sum of absolute errors minimization
    J=np.sum(abs(y-np.dot(X, w)))
    return J

def reg_soln(X, y, l):
    #closed form solution for mean square error minimization with regularization parameter l
    #l=0 is equivalent to non-regularized regression
    t=np.transpose(X)
    n=t.shape[0]
    lM=l*np.identity(n)
    p=np.linalg.pinv(np.dot(t, X)+lM)
    w1=np.dot(p, t)
    w=np.dot(w1, y)
    return w


if __name__ == '__main__':
    #reading the data for regression
    X_data, y_data=read_data()
    #varying no. of data points
    '''num=20
    X_data=X_data[0:num]
    y_data=y_data[0:num]'''
    numTrain=np.size(X_data)
    #addition of polynomial terms into the input features
    M=5 #degree of polynomial
    X_polydata=poly_transform(X_data, M)
    #------------------------------------------------------------
    #Finding the optimal weights of the polynomial function assumed
    #--------------------------------------------------------------
    #1) optimal weights by error function minimization for sum of squares error
    '''res=minimize(obj_sos, x0=np.zeros([M+1,1]), args=(X_polydata, y_data,),)
    w_coeff=res.x #optimal coeffs'''
    #----------------------------------------------------------------
    #2) optimal weights by error function minimization for sum of absolute error
    '''res=minimize(obj_abe, x0=np.zeros([M+1,1]), args=(X_polydata, y_data,),)
    w_coeff=res.x'''
    #-------------------------------------------------------------------
    #closed form solution of regression (from linear regression)
    w_coeff=reg_soln(X_polydata, y_data, 0)
    #---------------------------------------------------------------------
    #Regularised linear regression
    '''l=10 #regularization parameter
    w_coeff=reg_soln(X_polydata, y_data, l)'''
    #-----------------------------------------------------------------------
    #Evaluation of the regressor model obtained
    #predictions
    predict_y=np.dot(X_polydata, w_coeff)
    #noise variance estimate
    diff=y_data-predict_y
    sigma=np.sum(np.power(diff, 2))/(numTrain-M-1) #noise variance estimate
    print("Variance estimate is %.2f" %sigma)
    #r2 score of the model and MSE
    r2score=r2_score(y_data, predict_y)
    print("r2 score for the regression model is %.2f" %r2score)
    mse=np.sum(np.power(diff, 2))/numTrain
    print("MSE is %.2f" %mse)
    print("Coefficients are:")
    print(w_coeff)
    #plotting the results
    plt.scatter(X_data, y_data)
    plt.plot(X_data, predict_y, color='r')
    plt.show()
    print("Done.......")
