import numpy as np 
import matplotlib.pyplot as plt
#Load data
import scipy.io as sio
import numpy as np


a = sio.loadmat('mnist_binary.mat')
X_trn = a['X_trn']
X_tst = a['X_tst']
Y_trn = a['Y_trn'][0]
Y_tst = a['Y_tst'][0]
print(X_trn.shape)
print(X_tst.shape)
print(Y_trn.shape)
print(Y_tst.shape)


#Initialize parameters 
num_hidden = 20 #number of neurons in the hidden layer
num_op = 2 #number of neurons in the output layer

def initialize_parameters(size_input, size_hidden, size_output):
    np.random.seed(2)
    W1 = np.random.randn(size_hidden, size_input) * 0.01
    b1 = np.zeros(shape=(size_hidden, 1))
    W2 = np.random.randn(size_output, size_hidden) * 0.01
    b2 = np.zeros(shape=(size_output, 1))
    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}
    return parameters
parameters = initialize_parameters(X_trn.shape[0], num_hidden, num_op)
print('W1',parameters['W1'].shape)
print('b1',parameters['b1'].shape)
print('W2',parameters['W2'].shape)
print('b2',parameters['b2'].shape)

def softmax(Z2):
    # ip - (M,N) array where M is no. of neurons in output layer, N is number of samples.
    # You can modify the code if your output layer is of different dimension
   # =========Write your code below ==============
    n = Z2.shape[1]
    softmax = np.zeros(shape=(1, n))
    def sm(x,y):
        return np.exp(x)/(np.exp(x)+np.exp(y))
    
    smfunc = np.vectorize(sm)
    softmax[0] = smfunc(Z2[0],Z2[1])
    # =============================================
    assert(softmax.shape == (1, Z2.shape[1]))
    return softmax

def activ(ip,act):
    # ip - array obtained after multiplying inputs with weights (between input layer and hidden layer)
    # act - ReLU or Sigmoid
    # I am assuming that "ip" already includes the bias terms, since the bias terms were not separately passed as a parameter
    out = np.zeros(shape=ip.shape)
    
    def m(x):
        return np.maximum(x,0)
    def s(x):
        return 1.0/(1+np.exp(-1*x))
    
    if act =="ReLU":
        # =========Write your code below ==============
        f = np.vectorize(m)
        out = m(ip)

    # =============================================
    elif act == "Sigmoid":
        # =========Write your code below ==============
        f = np.vectorize(s)
        out = f(ip)
                

    # =============================================
    assert(out.shape == ip.shape)
    return out

#Forward Propagation   
def forward_propagation(X, parameters, act):
# =========Write your code below ==============
    #Z1 = W1*X+b1
    Z1 = parameters['W1'].dot(X) + parameters['b1']
    A1 = activ(Z1,act)
    Z2 = parameters['W2'].dot(A1) + parameters['b2']
    A2 = softmax(Z2)
    # =============================================
    assert(A2.shape == (1, X.shape[1]))
    
    neuron = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return neuron

def backprop(parameters, neuron, X, Y, act):
# =========Write your code below ==============
    # neuron = {"Z1": Z1,
    #         "A1": A1,
    #         "Z2": Z2,
    #         "A2": A2}
    # logistic function
    def sigm(x):
        return 1.0/(1+np.exp(-1*x))
    def sigmaderiv(x):
        return (1.0/(1+np.exp(-1*x)))*(1-1.0/(1+np.exp(-1*x)))
    # heaviside function, derivative of the ReLU activation function
    def heavy(x):
        return np.heaviside(x, 0)
    
    Z1 = neuron["Z1"]
    Z2 = neuron["Z2"]
    A1 = neuron["A1"]
    A2 = neuron["A2"]
    W2 = parameters['W2']
    W1 = parameters['W1']
    dldz2 = np.zeros(shape=Z2.shape)
    
    def dldz2top(x,y):
        return x - y
    def dldz2bottom(x,y):
        return y - x
    topfunc = np.vectorize(dldz2top)
    bottomfunc = np.vectorize(dldz2bottom)
    top = topfunc(A2[0],Y)
    bottom = bottomfunc(A2[0],Y)
    dldz2 = np.vstack((top,bottom))

    # gradients computation
    dW2 = dldz2.dot(A1.transpose())
    db2 = dldz2.dot(np.ones(shape=(X.shape[1],1)))
    # da1dz1 will depend on the activation function
    if act == "Sigmoid":
        da1dz1 = np.zeros(shape=Z1.shape)
        s = np.vectorize(sigmaderiv)
        da1dz1 = s(Z1)
        dW1 = (np.multiply((((dldz2.transpose()).dot(W2)).transpose()),da1dz1)).dot(X.transpose())
        db1 = (np.multiply((((dldz2.transpose()).dot(W2)).transpose()),da1dz1)).dot(np.ones(shape=(X.shape[1],1)))
    if act == "ReLU":
        da1dz1 = np.zeros(shape=Z1.shape)
        g = np.vectorize(heavy)
        da1dz1 = g(Z1)
        dW1 = (np.multiply((((dldz2.transpose()).dot(W2)).transpose()),da1dz1)).dot(X.transpose())
        db1 = (np.multiply((((dldz2.transpose()).dot(W2)).transpose()),da1dz1)).dot(np.ones(shape=(X.shape[1],1)))
    # =============================================
    
    assert(dW1.shape == W1.shape)
    assert(dW2.shape == W2.shape)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads



def cross_entropy_loss(softmax, Y):
# =========Write your code below ==============
    #L(y,y') = -ylog(y') - (1-y)log(1-y')
    loss = np.zeros(shape=Y.shape)
    def entropy(y,yhat):
        return -y*np.log(yhat) - (1-y)*np.log(1-yhat)
    e = np.vectorize(entropy)
    loss = e(Y,softmax[0])
# =============================================        
    assert(loss.shape == Y.shape)
    return loss


def update_parameters(parameters, grads, learning_rate):

# =========Write your code below ==============
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    
    W1 = W1 - learning_rate*grads['dW1']
    W2 = W2 - learning_rate*grads['dW2']
    b1 = b1 - learning_rate*grads['db1']
    b2 = b2 - learning_rate*grads['db2']

# =============================================

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

from sklearn.metrics import accuracy_score
def nn_model1(X_trn, X_tst, Y_trn, Y_tst, n_h, n_o, epochs, act, learning_rate):
    #X_trn: the training set
    #X_tst: the test set
    #Y_trn: training labels
    #Y_tst: test labels
    #n_h: number of neurons in the hidden layer
    #n_o: number of neurons in the output layer
    #epochs: number of epochs for the training
    #act: the activation function you choose
    #learning_rate: a list of length epochs, which consists of the learning rate in each step
    
    def predict(Y, x, parameters, act):
        pred = np.zeros(shape=Y.shape)
        z1 = parameters['W1'].dot(x) + parameters['b1']
        a1 = activ(z1,act)
        z2 = parameters['W2'].dot(a1) + parameters['b2']
        a2 = np.zeros(shape=(1, z2.shape[1]))                  
        for i in range(0,z2.shape[1]):
            a2[0][i] = np.exp(z2[0][i])/(np.exp(z2[0][i])+np.exp(z2[1][i]))
            
        for i in range(0,len(a2[0])):
            if a2[0][i] >= 0.5:
                pred[i] = 1
            else:
                pred[i] = 0
        return pred
    
    assert(len(learning_rate) == epochs)
    err_tst = []
    err_trn = []
    loss_trn = []
    parameters = initialize_parameters(X_trn.shape[0], n_h, n_o)
    
   # =========Write your code below ==============
    for i in range(0,epochs):
        print(i)
        neuron = forward_propagation(X_trn, parameters, act)
        loss = np.sum(cross_entropy_loss(neuron["A2"], Y_trn))/len(Y_trn)
        loss_trn.append(loss)
        grads = backprop(parameters, neuron, X_trn, Y_trn, act)
        parameters = update_parameters(parameters, grads, learning_rate[i])
        trn_pred = predict(Y_trn,X_trn,parameters,act)
        tst_pred = predict(Y_tst,X_tst,parameters,act)
        print("Train error: ")
        print(1-accuracy_score(Y_trn, trn_pred))
        print("Test error: ")
        print(1-accuracy_score(Y_tst, tst_pred))
        err_trn.append(1-accuracy_score(Y_trn, trn_pred))
        err_tst.append(1-accuracy_score(Y_tst, tst_pred))
        print("#################################")
        
    # =============================================    
    #err_tst: testing error (classification error) in each epoch
    #err_trn: training error (classification error) in each epoch
    #loss_trn: training loss (cross entropy loss) in each epoch
    #parameters: the final learned parameters
    return err_tst, err_trn, loss_trn, parameters

epochs = 20000
lr1 = 0.01/964*np.ones(epochs)
# =========Write your code below ==============
err_tst, err_trn, loss_trn, parameters = nn_model1(X_trn, X_tst, Y_trn, Y_tst, 20, 2, epochs, "ReLU", lr1)
# =============================================
plt.figure(1, figsize=(12, 8))
plt.plot(range(epochs), loss_trn, '-', color='orange',linewidth=2, label='training loss (lr = 0.01/964)')
plt.title('Training loss')
plt.xlabel('epoch')
plt.ylabel('Cross entropy error')
plt.legend(loc='best')
plt.grid()
plt.show()


epochs = 20000
lr1 = 0.01/964*np.ones(epochs)
# =========Write your code below ==============
err_tst, err_trn, loss_trn, parameters = nn_model1(X_trn, X_tst, Y_trn, Y_tst, 20, 2, epochs, "ReLU", lr1)

# =============================================
plt.figure(1, figsize=(12, 8))
plt.plot(range(epochs), err_trn, '-', color='orange',linewidth=2, label='training error (lr = 0.01/964)')
plt.plot(range(epochs), err_tst, '-b', linewidth=2, label='test error (lr = 0.01/964)')


plt.title('ReLU(Learning rate=0.01/964)')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend(loc='best')
plt.grid()
plt.show()

epochs = 20000
lr2 = 0.1/964*np.ones(epochs)
# =========Write your code below ==============
err_tst2, err_trn2, loss_trn2, parameters2 = nn_model1(X_trn, X_tst, Y_trn, Y_tst, 20, 2, epochs, "ReLU", lr2)
# =============================================
plt.figure(2, figsize=(12, 8))
# Classification errors for learning rate = 0.01/964, Relu Activation, n_hidden = 20
plt.plot(range(epochs), err_trn, '-', color='orange',linewidth=2, label='training error (lr = 0.01/964)')
plt.plot(range(epochs), err_tst, '-b', linewidth=2, label='test error (lr = 0.01/964)')

# Classification errors for learning rate = 0.1/964, Relu Activation, n_hidden = 20
plt.plot(range(epochs), err_trn2, '-', linewidth=2, label='training error (lr = 0.1/964)')
plt.plot(range(epochs), err_tst2, '-b', color='yellow', linewidth=2,  label='test error (lr = 0.1/964)')

plt.title('ReLU(Learning rate=0.1/964)')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend(loc='best')
plt.grid()
plt.show()

indices = np.array(range(epochs))
lr3 = 1/np.sqrt(indices + 1)*(1.0/964)
# =========Write your code below ==============
err_tst3, err_trn3, loss_trn3, parameters3 = nn_model1(X_trn, X_tst, Y_trn, Y_tst, 20, 2, epochs, "ReLU", lr3)
# =============================================
plt.figure(3, figsize=(12, 8))
# Classification errors for learning rate = 0.01/964, Relu Activation, n_hidden = 20
plt.plot(range(epochs), err_trn, '-', color='orange',linewidth=2, label='training error (lr = 0.01/964)')
plt.plot(range(epochs), err_tst, '-b', linewidth=2, label='test error (lr = 0.01/964)')

# Classification errors for learning rate = 0.1/964, Relu Activation, n_hidden = 20
plt.plot(range(epochs), err_trn2, '-', color='red', linewidth=2, label='training error (lr = 0.1/964)')
plt.plot(range(epochs), err_tst2, '-b', color='yellow', linewidth=2, label='test error (lr = 0.1/964)')

# Classification errors for variable learning rate, Relu Activation, n_hidden = 20
plt.plot(range(epochs), err_trn3, '-', color='purple', linewidth=2, label='training error (unfixed lr)')
plt.plot(range(epochs), err_tst3, '-b', color='green', linewidth=2, label='test error (unfixed lr)')
plt.title('ReLU(Variable learning rate)')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend(loc='best')
plt.grid()
plt.show()

num_hidden2 = 50
# =========Write your code below ==============
err_tst4, err_trn4, loss_trn4, parameters4 = nn_model1(X_trn, X_tst, Y_trn, Y_tst, num_hidden2, 2, epochs, "ReLU", lr1)

# =============================================
plt.figure(4, figsize=(12, 8))
# Classification errors for learning rate = 0.01/964, Relu Activation, n_hidden = 20
plt.plot(range(epochs), err_trn, '-', color='orange',linewidth=2, label='training error (#hidden = 20)')
plt.plot(range(epochs), err_tst, '-b', linewidth=2, label='test error (#hidden = 20)')

# Classification errors for learning rate = 0.01/964, Relu Activation, n_hidden = 50
plt.plot(range(epochs), err_trn4, '-', color='red', linewidth=2, label='training error (#hidden = 50)')
plt.plot(range(epochs), err_tst4, '-b', color='grey', linewidth=2, label='test error (#hidden = 50)')

plt.title('ReLU(Learning rate=0.01/964)')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend(loc='best')
plt.grid()
plt.show()

# =========Write your code below ==============
epochs = 20000
lr1 = 0.01/964*np.ones(epochs)
err_tst5, err_trn5, loss_trn5, parameters5 = nn_model1(X_trn, X_tst, Y_trn, Y_tst, 20, 2, epochs, "Sigmoid", lr1)

# =============================================
# Classification errors for learning rate = 0.01/964, Relu Activation, n_hidden = 20
plt.figure(5, figsize=(12, 8))
plt.plot(range(epochs), err_trn, '-', color='orange',linewidth=2, label='training error (ReLU)')
plt.plot(range(epochs), err_tst, '-b', linewidth=2, label='test error (ReLU)')

# Classification errors for learning rate = 0.01/964, Sigmoid Activation, n_hidden = 20
plt.plot(range(epochs), err_trn5, '-', color='red',  linewidth=2, label='training error (Sigmoid)')
plt.plot(range(epochs), err_tst5, '-b', color='green', linewidth=2, label='test error (Sigmoid)')

plt.title('Learning rate=0.01/964')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend(loc='best')
plt.grid()
plt.show()
