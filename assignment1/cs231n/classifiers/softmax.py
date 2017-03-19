import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) # [10, 3073]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
      
  num_class = W.shape[0]  # W [10,3073]
  num_train = X.shape[1]  # X [3073, 500]

   #something wrong with the following code
  
  for i in xrange(num_train):
    
    #get the score by dot product for each training pic
    #X[:,i] indicates each col of  X
    scores = W.dot(X[:, i])
    correct_score = scores[y[i]]
    
    #numerical stability, so the highest value is 0
    scores -= np.max(scores)
    
    # Compute loss (and add to it, divided later)
    # L_i = - f(x_i)_{y_i} + log (sum_j e^{f(x_i)_j})
    sum = 0.0
    for s in scores:
        sum += np.exp(s)
    loss+= (-correct_score + np.log(sum))
    
    #compute gradient
    # dw_j = 1/num_train * sum[x_i * (p(y_i = j)-Ind{y_i = j} )]
    # Here we are computing the contribution to the inner sum for a given i.
    for n in xrange(num_class):
        p = np.exp(scores[n])/sum
        dW[n,:] += (p-(n == y[i])) * X[:, i]
    
  loss/=num_train
  dW/=num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW+=reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_class = W.shape[0]  # W [10,3073]
  num_train = X.shape[1]  # X [3073, 500]
   
  scores = np.dot(W,X)
  
  #numericall stability
  scores -= np.max(scores)
  
  correct_scores = scores[y, range(num_train)]
  
  # Loss: L_i = - f(x_i)_{y_i} + log \sum_j e^{f(x_i)_j}
  loss -= np.log(np.exp(correct_scores)/np.sum(np.exp(scores)))
  loss = np.mean(loss)
  loss/=num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  #compute gradient
  # dw_j = 1/num_train * sum[x_i * (p(y_i = j)-Ind{y_i = j} )]
  # the gradient for all correct labeled position is original weight -1
  # for incorrect labeled position is itself
  p = np.exp(scores)/np.sum(np.exp(scores),axis=0) #for every pics
  ind = np.zeros(p.shape)
  ind[y, range(num_train)] = 1 #the correct labeled positions are 1
  dW = np.dot((p-ind), X.T)
  dW/=num_train
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

