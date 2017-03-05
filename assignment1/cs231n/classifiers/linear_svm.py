import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
    
  #go through all 500 pics
  for i in xrange(num_train):
    #get the score by dot product
    scores = W.dot(X[:, i])
    
    #find the score at the correct labeled position
    correct_class_score = scores[y[i]]
    
    #for every pic, get the sum of margin of all incorrect predicted classes score and the ground truth class score
    for j in xrange(num_classes):
      if j == y[i]:  #skip for the true class
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #calculate gradient of loss function
        #sum over j != y[i]
        #dW[:,y[i]]-=X[i,:]
        #dW[:,j] += X[i,:] # sums each contribution of the x_i's
        
        # Compute gradients (one inner and one outer sum)
        
        #print dW[y[i],:].shape
        #print X[:,i].T.shape
        dW[y[i],:] -= X[:,i].T # this is really a sum over j != y_i
        dW[j,:] += X[:,i].T # sums each contribution of the x_i's
        
      


    

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  
  scores = W.dot(X) #now scores is [10, 500], shows the class scores of all pics
  
  #y is the corresponding right label
  #np.arange(num_train) go through each column which shows each picture
  #correct_scores is [500,]
  correct_scores = scores[y,np.arange(num_train)]
  diff_matrix = scores - correct_scores + 1
  #set the score of the corrected labeled to 0 because we don't want them in the total loss

  diff_matrix[y, np.arange(num_train)] = 0
  
  threshold_matrix = np.maximum(np.zeros((num_classes,num_train)),diff_matrix)
  total_loss = np.sum(threshold_matrix)
  total_loss = total_loss/num_train
    
  #add regularization to the total loss  
  total_loss += 0.5 * reg * np.sum(W * W)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #calculate gradient of the loss with respect of the correct labeled weight
    
  binary = threshold_matrix  #[10,500]
  binary[threshold_matrix>0] = 1 
  col_sum = np.sum(binary, axis=0)  #[500,]  

  #every corrected labeled score minus sum of each column in the binary matrix
  binary[y, np.arange(num_train)] -= col_sum[np.arange(num_train)]
  
  #[10,500] * [500, 3703] = [10, 3703]
  dW = np.dot(binary, X.T)

  #normalize and regularize 
  dW/=num_train
  dW+= reg*W

  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return total_loss, dW
