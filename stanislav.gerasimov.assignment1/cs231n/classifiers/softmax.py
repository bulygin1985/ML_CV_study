from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros_like(W)

    N = X.shape[0]
    C = W.shape[1]

    for i in range(N):
        # Compute the scores
        scores = X[i].dot(W)
        
        # Stabilize the scores for numerical stability
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)
        
        # Compute the softmax probabilities
        probs = exp_scores / np.sum(exp_scores)
        
        # Compute the loss
        loss += -np.log(probs[y[i]])
        
        # Compute the gradient
        for j in range(C):
            if j == y[i]:
                dW[:, j] += (probs[j] - 1) * X[i]
            else:
                dW[:, j] += probs[j] * X[i]

    # Average loss and gradient
    loss /= N
    dW /= N

    # Add regularization to the loss
    loss += reg * np.sum(W * W)

    # Add regularization to the gradient
    dW += 2 * reg * W

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    # Compute the scores
    scores = X.dot(W)
   
     # Shift the scores for numerical stability
    scores -= np.max(scores, axis=1, keepdims=True)

    # Compute the class probabilities
    softmax_output = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    # Compute the loss: average cross-entropy loss and regularization
    correct_class_probabilities = softmax_output[range(X.shape[0]), y]
    loss = -np.sum(np.log(correct_class_probabilities)) / X.shape[0]
    loss += 0.5 * reg * np.sum(W * W)

    # Compute the gradient on scores
    softmax_output[range(X.shape[0]), y] -= 1
    dW = X.T.dot(softmax_output) / X.shape[0]
    dW += reg * W

    return loss, dW

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

   

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
