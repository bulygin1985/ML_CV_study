# Softmax naive loss function

## Input data:

### Given
* 2 examples (Q)
* 3 data points (M)
* 3 features (N)
* 3 classes (C)


$$
W = 
\begin{bmatrix}
0.1 & 0.2 & 0.3 \\[0.3em]
0.4 & 0.5 & 0.6\\[0.3em]
0.7 & 0.8 & 0.9
\end{bmatrix}
\newline \\[0.6em]
W \in \mathbb{R}^{N \times C}
$$

$$
X = 
\begin{bmatrix}
  \begin{bmatrix}
  1 & 2 & 3 \\[0.3em]
  4 & 5 & 6 \\[0.3em]
  7 & 8 & 9
  \end{bmatrix}
  \begin{bmatrix}
  11 & 12 & 13 \\[0.3em]
  14 & 15 & 16 \\[0.3em]
  17 & 18 & 19
  \end{bmatrix}
\end{bmatrix}
$$


$$
D = M \times N
\newline \\[0.6em]
X \in \mathbb{R}^{D \times Q}
$$

$$
y = 
\begin{bmatrix}
0 \\[0.3em]
1 \\[0.3em]
2
\end{bmatrix}
\newline \\[0.6em]
y \in \mathbb{R}^{C}
$$

Let's take only the first example, i = 1

$$
X_i = 
\begin{bmatrix}
1 & 2 & 3 \\[0.3em]
4 & 5 & 6 \\[0.3em]
7 & 8 & 9
\end{bmatrix}
\newline \\[0.6em]
X_i \in \mathbb{R}^{D}
$$


$$
y_i = 0 \\[0.3em]
$$

## Step 1. Find and normalize scores

### Matrix multiplication

$$
S_i = f(x_i, W)= X_i W
\newline \\[0.6em]
S_i \in \mathbb{R}^{M \times C}
$$


Formula to find one element ($s$) for *i* row and *j* column:

$$ s_{ij} = \sum_{k=1}^{N}x_{ik} \cdot w_{kj} $$

$$
S_i = 
\begin{bmatrix}
1 & 2 & 3 \\[0.3em]
4 & 5 & 6 \\[0.3em]
7 & 8 & 9
\end{bmatrix}
\cdot
\begin{bmatrix}
0.1 & 0.2 & 0.3 \\[0.3em]
0.4 & 0.5 & 0.6 \\[0.3em]
0.7 & 0.8 & 0.9
\end{bmatrix}
= 
\newline \\[0.6em]
= \begin{bmatrix}
1*0.1+2*0.4+3*0.7 & 1*0.2+2*0.5+3*0.8 & 1*0.3+2*0.6+3*0.9 \\[0.6em]
4*0.1+5*0.4+6*0.7 & 4*0.2+5*0.5+6*0.8 & 4*0.3+5*0.6+6*0.9 \\[0.6em]
7*0.1+8*0.4+9*0.7 & 7*0.2+8*0.5+9*0.8 & 7*0.3+8*0.6+9*0.9
\end{bmatrix}
=
\newline \\[0.6em]
= \begin{bmatrix}
3 & 3.6 & 4.2 \\[0.3em]
6.6 & 8.1 & 9.6 \\[0.3em]
10.2 & 12.6 & 15
\end{bmatrix}
$$

### Normalization

$$
\hat{S_i} = S_i - max(S_i)
\newline \\[0.3em]
max(S_i) = 15
\newline \\[0.6em]
\hat{S_i} = 
\begin{bmatrix}
-12  & -8.4 & -4.8 \\[0.3em]
-11.4 & -6.9 & -2.4 \\[0.3em]
-10.8 & -5.4 & 0
\end{bmatrix}
$$

## Step 2. Find loss function
Probabilities calculation


$$

$$

Loss function calculation

$$

$$


## Step 3. Gradient evaluation
Get the gradient


Evaluate the gradient


## Step 4. Apply regularization

$$
L = \frac{1}{N} \sum_{i=1}^{N} L_i + \lambda R(W)
\newline \\[0.6em]
R(W) = \sum_k \sum_l W_{kl}^2
$$
