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
b = 
\begin{bmatrix}
6 \\[0.3em]
1 \\[0.3em]
9
\end{bmatrix}
\newline \\[0.6em]
b \in \mathbb{R}^{C}
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

Let's take only the first example, q = 1

$$
X_1 = 
\begin{bmatrix}
1 & 2 & 3 \\[0.3em]
4 & 5 & 6 \\[0.3em]
7 & 8 & 9
\end{bmatrix}
\newline \\[0.6em]
X_q \in \mathbb{R}^{D}
$$


$$
y_1 = 0 \\[0.3em]
$$

## Step 1. Find and normalize scores

### Matrix multiplication

$$
S_q = f(x_q, W)= X_q W + b
\newline \\[0.6em]
S_q \in \mathbb{R}^{M \times C}
\newline \\[0.6em]
$$


Formula to find one element ($e$) for *i* row and *j* column in matrix multiplication:

$$ e_{ij} = \sum_{n=1}^{N}x_{in} \cdot w_{nj} $$

$$
S_1 = 
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
+
\begin{bmatrix}
6 \\[0.3em]
1 \\[0.3em]
9
\end{bmatrix}
= 
\newline \\[0.6em]
= \begin{bmatrix}
1*0.1+2*0.4+3*0.7 & 1*0.2+2*0.5+3*0.8 & 1*0.3+2*0.6+3*0.9 \\[0.6em]
4*0.1+5*0.4+6*0.7 & 4*0.2+5*0.5+6*0.8 & 4*0.3+5*0.6+6*0.9 \\[0.6em]
7*0.1+8*0.4+9*0.7 & 7*0.2+8*0.5+9*0.8 & 7*0.3+8*0.6+9*0.9
\end{bmatrix}
+
\begin{bmatrix}
6 \\[0.3em]
1 \\[0.3em]
9
\end{bmatrix}
=
\newline \\[0.6em]
= \begin{bmatrix}
3 & 3.6 & 4.2 \\[0.3em]
6.6 & 8.1 & 9.6 \\[0.3em]
10.2 & 12.6 & 15
\end{bmatrix}
+
\begin{bmatrix}
6 \\[0.3em]
1 \\[0.3em]
9
\end{bmatrix}
=
\begin{bmatrix}
9 & 9.6 & 10.2 \\[0.3em]
7.6 & 9.1 & 10.6 \\[0.3em]
19.2 & 21.6 & 24
\end{bmatrix}
$$

### Normalization

$$
\hat{S_q} = S_q - max(S_q)
\newline \\[0.3em]
max(S_1) = 24
\newline \\[0.6em]
\hat{S_1} = 
\begin{bmatrix}
-15  & -14.4 & -13.8 \\[0.3em]
-16.4 & -14.9 & -13.4 \\[0.3em]
-4.8 & -2.4 & 0
\end{bmatrix}
$$

## Step 2. Find loss function
Probabilities calculation

$$
\Large
p_{mj} (\hat{S}) = \dfrac{e^{\hat{S}_{mj}}}{\sum_{c=1}^C e^{\hat{S}_{mc}}} 
\newline \\[0.6em]
\normalsize
m \in M,\; j \in C
$$

$$
e^{\hat{S}} = 
\begin{bmatrix}
e^{-15}  & e^{-14.4} & e^{-13.8} \\[0.3em]
e^{-16.4} & e^{-14.9} & e^{-13.4} \\[0.3em]
e^{-4.8} & e^{-2.4} & e^0
\end{bmatrix}
=
\newline \\[0.6em]
= \begin{bmatrix}
3.059e-7 & 5.574e-7 & 1.016e-6 \\[0.3em]
7.543e-8 & 3.381e-7 & 1.515e-6 \\[0.3em]
8.23e-3 & 9.072e-2 & 1
\end{bmatrix}
$$

$$
p = 
\begin{bmatrix}
0.16280717 & 0.296654   & 0.54053883 \\[0.3em]
0.03911257 & 0.17529039 & 0.78559703  \\[0.3em]
0.00748875 & 0.08254984 & 0.90996141
\end{bmatrix}
$$

Loss function calculation

$$
L = - log(p_j) \text{,\; where $j$ is an index of the correct class}
$$

## Step 3. Find gradient
Get the gradient

$$
L = - log(p_j) \text{,\; where $j$ is an index of the correct class}
\newline \\[0.6em]
\nabla_{z_j} L = \dfrac{\delta L}{\delta p_i} \cdot \dfrac{\delta p_i}{\delta z_j}
\newline \\[0.6em]
\dfrac{\delta L}{\delta p_i} = - \dfrac {1}{p_i}
$$

$$
\large
\dfrac{\delta p_i}{\delta z_j} =
\dfrac{\delta}{\delta z_j} (\dfrac{e^{z_i}}{\sum_{c=1}^C e^{z_c}})
$$

### If $i=j$:

Use quotient rule

$$
\large
\dfrac{\delta p_i}{\delta z_i} = 
\dfrac{e^{z_i} \sum_{c=1}^C e^{z_c} - e^{z_i} e^{z_i}}
{(\sum_{c=1}^C e^{z_c})^2} = 
\newline \\[0.6em]
= p_i (\dfrac{\sum_{c=1}^C e^{z_c} - e^{z_i}}{\sum_{c=1}^C e^{z_c}}) = p_i (1 - p_i)
$$

### If $i \neq j$:

Use quotient rule

$$
\large
\dfrac{\delta p_i}{\delta z_j} = 
\dfrac{-e^{z_i} e^{z_j}}{(\sum_{c=1}^C e^{z_c})^2} = 
-p_i p_j
\newline \\[1.5em]
$$

$$
\nabla_{z_j} L = - \dfrac {1}{p_i} \cdot p_i (Y_{j} - p_j) = p_j - Y_{j}
\newline \\[0.6em]
j \in C, Y_{j} \in \{0,1\}
$$

Gradient calculation


## Step 4. Apply regularization

$$
L = \frac{1}{N} \sum_{n=1}^{N} L_n + \lambda R(W)
\newline \\[0.6em]
R(W) = \sum_k \sum_l W_{kl}^2
$$
