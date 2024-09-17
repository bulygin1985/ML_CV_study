import numpy as np

np.random.seed(1)


def flip_data(
    data: np.array,
    prob: float = 0.5,
    axis=0,
) -> np.array:
    """
    Flips an array with given probability for given axis

    Inputs:
    - data: A numpy array of any size S

    Outputs:
    A numpy array of size S
    """
    if np.random.rand() < prob:
        return np.flip(data, axis=axis)
    return data


def flip_horizontal(data: np.array, prob: float = 0.5) -> np.array:
    """
    Flips the image horizontally with given probability for each sample

    Inputs:
    - data: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

    Outputs:
    A numpy array of shape (N, D)
    """
    for i in range(data.shape[0]):
        data[i] = flip_data(data[i], prob=prob)

    return data
