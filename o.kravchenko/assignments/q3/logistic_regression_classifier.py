from typing import Tuple

import numpy as np
from linear_classifier import LinearClassifier


class LogisticRegression(LinearClassifier):
    """A subclass that uses L2 loss function and sigmoid"""

    def loss(self, X_batch, y_batch, reg) -> Tuple[float, np.array]:
        # TODO: implement sigmoid classifier
        pass
