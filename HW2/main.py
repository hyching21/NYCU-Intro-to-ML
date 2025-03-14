import typing as t
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[np.float_],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        m, n = inputs.shape
        self.weights = np.zeros(n)
        self.intercept = 0

        for i in range(self.num_iterations):
            y_pred = self.sigmoid(inputs @ self.weights + self.intercept)
            dW = (inputs.T @ (y_pred - targets)) / m
            dB = np.sum(y_pred - targets) / m
            self.weights -= self.learning_rate * dW
            self.intercept -= self.learning_rate * dB

            # CE loss function
            if (i % 50 == 0):
                epsilon = 1e-15
                loss = -np.mean(targets * np.log(y_pred + epsilon) + (1 - targets) * np.log(1 - y_pred + + epsilon))
                print(f'Iteration {i}: Loss = {loss:.4f}')

    def predict(
        self,
        inputs: npt.NDArray[np.float_],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        y_pred_prob = self.sigmoid(inputs @ self.weights + self.intercept)
        y_pred_class = [1 if prob >= 0.5 else 0 for prob in y_pred_prob]
        return y_pred_prob, y_pred_class

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[np.float_],
        targets: t.Sequence[int],
    ) -> None:
        # mean vector
        x0 = inputs[targets == 0]
        x1 = inputs[targets == 1]
        self.m0 = np.mean(x0, axis=0)
        self.m1 = np.mean(x1, axis=0)
        # within-class covariance matrix
        self.sw = (x0 - self.m0).T @ (x0 - self.m0) + (x1 - self.m1).T @ (x1 - self.m1)
        # between-class covariance matrix
        self.sb = (self.m1 - self.m0).T @ (self.m1 - self.m0)
        # weight
        self.w = np.linalg.inv(self.sw) @ (self.m1 - self.m0)
        self.w = self.w / np.linalg.norm(self.w)
        # slope
        self.slope = self.w[1] / self.w[0]

    def predict(
        self,
        inputs: npt.NDArray[np.float_],
    ) -> t.Sequence[t.Union[int, bool]]:
        pred_0 = abs((inputs - self.m0) @ self.w)
        pred_1 = abs((inputs - self.m1) @ self.w)
        y_pred_fld = np.zeros(inputs.shape[0])
        y_pred_fld[pred_0 > pred_1] = 1

        return y_pred_fld.astype(int)

    def plot_projection(self, inputs: npt.NDArray[np.float_]):
        pred = self.predict(inputs)
        x0 = inputs[pred == 0]
        x1 = inputs[pred == 1]
        intercept = 20

        plt.figure(figsize=(10, 10))
        # plot original points
        plt.scatter(x0[:, 0], x0[:, 1], c="b", label="Class 0", s=5)
        plt.scatter(x1[:, 0], x1[:, 1], c="r", label="Class 1", s=5)
        # plot projection points
        u = np.array([0, intercept])

        for i in range(inputs.shape[0]):
            proj = u + ((inputs[i] - u) @ self.w) * self.w
            plt.scatter(proj[0], proj[1], c=("b" if pred[i] == 0 else "r"), s=5)
            # plot connected line
            plt.plot([inputs[i][0], proj[0]], [inputs[i][1], proj[1]], color="gray", linewidth=0.3, alpha=0.5)

        # plot projection line
        pt = np.linspace(-50, 100, 100)
        plt.plot(pt, self.slope * pt + intercept, color="purple")

        plt.title(f"Projection Line: w={self.slope:.5f}, b={intercept:.5f}")
        # limit x and y axis range
        plt.xlim(-7, 5)
        # plt.ylim(-50, 50)
        plt.legend()
        plt.show()


def compute_auc(y_trues, y_preds):
    return roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds):
    correct_predictions = np.sum(y_trues == y_preds)
    total_predictions = len(y_trues)
    return correct_predictions / total_predictions


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=0.025,  # You can modify the parameters as you want
        num_iterations=300,   # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercept: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['10', '20']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """
    FLD_.fit(x_train, y_train)
    y_pred_fld = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_fld)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    FLD_.plot_projection(x_train)
    FLD_.plot_projection(x_test)


if __name__ == '__main__':
    main()
