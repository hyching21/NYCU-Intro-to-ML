import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        # beta = (X^T*X)^-1 * X^T * y
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept = beta[0]
        self.weights = beta[1:]

    def predict(self, X):
        pred = X @ self.weights + self.intercept
        return pred


class LinearRegressionGradientdescent(LinearRegressionBase):
    def normalize(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        norm_x = (X - mean) / std
        return mean, std, norm_x

    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        mean, std, X = self.normalize(X)
        m, n = X.shape
        self.weights = np.zeros(n)
        self.intercept = 0
        losses = []
        y = y.flatten()

        for _ in range(epochs):
            y_pred = X @ (self.weights) + self.intercept
            loss = compute_mse(y_pred, y)
            losses.append(loss)
            dW = (2 / m) * (X.T @ (y_pred - y))
            dB = (2 / m) * np.sum(y_pred - y)
            self.weights -= learning_rate * dW
            self.intercept -= learning_rate * dB

        self.weights = self.weights / std
        self.intercept = self.intercept - np.sum(self.weights * std * (mean / std))

        return losses

    def predict(self, X):
        pred = X @ self.weights + self.intercept
        return pred

    def plot_learning_curve(self, losses):
        loss = plt.plot(losses)
        plt.title("Training loss")
        plt.legend(loss, ["Train MSE loss"])
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.show()


def compute_mse(prediction, ground_truth):
    mse = np.mean((prediction - ground_truth) ** 2)
    return mse


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=0.025, epochs=200)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Mean prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()

