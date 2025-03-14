import typing as t
# import numpy as np
import torch
# import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001):
        """Implement your code here"""
        losses_of_models = []
        n_samples = X_train.shape[0]
        self.sample_weights = torch.ones(n_samples) / n_samples

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = 2 * y_train - 1
        y_train_tens = torch.tensor(y_train, dtype=torch.float32)

        for model in self.learners:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            # train
            for epoch in range(num_epochs):
                model.train()
                predictions = model(X_train).squeeze()
                loss = entropy_loss(torch.sigmoid(predictions), y_train_tens)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # eval
            model.eval()
            with torch.no_grad():
                predictions = model(X_train).squeeze()
                # print(f"predictions: {predictions}")
                sign_predictions = torch.sign(predictions)
                # print(f"sign_predictions: {predictions}")
                errors = (sign_predictions != y_train_tens).float()
                # print(f"error: {errors}")
                weighted_error = torch.sum(self.sample_weights * errors)
                # print(f"Weighted error: {weighted_error.item()}")
            # 更新alpha (alpha越大，該分類器效果越好)
            alpha = 0.5 * torch.log((1 - weighted_error) / (weighted_error + 1e-10))
            self.alphas.append(alpha)
            # 更新sample weights
            self.sample_weights *= torch.exp(-alpha * y_train_tens * sign_predictions)
            self.sample_weights /= torch.sum(self.sample_weights)
            losses_of_models.append(weighted_error.item())

        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        num_classifiers = len(self.learners)
        num_samples = X.shape[0]
        y_pred_probs = torch.zeros(num_classifiers, num_samples)
        final_prediction = torch.zeros(num_samples)
        X = torch.tensor(X, dtype=torch.float32)

        for idx, (alpha, model) in enumerate(zip(self.alphas, self.learners)):
            model.eval()
            with torch.no_grad():
                predictions = model(X).squeeze()
                y_pred_probs[idx, :] = torch.sigmoid(predictions)
                sign_predictions = torch.sign(predictions)
                final_prediction += alpha * (sign_predictions)
        # print("Final prediction:", final_prediction)
        y_pred_classes = torch.sign(final_prediction)
        # print("y_pred_classes:", y_pred_classes)
        # print("y_pred_probs:", y_pred_probs)
        return y_pred_classes.tolist(), y_pred_probs.tolist()

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        num_features = self.learners[0].model[0].weight.shape[1]
        feature_importance = torch.zeros(num_features)
        for alpha, model in zip(self.alphas, self.learners):
            with torch.no_grad():
                weights = model.model[0].weight.abs()
                weights_2 = model.model[1].weight.abs()
                # neuron_importance = weights.sum(dim=0)
                # print(f"weights shape: {weights.shape}, weights_2 shape: {weights_2.shape}")
                combined_importance = torch.matmul(weights_2, weights)
                neuron_importance = combined_importance.squeeze(0)
                feature_importance += alpha * neuron_importance
        return feature_importance.tolist()
