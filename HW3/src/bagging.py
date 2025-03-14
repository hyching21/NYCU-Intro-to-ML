import typing as t
import numpy as np
import torch
# import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """Implement your code here"""

        losses_of_models = []
        n_samples = X_train.shape[0]

        for model in self.learners:
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X_train[indices]
            y_bootstrap = y_train[indices]
            X_bootstrap = torch.tensor(X_bootstrap, dtype=torch.float32)
            # y_bootstrap = 2 * y_bootstrap - 1 # convert{0,1} to {-1,1}
            y_bootstrap = torch.tensor(y_bootstrap, dtype=torch.float32)

            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                model.train()
                predictions = model(X_bootstrap).squeeze()
                predictions = torch.sigmoid(predictions)
                loss = entropy_loss(predictions, y_bootstrap)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses_of_models.append(loss.item())

        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        num_classifiers = len(self.learners)
        num_samples = X.shape[0]
        y_pred_probs = torch.zeros(num_classifiers, num_samples)
        final_prediction = torch.zeros(num_samples)
        X = torch.tensor(X, dtype=torch.float32)

        for idx, model in enumerate(self.learners):
            model.eval()
            with torch.no_grad():
                predictions = model(X).squeeze()
                predictions = torch.sigmoid(predictions)
                y_pred_probs[idx, :] = predictions
                final_prediction += predictions
                # final_prediction += torch.sign(predictions)
        final_prediction /= num_classifiers
        # print(f"final_prediction: {final_prediction}")
        y_pred_classes = (final_prediction >= 0.5).int()
        # y_pred_classes = torch.sign(final_prediction)
        # print(f"y_pred_classes: {y_pred_classes}")
        # print(f"y_pred_probs: {y_pred_probs}")

        return y_pred_classes.tolist(), y_pred_probs.tolist()

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        num_features = self.learners[0].model[0].weight.shape[1]
        feature_importance = torch.zeros(num_features)
        for model in self.learners:
            with torch.no_grad():
                weights = model.model[0].weight.abs()
                weights_2 = model.model[1].weight.abs()
                combined_importance = torch.matmul(weights_2, weights)
                neuron_importance = combined_importance.squeeze(0)
                # neuron_importance = weights.sum(dim=0)
                feature_importance += neuron_importance
        return feature_importance.tolist()
