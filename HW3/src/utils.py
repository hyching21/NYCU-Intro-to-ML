import typing as t
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


def preprocess(df: pd.DataFrame):
    """
    (TODO): Implement your preprocessing function.
    """
    df = df.copy()
    binary_columns = ['person_gender', 'person_home_ownership', 'previous_loan_defaults_on_file']
    multi_class_columns = ['person_education', 'loan_intent']
    columns_to_process = binary_columns + multi_class_columns
    for col in columns_to_process:
        df[col] = pd.Categorical(df[col]).codes
    # scaler = StandardScaler()
    # numeric_columns = df.select_dtypes(include=['float64', 'int']).columns
    # df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df.to_numpy()


class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layers model.
    Here, for example:
        - Linear(input_dim, 1) is a single-layer model.
        - Linear(input_dim, k) -> Linear(k, 1) is a two-layer model.

    No non-linear activation allowed.
    """
    def __init__(self, input_dim, hidden_dim: int = 12):
        super(WeakClassifier, self).__init__()
        if hidden_dim == 0:
            self.model = nn.Linear(input_dim, 1)
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x):
        output = self.model(x)
        return output


def accuracy_score(y_trues, y_preds) -> float:
    correct_predictions = np.sum(y_trues == y_preds)
    total_predictions = len(y_trues)
    accuracy = correct_predictions / total_predictions
    return accuracy


def entropy_loss(outputs, targets):
    epsilon = 1e-10
    outputs = torch.clamp(outputs, min=epsilon, max=1 - epsilon)
    log_probs = targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs)
    loss = -torch.mean(log_probs)
    return loss


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    if not y_preds or len(y_preds) == 0:
        raise ValueError("y_preds is empty")
    plt.figure(figsize=(10, 8))
    for i, preds in enumerate(y_preds):
        fpr, tpr, _ = roc_curve(y_trues, preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Learner {i+1} (AUC= {roc_auc:.3f})')
    # plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=18)
    plt.ylabel('TPR', fontsize=18)
    plt.title('AUC curve for each weak classifier', fontsize=16)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    # save
    plt.savefig(fpath)
    plt.close()


def plot_feature_importance(feature_importance, feature_names, fpath):
    feature_importance = np.array(feature_importance)
    indices = np.argsort(feature_importance)[::-1]
    sorted_importance = feature_importance[indices]
    sorted_names = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importance)), sorted_importance, align="center", alpha=0.7, color='skyblue')
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha="right", fontsize=10)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Importance", fontsize=12)
    plt.title("Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.grid(axis="y", alpha=0.3)
    # plt.show()
    plt.savefig(fpath)
    plt.close()
