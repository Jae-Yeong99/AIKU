import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    total = 0
    result = []
    for k in x:
        total += np.exp(k)
    for i in x:
        result.append(np.exp(i) / total)
    return np.array(result)

def cross_entropy(label, output):
    loss = -np.sum(label * np.log(output)) / label.shape[0]
    return loss

def feedforward(X, W1, b1, W2, b2):
    Z1 = np.matmul(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(A1, W2) + b2
    A2 = softmax(Z2)
    return A1, A2


def Gradient(X, Y, A1, A2, W1, b1, W2, b2, learning_rate, lambda_):
    # 출력층 오차
    error2 = A2 - Y
    DW2 = np.dot(A1.T, error2) / X.shape[0] + lambda_ * W2
    db2 = np.sum(error2, axis=0, keepdims=True) / X.shape[0]

    # 은닉층 오차
    error1 = np.dot(error2, W2.T) * A1 * (1 - A1)
    DW1 = np.dot(X.T, error1) / X.shape[0] + lambda_ * W1
    db1 = np.sum(error1, axis=0, keepdims=True) / X.shape[0]

    # 가중치 및 편향 업데이트
    W1 -= learning_rate * DW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * DW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2


#################################################################################################
# Load Data
import os
os.chdir(os.path.dirname(__file__))
print('Loading the dataset...')
dataset = pd.read_csv("olfactory_encoding_train.csv")
# classify data
features = dataset.iloc[:, 1:].values  # numpy array type
labels = dataset.iloc[:, 0].values
features = features /255.0 # 0~1 normalization
#one - hot encoding
num_classes = 5
labels_one_hot = np.zeros((labels.size, num_classes))
labels_one_hot[np.arange(labels.size), labels] = 1

#10-fold validation
folds = 10
indices = np.arange(features.shape[0])
np.random.shuffle(indices)
fold_size = len( indices) // folds

learning_rates = [0.001, 0.01, 0.1]
lambdas = [0.001, 0.01, 0.1]

best_hyperparams = None
best_accuracy = 0
accuracies = {}

# 신경망 구조 정의
input_size = features.shape[1]
hidden_size = 128 #hidden layer개수
output_size = num_classes

for fold in range(folds):
    val_indices = indices[fold * fold_size:(fold + 1) * fold_size]
    train_indices = np.setdiff1d(indices, val_indices)
    X_train, X_val = features[train_indices], features[val_indices]
    Y_train, Y_val = labels_one_hot[train_indices], np.argmax(labels_one_hot[val_indices], axis=1)

    # 가중치 및 편향 초기화
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    # 학습
    epochs = 50
    learning_rate = 0.01
    lambda_ = 0.001
    for epoch in range(epochs):
        # 순전파(feed-forward)
        A1, A2 = feedforward(X_train, W1, b1, W2, b2)
        # 손실 계산
        loss = cross_entropy(Y_train, A2)
        # 경사 하강법으로 가중치 업데이트
        W1, b1, W2, b2 = Gradient(X_train, Y_train, A1, A2, W1, b1, W2, b2, learning_rate, lambda_)

        if epoch % 10 == 0:
            print(f"Fold {fold + 1}, Epoch {epoch}, Loss: {loss:.4f}")

    # validation
    fold_accuracies = []
    _, A2_val = feedforward(X_val, W1, b1, W2, b2)
    predictions = np.argmax(A2_val, axis=1)
    accuracy = np.mean(predictions == Y_val)
    fold_accuracies.append(accuracy)

    #store results
    accuracy_avg = np.mean(fold_accuracies)
    accuracies[(learning_rate, lambda_)] = accuracy_avg
    if accuracy_avg > best_accuracy:
        best_accuracy = accuracy_avg
        best_params = (learning_rate, lambda_)

# 교차검증 결과 출력
print(f"10-Fold Cross-Validation Accuracy: {np.mean(fold_accuracies):.2f}")

#plot results
best_learning_rate, best_lambda = best_params
print(f"Best Parameters: Learning Rate = {best_learning_rate}, Lambda = {best_lambda}, accuracy={best_accuracy:.4f}")

# Train with best hyperparameters and plot training loss
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

train_losses = []
for epoch in range(epochs):
    A1, A2 = feedforward(X_train, W1, b1, W2, b2)
    loss = cross_entropy(Y_train, A2)
    train_losses.append(loss)
    W1, b1, W2, b2 = Gradient(X_train, Y_train, A1, A2, W1, b1, W2, b2, best_learning_rate, best_lambda)

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.show()

# Evaluate on test set
_, A2_test = feedforward(features[indices[-fold_size:]], W1, b1, W2, b2)
predictions = np.argmax(A2_test, axis=1)
true_labels = np.argmax(labels_one_hot[indices[-fold_size:]], axis=1)
accuracy = np.mean(predictions == true_labels)
print(f"Test set accuracy with best hyperparameters: {accuracy:.4f}")

# Output the best weights
print("Best weights:")
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)