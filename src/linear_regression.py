import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv('data/wine.csv').to_numpy()

# 説明変数
X = df[:, [0, 1]]

# 目的変数
y = df[:, -1]

X = np.c_[np.ones(X.shape[0]), X]

beta = np.linalg.inv(X.T @ X) @ X.T @ y

y_pred = X @ beta

plt.plot(y, y_pred, '.')

plt.xlabel("Actual quality")
plt.ylabel("Predicted quality")

plt.title("Linear Regression")

plt.savefig("linear_regression.png")

plt.show()
