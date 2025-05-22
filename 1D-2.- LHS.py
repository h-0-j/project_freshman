import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. 함수 정의
def f(x):
    return (6 * x - 2)**2 * np.sin(12 * x - 4)

# 2. LHS 샘플링 함수
def lhs_1d(n_samples, x_min=0.0, x_max=1.0, seed=42):
    np.random.seed(seed)
    cut = np.linspace(0, 1, n_samples + 1)
    u = np.random.rand(n_samples)
    points = cut[:n_samples] + u * (cut[1:] - cut[:n_samples])
    np.random.shuffle(points)
    return x_min + points * (x_max - x_min)

# 3. 데이터 생성 (LHS)
n_samples = 10
x = lhs_1d(n_samples).reshape(-1, 1)
y = f(x).ravel()

# 4. 다항 회귀 모델 생성
degree = 4
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(X_poly, y)

# 5. 예측용 데이터 생성
x_test = np.linspace(0, 1, 200).reshape(-1, 1)
X_test_poly = poly.transform(x_test)
y_pred = model.predict(X_test_poly)

# 6. 시각화
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='black', label='LHS Sampled Data')
plt.plot(x_test, y_pred, color='red', label=f'Polynomial Fit (degree={degree})')
plt.title("Polynomial Regression using LHS Data")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
