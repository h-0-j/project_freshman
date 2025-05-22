import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. 목표 함수 정의
def f(x):
    return (6 * x - 2)**2 * np.sin(12 * x - 4)

# 2. Full Factorial 데이터 생성
n_samples = 10  # 등간격 샘플 수
x = np.linspace(0, 1, n_samples).reshape(-1, 1)
y = f(x).ravel()

# 3. 다항 회귀 모델 설정 (4차)
degree = 4
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(X_poly, y)

# 4. 예측용 고해상도 데이터 생성
x_test = np.linspace(0, 1, 200).reshape(-1, 1)
X_test_poly = poly.transform(x_test)
y_pred = model.predict(X_test_poly)

# 5. 시각화
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='black', label='Full Factorial Data')
plt.plot(x_test, y_pred, color='red', label=f'Polynomial Fit (degree={degree})')
plt.title("Polynomial Regression using Full Factorial Data")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
