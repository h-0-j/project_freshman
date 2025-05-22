import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. Himmelblau's function 정의
def f(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# 2. Full Factorial grid 생성
x_vals = np.linspace(-6, 6, 10)
y_vals = np.linspace(-6, 6, 10)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
X = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
y = f(X[:, 0], X[:, 1])

# 3. 다항 특징 변환 (3차)
degree = 6
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)

# 4. 회귀 모델 학습
model = LinearRegression()
model.fit(X_poly, y)

# 5. 예측용 고해상도 격자 생성
x_plot = np.linspace(-6, 6, 100)
y_plot = np.linspace(-6, 6, 100)
Xq, Yq = np.meshgrid(x_plot, y_plot)
X_query = np.column_stack([Xq.ravel(), Yq.ravel()])
X_query_poly = poly.transform(X_query)
Z_pred = model.predict(X_query_poly).reshape(Xq.shape)

# 6. 시각화
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xq, Yq, Z_pred, cmap='plasma', alpha=0.85)
ax.set_title("Polynomial Regression Model (Full Factorial Data)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
plt.tight_layout()
plt.show()
