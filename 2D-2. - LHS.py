import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. Himmelblau's function 정의
def f(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# 2. Latin Hypercube Sampling (LHS) in 2D
def lhs_2d(n_samples, x_range, y_range, seed=42):
    np.random.seed(seed)
    cut = np.linspace(0, 1, n_samples + 1)
    u_x = np.random.rand(n_samples)
    u_y = np.random.rand(n_samples)
    points_x = cut[:n_samples] + u_x * (cut[1:] - cut[:n_samples])
    points_y = cut[:n_samples] + u_y * (cut[1:] - cut[:n_samples])
    np.random.shuffle(points_x)
    np.random.shuffle(points_y)
    x = x_range[0] + points_x * (x_range[1] - x_range[0])
    y = y_range[0] + points_y * (y_range[1] - y_range[0])
    return np.column_stack((x, y))

# 3. 데이터 생성
n_samples = 100
data = lhs_2d(n_samples, x_range=(-6, 6), y_range=(-6, 6))
X = data
y = f(X[:, 0], X[:, 1])

# 4. 다항 특징 변환 (3차)
degree = 4
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)

# 5. 회귀 모델 학습
model = LinearRegression()
model.fit(X_poly, y)

# 6. 예측용 고해상도 격자 생성
x_plot = np.linspace(-6, 6, 100)
y_plot = np.linspace(-6, 6, 100)
X_grid, Y_grid = np.meshgrid(x_plot, y_plot)
X_query = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
X_query_poly = poly.transform(X_query)
Z_pred = model.predict(X_query_poly).reshape(X_grid.shape)

# 7. 시각화
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid, Y_grid, Z_pred, cmap='viridis', alpha=0.8)
ax.set_title("Polynomial Regression Model (LHS Data)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
plt.tight_layout()
plt.show()
