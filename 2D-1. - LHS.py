import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# 1. 함수 정의 (Himmelblau)
def f(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# 2. LHS 2D 샘플링 함수
def lhs_2d(n_samples, x_range, y_range, seed=42):
    np.random.seed(seed)
    cut = np.linspace(0, 1, n_samples + 1)
    u_x = np.random.rand(n_samples)
    u_y = np.random.rand(n_samples)
    points_x = cut[:n_samples] + u_x * (cut[1:] - cut[:n_samples])
    points_y = cut[:n_samples] + u_y * (cut[1:] - cut[:n_samples])
    np.random.shuffle(points_x)
    np.random.shuffle(points_y)
    x_lhs = x_range[0] + points_x * (x_range[1] - x_range[0])
    y_lhs = y_range[0] + points_y * (y_range[1] - y_range[0])
    return np.column_stack((x_lhs, y_lhs))

# 3. 데이터 생성 (LHS 기반)
n_samples = 100
data_points = lhs_2d(n_samples, x_range=(-6, 6), y_range=(-6, 6))
targets = f(data_points[:, 0], data_points[:, 1])

# 4. 다항 회귀 특징변환 (3차)
degree = 3
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(data_points)

# 5. K-Fold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 6. Fold별 Train/Validation 그래프
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_poly)):
    X_train, X_test = X_poly[train_idx], X_poly[test_idx]
    y_train, y_test = targets[train_idx], targets[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # 추세선
    m_train, b_train = np.polyfit(y_train, y_train_pred, 1)
    m_test, b_test = np.polyfit(y_test, y_test_pred, 1)

    # 시각화 (Fold별 2개 subplot)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"LHS Fold {fold_idx + 1} Regression Performance", fontsize=14)

    # Train
    axes[0].scatter(y_train, y_train_pred, edgecolors='k', facecolors='none', label='Data')
    axes[0].plot(y_train, m_train * y_train + b_train, 'b-', label='Fit')
    axes[0].plot(y_train, y_train, 'k--', label='Y = T')
    axes[0].set_title(f"Training: R={r2_train:.5f}")
    axes[0].set_xlabel("Target")
    axes[0].set_ylabel("Output")
    axes[0].legend()
    axes[0].grid(True)

    # Validation
    axes[1].scatter(y_test, y_test_pred, edgecolors='k', facecolors='none', label='Data')
    axes[1].plot(y_test, m_test * y_test + b_test, 'g-', label='Fit')
    axes[1].plot(y_test, y_test, 'k--', label='Y = T')
    axes[1].set_title(f"Validation: R={r2_test:.5f}")
    axes[1].set_xlabel("Target")
    axes[1].set_ylabel("Output")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
