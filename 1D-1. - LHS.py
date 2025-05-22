import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

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

# 3. 데이터 생성
n_samples = 50
x_lhs = lhs_1d(n_samples).reshape(-1, 1)
y_lhs = f(x_lhs).ravel()

# 4. 다항 회귀 모델 설정
degree = 4
poly = PolynomialFeatures(degree)

# 5. K-Fold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = []

# 6. Fold별 시각화
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x_lhs)):
    x_train, x_test = x_lhs[train_idx], x_lhs[test_idx]
    y_train, y_test = y_lhs[train_idx], y_lhs[test_idx]

    X_train_poly = poly.fit_transform(x_train)
    X_test_poly = poly.transform(x_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    r2_scores.append(r2_test)

    # 추세선 계산
    m_train, b_train = np.polyfit(y_train, y_train_pred, 1)
    m_test, b_test = np.polyfit(y_test, y_test_pred, 1)

    # Subplot 출력 (Train / Validation)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Fold {fold_idx + 1} (LHS)", fontsize=14)

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

# 평균 R² 출력
mean_r2 = np.mean(r2_scores)
print(f"\n✅ 평균 Validation R² Score across folds (LHS): {mean_r2:.5f}")
