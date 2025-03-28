#전진비행을 한다면 방위각에 대한 Cn그래프가 전방블레이드 : 방위각=90도일때 양력이 감소하고 후방블레이드 방위각=270도일때 양력이 증가하도록 해야한다.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid  
import numpy.ma as ma

# 🔹 Rotor Geometry Input
R = 7.1287 # m
R_cut = 0.2 # R의 20%
c_R = 0.102
Nb = 2
theta_tw = np.radians(-10)
mu = 0.19
rho = 1.225
M_tip = 0.65
a = 340
U_tip = M_tip * a
Omega = M_tip * a / R
chord=c_R*R
beta_1c = np.radians(2.13)
beta_1s = np.radians(-0.15)
tolerance = 1e-6
max_iter = 200
eps = 1e-9

n_psi = 36
n_r = 14
psi_vals = np.linspace(0, 2*np.pi, n_psi)
r_vals = np.linspace(0, 1, n_r)

C_T_target = 0.00464
C_Mx_target = 0
C_My_target = 0
y_target = np.array([C_T_target, C_Mx_target, C_My_target])

theta_0 = np.radians(5.62)
theta_1c = np.radians(0.64)
theta_1s = np.radians(-4.84)
x = np.array([theta_0, theta_1c, theta_1s])

lambda_0_sections = np.ones_like(r_vals) * (C_T_target / (2 * np.sqrt(mu**2 + 1e-6)))

def compute_aero_coefficients(x):
    global lambda_0_sections
    theta_0, theta_1c, theta_1s = x
    C_T, C_Mx, C_My = 0, 0, 0
    C_n_total = np.zeros((n_psi, n_r))
    Alpha_effective = np.zeros((n_psi, n_r))

    # [1단계] Lambda 계산 루프 (선 계산)
    for j, r_R in enumerate(r_vals):
        r = r_R * R
        lambda_0 = lambda_0_sections[j]
        lambda_new = np.ones(n_psi) * lambda_0

        for _ in range(max_iter):
            lambda_old = lambda_new.copy()
            beta = beta_1c * np.cos(psi_vals) + beta_1s * np.sin(psi_vals)
            beta_dot = -Omega * (beta_1c * np.sin(psi_vals) - beta_1s * np.cos(psi_vals))
            U_T = Omega * r + mu * U_tip * np.sin(psi_vals)
            U_P = (lambda_new + (r * beta_dot) / Omega + mu * beta * np.cos(psi_vals)) * U_tip
            x_angle = np.arctan(mu / (1e-6 + lambda_new))
            k_x = 4/3 * ((1 - np.cos(x_angle) - 1.8 * mu**2) / np.maximum(np.abs(np.sin(x_angle)), 0.05))
            k_y = -2 * mu
            lambda_theory = lambda_0 * (1 + ((k_x * np.cos(psi_vals)) / r_R + (k_y * np.sin(psi_vals)) / r_R))
            lambda_new = lambda_theory / np.clip((2 / np.pi) * np.arccos(np.exp(-np.clip((Nb / 2) * ((1 - r_R) / (r_R * np.maximum(np.abs(np.sin(np.arctan(U_P / U_T))), 0.01))), 0, 20))), 0.3, 1.0)

            if np.max(np.abs(lambda_new - lambda_old)) < tolerance:
                # lambda_0_sections[j] = 0.7 * lambda_0_sections[j] + 0.3 * np.mean(lambda_new)
                break

    # [2단계] 공력 계수 계산 루프
    for j, r_R in enumerate(r_vals):
        r = r_R * R
        theta = theta_0 + theta_tw * (r_R - R_cut) + theta_1c * np.cos(psi_vals) + theta_1s * np.sin(psi_vals)
        lambda_new = np.ones(n_psi) * lambda_0_sections[j]

        beta = beta_1c * np.cos(psi_vals) + beta_1s * np.sin(psi_vals)
        beta_dot = -Omega * (beta_1c * np.sin(psi_vals) - beta_1s * np.cos(psi_vals))
        U_T = Omega * r + mu * U_tip * np.sin(psi_vals)
        U_P = (lambda_new + (r * beta_dot) / Omega + mu * beta * np.cos(psi_vals)) * U_tip
        U_R = mu * np.cos(psi_vals) * U_tip
        U_eff = np.sqrt(U_T**2 + U_P**2 )

        phi = np.arctan(U_P / np.sqrt(U_T**2 + U_R**2))
        alpha = theta - phi

        f = (Nb / 2) * ((1 - r_R) / (r_R * np.maximum(np.abs(np.sin(phi)), 0.01)))
        F = np.clip((2 / np.pi) * np.arccos(np.exp(-np.clip(f, 0, 20))), 0.3, 1.0)

        Cl_alpha = 2 * np.pi
        Cl = F * Cl_alpha * alpha
        Cd0 = 0.011
        Cd = F * Cd0 + (Cl**2) / (np.pi * 0.7 * 6)
 
        
        dr = (r_vals[1] - r_vals[0]) * R
        dL = 0.5 * rho * chord * Cl_alpha * (theta*U_T**2-U_P*U_T)*dr
        dD = 0.5*rho*U_eff**2*chord*Cd*dr
        dT = dL * np.cos(phi) - dD * np.sin(phi)

        dC_T = dT / (0.5 * rho * (Omega * R)**2 * np.pi * R**2)
        dMx = dT * np.sin(psi_vals) * r_R / R
        dMy = dT * np.cos(psi_vals) * r_R / R

        C_T += trapezoid(dC_T * r_R, psi_vals) / (2 * np.pi)
        C_Mx += trapezoid(dMx * r_R**2, psi_vals) / (2 * np.pi)
        C_My += trapezoid(dMy * r_R**2, psi_vals) / (2 * np.pi)
       

        area_section = c_R * R * dr 
        C_n_total[:, j] = dT / (0.5 * rho * U_tip**2 * area_section)
        Alpha_effective[:, j] = np.degrees(alpha)

    return C_T, C_Mx, C_My, C_n_total, Alpha_effective, theta


def compute_jacobian(x):
    delta = 1e-4
    J = np.zeros((3, 3))
    for i in range(3):
        x_fwd = x.copy(); x_bwd = x.copy()
        x_fwd[i] += delta
        x_bwd[i] -= delta
        y_fwd = np.array(compute_aero_coefficients(x_fwd)[:3])
        y_bwd = np.array(compute_aero_coefficients(x_bwd)[:3])
        J[:, i] = (y_fwd - y_bwd) / (2 * delta)
    return J

# Jacobian + Newton-raphson
for iteration in range(max_iter):
    C_T, C_Mx, C_My, _, _,_ = compute_aero_coefficients(x)
    y_current = np.array([C_T, C_Mx, C_My])
    error = y_current - y_target

    print(f"Iter {iteration+1}: CT={C_T:.6f}, CMx={C_Mx:.6f}, CMy={C_My:.6f}, |Error|={np.linalg.norm(error):.6e}")

    if np.linalg.norm(error) < tolerance:
        print(" Converged:", iteration + 1)
        break

    J = compute_jacobian(x)
    try:
        dx = np.linalg.solve(J, -error)
    except np.linalg.LinAlgError:
        print(" Jacobian Matrix is Singular. Using pseudo-inverse and continuing.")
        dx = np.linalg.pinv(J) @ -error

    # Damping 적용
    #damping_factor = 0.3
    #x += damping_factor * dx
    x += dx

    print(f"Iter {iteration+1}: CT={C_T:.6f}, CMx={C_Mx:.6f}, CMy={C_My:.6f}")


theta_0_final, theta_1c_final, theta_1s_final = np.degrees(x)
print(f"\n Final Trimmed Angles:")
print(f"theta_0  = {theta_0_final:.4f}°")
print(f"theta_1c = {theta_1c_final:.4f}°")
print(f"theta_1s = {theta_1s_final:.4f}°")


# 🔹 Compute C_n and Effective Alpha for Graphs
_, _, _, C_n_total, Alpha_effective,_ = compute_aero_coefficients(x)
_,_,_,_,_,theta = compute_aero_coefficients(x)
Theta_grid = np.zeros((n_psi, n_r))  # shape: (방위각, 반지름)

# 예: 반지름 비율 r/R < 0.3 인 부분은 마스킹
R_mask = np.tile(r_vals < 0.3, (n_psi, 1))  # shape (n_psi, n_r)
aoa_masked = ma.masked_array(Alpha_effective, mask=R_mask)

for j, r_R in enumerate(r_vals):
    Theta_grid[:, j] = theta_0_final + theta_tw * (r_R - 0.75) + theta_1c_final * np.cos(psi_vals) + theta_1s_final * np.sin(psi_vals)
Psi, R_grid = np.meshgrid(psi_vals, r_vals, indexing='ij')  # shape (n_psi, n_r)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

contour = ax.contourf(Psi, R_grid, np.degrees(Theta_grid), levels=50, cmap='viridis')
# 🔹 컬러바
cbar = plt.colorbar(contour, ax=ax, orientation='vertical')
cbar.set_label("Pitch Angle θ (deg)", fontsize=12)

# 🔹 축 설정
ax.set_theta_zero_location("S")  # 0도 아래쪽
ax.set_theta_direction(1)        # 시계 방향
ax.set_xticks(np.radians([0, 90, 180, 270]))
ax.set_xticklabels(["ψ=0°", "90°", "180°", "270°"])

plt.title("Polar Contour of Blade Pitch Angle θ(ψ, r)", fontsize=14)

# 🔹 Plot Normal Thrust Coefficient vs Azimuth Angle
plt.figure(figsize=(10, 6))

#for r_R_target in [0.6,0.75,0.91,0.99]:
for r_R_target in [0.6]:
    j_idx = np.argmin(np.abs(r_vals - r_R_target))  # r/R 위치에 가장 가까운 인덱스 찾기
    C_n = C_n_total[:, j_idx]  # 해당 r/R 위치의 모든 psi에 대한 C_n 값
    plt.plot(np.degrees(psi_vals), C_n, linestyle='-', marker='o', label=f"r/R = {r_R_target:.2f}")

# 그래프 라벨링 및 스타일
plt.xlabel("Azimuth Angle (ψ) [deg]", fontsize=12)
plt.ylabel("Normal Thrust Coefficient ($C_n$)", fontsize=12)
plt.title("Variation of $C_n$ with Azimuth Angle at Different $r/R$", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

# 🔹 Contour Plot of Effective AoA
Psi, R_grid = np.meshgrid(psi_vals, r_vals)  # 방위각(ψ)과 반지름(r/R) 그리드 생성

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))  # 극좌표 플롯 생성
Z = np.sin(2 * Psi + np.pi / 4)  # 가로 방향 등고선이 되게 만듦

# 🔹 Contour plot
contour = ax.contourf(Psi, R_grid, aoa_masked.T, levels = np.linspace(-10, 10, 21), cmap="gray")  # AoA 값 degree 변환



# 🔹 Theta Axis 설정 (Azimuth Angle, ψ)
ax.set_theta_zero_location("S")  # 0도(ψ=0) 위치를 아래(South)로 설정
ax.set_theta_direction(1)  # 시계 방향 설정
ax.set_xticks(np.radians([0, 90, 180, 270]))  # 0°, 90°, 180°, 270° 표시
ax.set_xticklabels(["ψ=0°", "90°", "180°", "270°"])
ax.set_ylim(0.2, 1.0)

# 🔹 Colorbar 설정
cbar = plt.colorbar(contour, ax=ax, orientation='vertical', ticks=np.arange(-11, 11, 1))
cbar.set_label("DEG", fontsize=12, fontweight='bold')  # 레이블 설정
cbar.ax.tick_params(labelsize=10)  # 숫자 크기 조정
cbar.ax.set_yticklabels([str(i) for i in range(-11, 11)])  # 숫자 직접 기입
cbar.outline.set_edgecolor("black")  # 테두리 색 설정
cbar.outline.set_linewidth(1)  # 테두리 두께 설정

plt.show()


