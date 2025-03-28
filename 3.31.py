#3.30 = 3.3 + tip loss factor
#3.31 = 3.3 + tip loss factor + C_n_total 수정 -> 정규화방식이 잘못되었다. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid  

# 🔹 Rotor Geometry Input
R = 7.1287  # Rotor Radius (m)
R_cut = 0.2
c_R = 0.102  # Chord/R ratio
Nb = 2  # Number of blades
theta_tw = np.radians(-10)  # Twist angle (radians)
mu = 0.19  # Advance ratio
rho = 1.225  # Air density (kg/m^3)
M_tip = 0.65  # Tip Mach number
a = 340  # Speed of sound (m/s)
U_tip = M_tip * a  # Tip speed 
Omega = M_tip * a / R  # Rotor angular velocity (rad/s)

beta_1c = np.radians(2.13)  # Longitudinal flapping angle
beta_1s = np.radians(-0.15)  # Lateral flapping angle
tolerance = 1e-6  # Convergence criteria
max_iter = 10000  # Newton-Raphson 최대 반복 횟수

eps = 1e-9  # Small value to prevent division by zero

# 🔹 Computational Grid (Azimuth & Radial Stations)
n_psi = 36  # Number of azimuth steps
n_r = 14  # Number of radial steps
psi_vals = np.linspace(0, 2*np.pi, n_psi)  # Azimuth angles (0-360 degrees)
r_vals = np.linspace(0.2, 1, n_r)  # Normalized radial positions (excluding hub)

# 🔹 Target Trim Values
C_T_target = 0.00464
C_Mx_target = 0
C_My_target = 0
y_target = np.array([C_T_target, C_Mx_target, C_My_target])

# 🔹 Initial Guesses (Collective and Cyclic Angles)
theta_0 = np.radians(10)  
theta_1c = np.radians(0)  
theta_1s = np.radians(0)  
x = np.array([theta_0, theta_1c, theta_1s])  

# 🔹 Compute C_T using Blade Element Theory with Lambda Iteration
def compute_aero_coefficients(x):
    theta_0, theta_1c, theta_1s = x
    C_T, C_Mx, C_My = 0, 0, 0
    C_n_total = np.zeros((n_psi, n_r))  
    Alpha_effective = np.zeros((n_psi, n_r))  

    for j, r_R in enumerate(r_vals):
        r = r_R * R  
        theta = theta_0 + theta_tw * (r_R - R_cut) + theta_1c * np.cos(psi_vals) + theta_1s * np.sin(psi_vals)

        # 🔹 Lambda 초기값 설정
        mu_x = mu
        mu_z = 0
        lambda_0 = C_T_target / (2 * np.sqrt(mu_x**2 + mu_z**2))  
        lambda_new = np.ones(n_psi) * lambda_0  
        

        # 🔹 Lambda & Tip loss factor 반복 계산 (수렴할 때까지)
        for iteration in range(max_iter):
            lambda_old = lambda_new.copy()

            beta = beta_1c * np.cos(psi_vals) + beta_1s * np.sin(psi_vals)  
            beta_dot = -Omega * (beta_1c * np.sin(psi_vals) - beta_1s * np.cos(psi_vals))

            # 🔹 Blade Element Velocities (BEMT)
            U_T = Omega * r + mu * U_tip * np.sin(psi_vals)
            U_P = (lambda_new + (r * beta_dot) / Omega + mu * beta * np.cos(psi_vals)) * U_tip  # ✅ Lambda 포함
            U_R = mu * np.cos(psi_vals) * U_tip  # ✅ 추가된 반경 방향 속도
            
            # 🔹 전체 유동 속도 계산
            U_eff = np.sqrt(U_T**2 + U_P**2 + U_R**2)  # ✅ U_R 포함
            
            phi = np.arctan(U_P / (U_T + eps))  
            alpha = theta - phi  

            # 🔹 Tip Loss Factor 적용
            f = (Nb / 2) * ((1 - r_R) / (r_R * np.maximum(np.abs(np.sin(phi)), 0.01)))  # 작은 값 제한 (0.01 추천)

            exp_term = np.exp(-np.clip(f, 0, 20))  # f를 너무 크지 않게 제한 (0~20)
            F = (2 / np.pi) * np.arccos(np.clip(exp_term, 0, 1))
            F = np.clip(F, 0.1, 1.0)  # F를 너무 작지 않게 제한 (최소 0.1 추천)

            # 🔹 Aerodynamic Forces (NACA 0012)
            Cl_alpha = 2 * np.pi  
            Cl = F * Cl_alpha * alpha  
            Cd0 = 0.011  
            Cd = F * Cd0 + (Cl**2) / (np.pi * 0.7 * 6)  

            # 🔹 Sectional Force Coefficients
            dC_L = Cl * 0.5 * rho * U_eff**2 * (c_R * R)
            dC_D = Cd * 0.5 * rho * U_eff**2 * (c_R * R)
            dC_T = (dC_L * np.cos(phi) - dC_D * np.sin(phi) )
              
            dC_Mx = dC_T * np.sin(psi_vals) *r/R
            dC_My = dC_T * np.cos(psi_vals) *r/R
            
            # 🔹 Lambda 업데이트
            k_x = 4/3 * ((1 - np.cos(phi) - 1.8 * mu**2) / (np.sin(phi) + eps))
            k_y = -2 * mu  

            lambda_theory = lambda_0 * (1 + k_x * r_R * np.cos(psi_vals) + k_y * r_R * np.sin(psi_vals) + (U_R / U_tip))
            lambda_new = lambda_theory / F
            
            if np.max(np.abs(lambda_new - lambda_old)) < tolerance:
                break  

        # 🔹 Compute total thrust coefficient
        C_T += trapezoid(dC_T * r_R, psi_vals) / (2 * np.pi)
        C_Mx += trapezoid(dC_Mx * r_R**2, psi_vals) / (2 * np.pi)
        C_My += trapezoid(dC_My * r_R**2, psi_vals) / (2 * np.pi)
        

        # 수정안: 단면 기준으로 정규화 (단면적 A_section = c * dr)
        dr = (r_vals[1] - r_vals[0]) * R
        area_section = c_R * R * dr
        C_n_total[:, j] = dC_T / (0.5 * rho * U_tip**2 * area_section)
        
        Alpha_effective[:, j] = np.degrees(alpha)

    return C_T, C_Mx, C_My, C_n_total, Alpha_effective

# Jacobian

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

# Newton-Raphson
# 🔹 Newton-Raphson Iteration
for iteration in range(max_iter):
    C_T, C_Mx, C_My, _, _ = compute_aero_coefficients(x)
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
        print("Jacobian Matrix is Singular. Using pseudo-inverse and continuing.")
        dx = np.linalg.pinv(J) @ -error  # 계속 진행

    x += dx

    print(f"Iter {iteration+1}: CT={C_T:.6f}, CMx={C_Mx:.6f}, CMy={C_My:.6f}")

theta_0_final, theta_1c_final, theta_1s_final = np.degrees(x)
print(f"\n Final Trimmed Angles:")
print(f"theta_0  = {theta_0_final:.4f}°")
print(f"theta_1c = {theta_1c_final:.4f}°")
print(f"theta_1s = {theta_1s_final:.4f}°")


# 🔹 Compute C_n and Effective Alpha for Graphs
_, _, _, C_n_total, Alpha_effective = compute_aero_coefficients(x)


# 🔹 Plot Normal Thrust Coefficient vs Azimuth Angle
plt.figure(figsize=(10, 6))

for r_R_target in [0.6, 0.75, 0.91, 0.99]:
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

# 🔹 Contour plot
contour = ax.contourf(Psi, R_grid, Alpha_effective.T, levels=np.linspace(0, 10, 11), cmap="rainbow")  # AoA 값 degree 변환
#plt.ylim(0.3, 0.6)

# 🔹 Colorbar 설정
cbar = plt.colorbar(contour, ax=ax, orientation='vertical', ticks=np.arange(0, 11, 1))
cbar.set_label("DEG", fontsize=12, fontweight='bold')  # 레이블 설정
cbar.ax.tick_params(labelsize=10)  # 숫자 크기 조정
cbar.ax.set_yticklabels([str(i) for i in range(0, 11)])  # 숫자 직접 기입
cbar.outline.set_edgecolor("black")  # 테두리 색 설정
cbar.outline.set_linewidth(1)  # 테두리 두께 설정

# 🔹 Theta Axis 설정 (Azimuth Angle, ψ)
ax.set_theta_zero_location("S")  # 0도(ψ=0) 위치를 아래(South)로 설정
ax.set_theta_direction(1)  # 시계 방향 설정
ax.set_xticks(np.radians([0, 90, 180, 270]))  # 0°, 90°, 180°, 270° 표시
ax.set_xticklabels(["ψ=0°", "90°", "180°", "270°"])

plt.show()
