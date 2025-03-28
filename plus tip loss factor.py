#Tip loss factor 추가한것.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid  

# 🔹 Rotor Geometry Input
R = 7.1287  # Rotor Radius (m)
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
max_iter = 100  # Newton-Raphson 최대 반복 횟수

eps = 1e-9  # Small value to prevent division by zero

# 🔹 Computational Grid (Azimuth & Radial Stations)
n_psi = 36  # Number of azimuth steps
n_r = 14  # Number of radial steps
psi_vals = np.linspace(0, 2*np.pi, n_psi)  # Azimuth angles (0-360 degrees)
r_vals = np.linspace(0.2, 1, n_r)  # Normalized radial positions (excluding hub)

# 🔹 Target Trim Values
C_T_target = 0.05
C_Mx_target = 0.02
C_My_target = -0.1
y_target = np.array([C_T_target, C_Mx_target, C_My_target])

# 🔹 Initial Guesses (Collective and Cyclic Angles)
theta_0 = np.radians(10)  
theta_1c = np.radians(2.5)  
theta_1s = np.radians(-6.5) 
x = np.array([theta_0, theta_1c, theta_1s])  

# 🔹 Compute C_T using Blade Element Theory with Lambda Iteration
def compute_aero_coefficients(x):
    theta_0, theta_1c, theta_1s = x
    C_T, C_Mx, C_My = 0, 0, 0
    C_n_total = np.zeros((n_psi, n_r))  
    Alpha_effective = np.zeros((n_psi, n_r))  

    for j, r_R in enumerate(r_vals):
        r = r_R * R  
        theta = theta_0 + theta_tw * (r_R - 0.2) + theta_1c * np.cos(psi_vals) + theta_1s * np.sin(psi_vals)

        # 🔹 Lambda 초기값 설정
        lambda_0 = C_T_target / (2 * np.sqrt(mu**2 + 0.01**2))  
        lambda_new = np.ones(n_psi) * lambda_0  

        # 🔹 Lambda 반복 계산 (수렴할 때까지)
        for iteration in range(max_iter):
            lambda_old = lambda_new.copy()

            beta = beta_1c * np.cos(psi_vals) + beta_1s * np.sin(psi_vals)  
            beta_dot = Omega * (beta_1c * np.cos(psi_vals) - beta_1s * np.sin(psi_vals))

            # 🔹 Blade Element Velocities (BEMT)
            U_T = Omega * r + mu * U_tip * np.sin(psi_vals)  # 접선 속도
            U_P = (lambda_new + (r * beta_dot) / Omega + mu * beta * np.cos(psi_vals)) * U_tip  # 유도 속도
            U_R = mu * np.cos(psi_vals) * U_tip  # ✅ 추가된 반경 방향 속도

            # 🔹 전체 유동 속도 계산
            U_eff = np.sqrt(U_T**2 + U_P**2 + U_R**2)  # ✅ U_R 포함

            phi = np.arctan(U_P / (U_T + eps))  
            alpha = theta - phi  

            # 🔹 Tip Loss Factor 적용
            f = (Nb / 2) * ((1 - r_R) / (r_R*np.abs(np.sin(phi)) + eps))
            F = (2 / np.pi) * np.arccos(np.exp(-f))  # Prandtl Tip Loss Factor

            # 🔹 Aerodynamic Forces (NACA 0012)
            Cl_alpha = 2 * np.pi  
            Cl = F * Cl_alpha * alpha  # ✅ Tip Loss Factor 적용
            Cd0 = 0.011  
            Cd = F * (Cd0 + (Cl**2) / (np.pi * 0.7 * 6))  # ✅ Tip Loss Factor 적용

            # 🔹 Sectional Force Coefficients
            dC_L = Cl * 0.5 * rho * U_eff**2 * (c_R * R)
            dC_D = Cd * 0.5 * rho * U_eff**2 * (c_R * R)
            dC_T = dC_L * np.cos(phi) - dC_D * np.sin(phi)  
            dC_Mx = dC_T * np.sin(psi_vals)  
            dC_My = dC_T * np.cos(psi_vals) 
            
            # 🔹 Lambda 업데이트 (U_R 추가)
            k_x = 4/3 * ((1 - np.cos(phi) - 1.8 * mu**2) / (np.sin(phi) + eps))
            k_y = -2 * mu  

            lambda_new = lambda_0 * (1 + k_x * r_R * np.cos(psi_vals) + k_y * r_R * np.sin(psi_vals) + (U_R / U_tip))

            if np.max(np.abs(lambda_new - lambda_old)) < tolerance:
                break  

        # 🔹 Compute total thrust coefficient
        C_T += trapezoid(dC_T * r_R, psi_vals) / (2 * np.pi)
        C_Mx += trapezoid(dC_Mx * r_R, psi_vals) / (2 * np.pi)
        C_My += trapezoid(dC_My * r_R, psi_vals) / (2 * np.pi)
        
        C_n_total[:, j] = dC_T / (0.5 * rho * (U_tip**2) * np.pi * R**2)
        Alpha_effective[:, j] = np.degrees(alpha)

    return C_T, C_Mx, C_My, C_n_total, Alpha_effective  

# 🔹 Compute Jacobian Matrix (J)
def compute_jacobian(x):
    delta = 1e-4
    J = np.zeros((3,3))

    for i in range(3):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += delta
        x_backward[i] -= delta

        y_forward = compute_aero_coefficients(x_forward)[0]  
        y_backward = compute_aero_coefficients(x_backward)[0]  

        J[:,i] = (y_forward - y_backward) / (2 * delta)  

    return J

# 🔹 Newton-Raphson Iteration
for iteration in range(max_iter):
    y_current,C_Mx_current,C_My_current,_,_= compute_aero_coefficients(x)
    error = y_current - y_target

    # 수렴여부 확인인 
    if np.linalg.norm(error) < tolerance:
        print(f" Converged in {iteration} iterations: C_T = {y_current:.6f}, C_Mx = {C_Mx_current:.6f}, C_My = {C_My_current:.6f}")
        break

    J = compute_jacobian(x)

    try:
        dx = np.linalg.solve(J, -error)  
        
    except np.linalg.LinAlgError:
        print(" Jacobian Matrix is Singular. Adjust initial guess.")
        dx = np.linalg.pinv(J) @ -error  # 유사 역행렬 사용

        break
    
    x = x + dx 
     

    print(f"Iteration {iteration+1}: C_T = {y_current:.6f}, Error = {error:.6f}")
    print(f"Iteration {iteration+1}: C_Mx = {C_Mx_current:.6f}, Error = {error:.6f}")
    print(f"Iteration {iteration+1}: C_My = {C_My_current:.6f}, Error = {error:.6f}")


theta_0_final, theta_1c_final, theta_1s_final = np.degrees(x)
print(f"\n Final Trimmed Angles:")
print(f"theta_0  = {theta_0_final:.4f}°")
print(f"theta_1c = {theta_1c_final:.4f}°")
print(f"theta_1s = {theta_1s_final:.4f}°")

# 🔹 Compute C_n and Effective Alpha for Graphs
_,_,_, C_n_total, Alpha_effective = compute_aero_coefficients(x)



# 🔹 Plot Normal Thrust Coefficient vs Azimuth Angle
plt.figure(figsize=(10, 6))

for idx, r_R in enumerate([0.6, 0.75, 0.91, 0.99]):
    j_idx = np.argmin(np.abs(r_vals - r_R))
    C_n = C_n_total[:, j_idx]
    plt.plot(np.degrees(psi_vals), C_n, linestyle='-', marker='o', label=f"r/R = {r_R:.2f}")

plt.xlabel("Azimuth Angle (ψ) [deg]")
plt.ylabel("Normal Thrust Coefficient ($C_n$)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 🔹 Contour Plot of Effective AoA
Psi, R_grid = np.meshgrid(psi_vals, r_vals)  # 방위각(ψ)과 반지름(r/R) 그리드 생성

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))  # 극좌표 플롯 생성

# 🔹 Contour plot
contour = ax.contourf(Psi, R_grid, Alpha_effective.T, levels=np.linspace(0, 10, 11), cmap="gray")  # AoA 값 degree 변환

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
