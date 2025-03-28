import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# 🔹 Rotor Parameters
R = 7.1287
R_cut = 0.2
c_R = 0.102
Nb = 2
theta_tw = np.radians(-10)
mu = 0.19
rho = 1.225
M_tip = 0.65
a = 340
U_tip = M_tip * a
Omega = M_tip * a / R
beta_1c = np.radians(2.13)
beta_1s = np.radians(-0.15)
tolerance = 1e-6
max_iter = 100
eps = 1e-9

# 🔹 Computational Grid
n_psi = 36
n_r = 14
psi_vals = np.linspace(0, 2*np.pi, n_psi)
r_vals = np.linspace(0.2, 1, n_r)

# 🔹 Trim Targets
C_T_target = 0.00464
C_Mx_target = 0
C_My_target = 0
y_target = np.array([C_T_target, C_Mx_target, C_My_target])

# 🔹 Initial Guesses (Trimmed Values)
theta_0 = np.radians(5.62)
theta_1c = np.radians(0.64)
theta_1s = np.radians(-4.84)
x = np.array([theta_0, theta_1c, theta_1s])

# 🔹 Compute C_T using Blade Element Theory with Lambda Iteration

def compute_aero_coefficients(x):
    theta_0, theta_1c, theta_1s = x
    C_T, C_Mx, C_My = 0, 0, 0
    C_n_total = np.zeros((n_psi, n_r))  
    Alpha_effective = np.zeros((n_psi, n_r))  

    for j, r_R in enumerate(r_vals):
        r = r_R * R  
        theta = theta_0 + theta_tw * (r_R - 0.75) + theta_1c * np.cos(psi_vals) + theta_1s * np.sin(psi_vals)

        lambda_0 = C_T_target / (2 * np.sqrt(mu**2 + 0.01**2))  
        lambda_new = np.ones(n_psi) * lambda_0  

        for iteration in range(max_iter):
            lambda_old = lambda_new.copy()

            beta = beta_1c * np.cos(psi_vals) + beta_1s * np.sin(psi_vals)  
            beta_dot = Omega * (beta_1c * np.cos(psi_vals) - beta_1s * np.sin(psi_vals))

            U_T = Omega * r + mu * U_tip * np.sin(psi_vals)
            U_P = (lambda_new + (r * beta_dot) / Omega + mu * beta * np.cos(psi_vals)) * U_tip
            U_R = mu * np.cos(psi_vals) * U_tip
            U_eff = np.sqrt(U_T**2 + U_P**2 + U_R**2)

            phi = np.arctan(U_P / (U_T + eps))  
            alpha = theta - phi  

            Cl_alpha = 2 * np.pi
            Cl = Cl_alpha * alpha
            Cd0 = 0.011
            Cd = Cd0 + (Cl**2) / (np.pi * 0.7 * 6)

            dL = Cl * 0.5 * rho * U_eff**2 * (c_R * R)
            dD = Cd * 0.5 * rho * U_eff**2 * (c_R * R)
            dT = dL * np.cos(phi) - dD * np.sin(phi)
            dC_T = dT / (0.5 * rho * (Omega * R)**2 * np.pi * R**2)

            dMx = dT * np.sin(psi_vals) * r / R
            dMy = dT * np.cos(psi_vals) * r / R

            k_x = 4/3 * ((1 - np.cos(phi) - 1.8 * mu**2) / (np.sin(phi) + eps))
            k_y = -2 * mu
            lambda_new = lambda_0 * (1 + k_x * r_R * np.cos(psi_vals) + k_y * r_R * np.sin(psi_vals) + (U_R / U_tip))

            if np.max(np.abs(lambda_new - lambda_old)) < tolerance:
                break

        C_T += trapezoid(dC_T * r_R, psi_vals) / (2 * np.pi)
        C_Mx += trapezoid(dMx * r_R**2, psi_vals) / (2 * np.pi)
        C_My += trapezoid(dMy * r_R**2, psi_vals) / (2 * np.pi)

        dr = (r_vals[1] - r_vals[0]) * R
        area_section = c_R * R * dr
        C_n_total[:, j] = dT / (0.5 * rho * U_tip**2 * area_section)
        Alpha_effective[:, j] = np.degrees(alpha)

    return C_T, C_Mx, C_My, C_n_total, Alpha_effective

# 🔹 방위각에 대한 Cn (XY 플로트)
C_T_final, C_Mx_final, C_My_final, C_n_total, Alpha_effective = compute_aero_coefficients(x)

plt.figure(figsize=(10, 6))
r_target = 0.60
j_idx = np.argmin(np.abs(r_vals - r_target))
Cn_actual = C_n_total[:, j_idx]
psi_deg = np.degrees(psi_vals)

plt.plot(psi_deg, Cn_actual, marker='o', linestyle='-', label=f"r/R = {r_target:.2f}")
plt.xlabel("Azimuth Angle ψ (deg)", fontsize=12)
plt.ylabel("Normal Thrust Coefficient $C_n$", fontsize=12)
plt.title(f"$C_n$ vs Azimuth Angle at r/R = {r_target:.2f}", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
