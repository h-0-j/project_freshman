import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

# 🔹 Rotor Parameters
R = 7.1287  # m
R_cut = 0.2
c_R = 0.102
Nb = 2
theta_tw = np.radians(-10)
mu = 0.19
rho = 1.225
M_tip = 0.65
a = 340
U_tip = M_tip * a
Omega = U_tip / R
beta_1c = np.radians(2.13)
beta_1s = np.radians(-0.15)
tolerance = 1e-6
max_iter = 100
eps = 1e-9
chord = c_R * R

# 🔹 Computational Grid
n_psi = 72
n_r = 14
psi_vals = np.linspace(0, 2 * np.pi, n_psi)
r_vals = np.linspace(0.2, 1, n_r)


def compute_rotor_trim(x):
    theta_0, theta_1c, theta_1s = x
    C_n_total = np.zeros((n_psi, n_r))
    CT, CMx, CMy = 0.00464, 0.0, 0.0

    lambda_h = np.sqrt(CT/2)
    lambda_0 = lambda_h * np.sqrt((np.sqrt(1/4 * (mu/lambda_h)**4 + 1) - 1/2 * (mu/lambda_h)**2))  # Linear inflow

    for j, r_R in enumerate(r_vals):
        if r_R < 1e-6:
            continue
        r = r_R * R
        theta = theta_0 + theta_tw * (r_R - 0.75) + theta_1c * np.cos(psi_vals) + theta_1s * np.sin(psi_vals)

        beta = beta_1c * np.cos(psi_vals) + beta_1s * np.sin(psi_vals)
        beta_dot = - Omega * (beta_1c * np.sin(psi_vals) - beta_1s * np.cos(psi_vals))
        
        mu_x = mu
        mu_z = 0
        x_angle = np.arctan2(mu_x , (mu_z + lambda_0))
        k_x = 4/3 * ((1 - np.cos(x_angle) - 1.8 * mu**2) / np.sin(x_angle))
        k_y = -2 * mu
        lambda_i = ( 1 + k_x*np.cos(psi_vals)/r_R + k_y*np.sin(psi_vals)/r_R ) * lambda_0
    

        U_T = Omega * r + mu * Omega * R * np.sin(psi_vals)
        U_P = Omega * R * lambda_i + r * beta_dot * R + Omega * R * mu * beta * np.cos(psi_vals)
        U_eff = np.sqrt(U_T**2 + U_P**2)

        phi = np.arctan2(U_P , U_T)
        alpha = theta - phi

        Cl_alpha = 2 * np.pi
        Cl = Cl_alpha * alpha
        Cd0 = 0.011
        Cd = Cd0 + (Cl**2) / (np.pi * 0.7 * 6)
        
        #tip loss factor
        B = 1 - np.sqrt(CT) / Nb
        
        dpsi = (2 * np.pi) / n_psi
        dr = (r_vals[1] - r_vals[0]) * R
        dL = 0.5 * rho * chord * Cl * U_eff**2 * dr
        dD = 0.5 * rho * chord * Cd * U_eff**2 * dr
        
        dFx = dL * np.sin(phi) + dD * np.cos(phi)  # Body x-direction
        dFz = dL * np.cos(phi) - dD * np.sin(phi)  # Body z-direction
        moment_arm = r * np.cos(psi_vals)  # for CMy
        moment_arm_x = r * np.sin(psi_vals)

        dT = (dL * np.cos(phi) - dD * np.sin(phi))*dpsi
        
        sigma = Nb*chord/(np.pi*R)
        
        CT += np.sum(Nb * dT / (0.5 * rho * (Omega)**2 * np.pi * R**2))
        CMx += np.sum(Nb * dFz * moment_arm_x * dpsi/ (0.5 * rho * (Omega)**2 * np.pi * R**3))
        CMy += np.sum(Nb * dFx * moment_arm * dpsi) / (0.5 * rho * (Omega)**2 * np.pi * R**3)
        
    return CT, CMx, CMy

def compute_trim_residual(x, target_CT, target_CMx, target_CMy):
    theta_0, theta_1c, theta_1s = x
    CT, CMx, CMy = compute_rotor_trim(x)  
    residual = np.array([CT - target_CT, CMx - target_CMx, CMy - target_CMy])
    return residual

def newton_trim(x0, target_CT, target_CMx, target_CMy, tol=1e-6, max_iter=50):
    x = x0.copy()
    for i in range(max_iter):
        CT, CMx, CMy = compute_rotor_trim(x)  

        res = compute_trim_residual(x, target_CT, target_CMx, target_CMy)
        
        print(f"[Iteration {i+1}] Residual: {res},CT={CT:.6f}, CMx={CMx:.6f}, CMy={CMy:.6f}")

        if np.linalg.norm(res) < tol:
            print("Converged!")
            return x
        
        # Jacobian (finite difference 근사)
        J = np.zeros((3,3))
        delta = 1e-5
        for j in range(3):
            x_perturb = x.copy()
            x_perturb[j] += delta
            res_perturb = compute_trim_residual(x_perturb, target_CT, target_CMx, target_CMy)
            J[:, j] = (res_perturb - res) / delta
        
        dx = -np.linalg.solve(J, res)
        
        x += dx
        

    print("Not converged.")
    return x

# 초기 추정값
x_initial = np.array([np.radians((6.0)), np.radians(1.7), np.radians(-5.5)])

# 목표 trim 조건 
target_CT = 0.00464
target_CMx = 0
target_CMy = 0.00005

# Newton-Raphson trim 계산
trimmed_x = newton_trim(x_initial, target_CT, target_CMx, target_CMy)

theta_0_trim, theta_1c_trim, theta_1s_trim = np.degrees(trimmed_x)
print(f"Trimmed pitch angles:\n theta0={theta_0_trim:.3f}, theta1c={theta_1c_trim:.3f}, theta1s={theta_1s_trim:.3f}")
