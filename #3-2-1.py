import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy.integrate import trapezoid

Cl_alpha = 2 * np.pi
R = 7.1287
R_cut = 0.20
c_R = 0.102
Nb = 2
chord = R * c_R
theta_tw = np.radians(-10)
mu = 0.19
rho = 1.225
M_tip = 0.65
a = 340
U_tip = M_tip * a
omega = U_tip / R
sigma = Nb * chord / (np.pi * R)
area_section = np.pi * R**2

CT_req = 0.00464
beta_1c = np.radians(2.13)
beta_1s = np.radians(-0.15)

# x = np.array([theta0, theta1c, theta1s])
x = np.array([np.radians((5.62)), np.radians(0.64), np.radians(-4.84)])
y_target = np.array([CT_req, 0, 0])

# 격자 설정
n_psi = 36
n_r = 14
psi_vals = np.linspace(0, 2 * np.pi, n_psi)
r_vals = np.linspace(R_cut, 1, n_r)
dr = (r_vals[1] - r_vals[0]) * R
dpsi = 2 * np.pi / (n_psi - 1)
max_iter = 100

def compute_lambda_0(CT, mu):
    lambda_h = np.sqrt(CT / 2)
    lambda_0 = lambda_h * np.sqrt(
        (np.sqrt(0.25 * (mu / lambda_h)**4 + 1) - 0.5 * (mu / lambda_h)**2)
    )
    return lambda_0

def compute_lambda_distribution(lambda_0, mu, psi_vals, r_R):
    mu_x = mu
    mu_z = 0
    x_angle = np.arctan2(mu_x, mu_z + lambda_0)
    k_x = 4 / 3 * ((1 - np.cos(x_angle) - 1.8 * mu**2) / np.sin(x_angle))
    k_y = -2 * mu
    lambda_i = (1 + k_x * np.cos(psi_vals) / r_R + k_y * np.sin(psi_vals) / r_R) * lambda_0
    return lambda_i

def compute_aerodynamics(theta_0, theta_1c, theta_1s):
    lambda_0 = compute_lambda_0(CT_req, mu)
    CT, CMx, CMy = 0, 0, 0

    for j, r_R in enumerate(r_vals):
        r = r_R * R
        theta = theta_0 + theta_tw * (r_R - 0.75) + theta_1c * np.cos(psi_vals) + theta_1s * np.sin(psi_vals)

        k_flap = 0.25
        beta_0 = k_flap * theta_0
        beta_1c_eff = beta_1c + k_flap * theta_1c
        beta_1s_eff = beta_1s + k_flap * theta_1s

        beta = beta_0 + beta_1c_eff * np.cos(psi_vals) + beta_1s_eff * np.sin(psi_vals)
        beta_dot = -omega * (beta_1c_eff * np.sin(psi_vals) - beta_1s_eff * np.cos(psi_vals))

        lambda_i = compute_lambda_distribution(lambda_0, mu, psi_vals, r_R)

        U_T = omega * r + mu * omega * R * np.sin(psi_vals)
        U_P = omega * R * lambda_i + r * beta_dot + omega * R * mu * beta * np.cos(psi_vals)
        U_eff = np.sqrt(U_T**2 + U_P**2)

        phi = np.arctan2(U_P, U_T)
        alpha = theta - phi
        Cl = Cl_alpha * alpha
        Cd0 = 0.011
        Cd = Cd0 + (Cl**2) / (np.pi * 0.7 * 6)

        B = 1 - np.sqrt(CT_req) / Nb

        dL = 0.5 * rho * chord * Cl * U_eff**2 * dr * B
        dD = 0.5 * rho * chord * Cd * U_eff**2 * dr * B

        dFx = dL * np.sin(phi) + dD * np.cos(phi)
        dFz = dL * np.cos(phi) - dD * np.sin(phi)

        arm_x = r * np.sin(psi_vals)
        arm_y = r * np.cos(psi_vals)

        dT = dFz

        CT += np.sum(Nb * dT * dpsi) / (np.pi * rho * (omega * R)**2 * area_section)
        CMx += np.sum(Nb * dFz * arm_x * dpsi) / (np.pi * rho * (omega * R)**2 * R * area_section)
        CMy += np.sum(Nb * dFx * arm_y * dpsi) / (np.pi * rho * (omega * R)**2 * R * area_section)

    #return CT / (2 * np.pi), CMx / (2 * np.pi), - CMy / (2 * np.pi)
    return CT , CMx , CMy 
def compute_jacobian(x):
    J = np.zeros((3, 3))
    delta = 1e-7
    for i in range(3):
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[i] += delta
        x_bwd[i] -= delta
        y_fwd = np.array(compute_aerodynamics(*x_fwd))
        y_bwd = np.array(compute_aerodynamics(*x_bwd))
        J[:, i] = (y_fwd - y_bwd) / (2 * delta)
    return J

# 뉴턴 반복 수행
for iteration in range(1, 100):
    CT_new, CMx, CMy = compute_aerodynamics(*x)
    error = np.array([CT_new, CMx, CMy]) - y_target
    print(f"Iteration {iteration}: C_T={CT_new:.6f}  C_Mx={CMx:.6f}  C_My={CMy:.6f}")
    if np.linalg.norm(error) < 1e-6:
        print("Converged @", iteration)
        break
    J = compute_jacobian(x)
    dx = np.linalg.pinv(J) @ -error

    damping_factor = 0.5
    x +=  damping_factor * dx

# 결과 출력
trimmed_theta = x * 180 / np.pi
print(f"theta_0  = {trimmed_theta[0]: .4f} deg")
print(f"theta_1c = {trimmed_theta[1]: .4f} deg")
print(f"theta_1s = {trimmed_theta[2]: .4f} deg")