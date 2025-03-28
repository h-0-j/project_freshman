import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import trapezoid  

# 🔹 AH-1G Rotor Parameters
R = 7.1287  # Rotor Radius (m)
c_R = 0.102  # Chord/R ratio
Nb = 2  # Number of blades
theta_tw = np.radians(-10)  # Twist angle (radians)
mu = 0.19  # Advance ratio
M_tip = 0.65  # Tip Mach number
rho = 1.225  # Air density (kg/m³)
a = 340  # Speed of sound (m/s) at sea level
U_tip = M_tip * a  # Tip speed
Omega = M_tip * a / R  # Rotor angular velocity (rad/s)

# 🔹 Airfoil Properties (NACA 0012)
Cl_alpha = 2 * np.pi  # Lift curve slope (rad⁻¹)
Cd0 = 0.011  # Profile drag coefficient

# 🔹 Target Trim Values
C_T_target = 0.000464
C_Mx_target = 0.0002
C_My_target = -0.0001
y_target = np.array([C_T_target, C_Mx_target, C_My_target])

# 🔹 Initial Guesses (Collective and Cyclic Angles)
theta_0 = np.radians(5)  # Initial collective pitch [rad]
theta_1c = np.radians(1)  # Initial longitudinal cyclic pitch [rad]
theta_1s = np.radians(-1)  # Initial lateral cyclic pitch [rad]
x = np.array([theta_0, theta_1c, theta_1s])  # Initial guess vector

# 🔹 Newton-Raphson Convergence Conditions
tolerance = 1e-6
max_iter = 100

# 🔹 Rotor Disk Azimuth and Radial Stations
n_psi = 36  # Azimuthal resolution
n_r =   # Radial resolution
psi_vals = np.linspace(0, 2*np.pi, n_psi)  # Azimuth angles (0-360 degrees)
r_vals = np.linspace(0.2, 1, n_r)  # Normalized radial positions (excluding hub)

# 🔹 Compute C_T, C_Mx, C_My from Blade Element Theory
def compute_aero_coefficients(x):
    theta_0, theta_1c, theta_1s = x
    C_T, C_Mx, C_My = 0, 0, 0

    for r_R in r_vals:
        r = r_R * R  # Actual radial position
        theta = theta_0 + theta_tw * (r_R - 0.2) + theta_1c * np.cos(psi_vals) + theta_1s * np.sin(psi_vals)

        # 🔹 Blade Element Velocities (BEMT)
        U_T = Omega * r + mu * U_tip * np.sin(psi_vals)  # Tangential velocity
        U_R = mu * U_tip * np.cos(psi_vals)  # Radial velocity
        U_P = 0.05 * U_tip  # Assumed uniform inflow

        # 🔹 Flow Angles
        phi = np.arctan(U_P / U_T)  # Inflow angle
        alpha = theta - phi  # Angle of attack

        # 🔹 Aerodynamic Forces (NACA 0012)
        Cl = Cl_alpha * alpha  # Lift coefficient
        Cd = Cd0 + (Cl**2) / (np.pi * 0.7 * 6)  # Induced drag model

        # 🔹 Sectional Force Coefficients
        dC_L = Cl * 0.5 * rho * U_T**2 * (c_R * R)
        dC_D = Cd * 0.5 * rho * U_T**2 * (c_R * R)
        dC_T = dC_L * np.cos(phi) - dC_D * np.sin(phi)  # Thrust
        dC_Mx = dC_T * np.sin(psi_vals)  # Roll moment
        dC_My = dC_T * np.cos(psi_vals)  # Pitch moment

        # 🔹 Summation
        C_T += trapezoid(dC_T * r_R, psi_vals) / (2 * np.pi)
        C_Mx += trapezoid(dC_Mx * r_R, psi_vals) / (2 * np.pi)
        C_My += trapezoid(dC_My * r_R, psi_vals) / (2 * np.pi)

    return np.array([C_T, C_Mx, C_My])

# 🔹 Compute Jacobian Matrix (J)
def compute_jacobian(x):
    delta = 1e-6
    J = np.zeros((3, 3))

    for i in range(3):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += delta
        x_backward[i] -= delta

        y_forward = compute_aero_coefficients(x_forward)
        y_backward = compute_aero_coefficients(x_backward)

        J[:, i] = (y_forward - y_backward) / (2 * delta)  # Central Difference

    return J

# 🔹 Newton-Raphson Iteration
for iteration in range(max_iter):
    y_current = compute_aero_coefficients(x)
    error = y_current - y_target

    if np.linalg.norm(error) < tolerance:
        print(f"✅ Converged in {iteration} iterations")
        break

    J = compute_jacobian(x)

    try:
        dx = np.linalg.solve(J, -error)  # Solve Δx = J⁻¹(y(x+Δx) - y_target)
    except np.linalg.LinAlgError:
        print("⚠️ Jacobian Matrix is Singular. Adjust initial guess.")
        break

    x = x + dx  # Update variables
    print(f"Iteration {iteration+1}: CT = {y_current[0]:.6f}, Error = {error[0]:.6f}")

# 🔹 Final Trimmed Angles
theta_0_final, theta_1c_final, theta_1s_final = np.degrees(x)
print(f"\n✅ Trimmed Angles for AH-1G:")
print(f"theta_0  = {theta_0_final:.4f}°")
print(f"theta_1c = {theta_1c_final:.4f}°")
print(f"theta_1s = {theta_1s_final:.4f}°")
