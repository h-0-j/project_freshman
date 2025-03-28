import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
# 주어진 변수들
theta_0 = np.radians(5.62)
theta_1c = np.radians(0.64)
theta_1s = np.radians(-4.84)
R = 7.1287  # m
R_cut = 0.2  # R의 20%
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

n_psi = 36
n_r = 14
psi_vals = np.linspace(0, 2 * np.pi, n_psi)
r_vals = np.linspace(0, 1, n_r)

# 블레이드 플랩각 모델
def beta(psi):
    return beta_1c * np.cos(psi) + beta_1s * np.sin(psi)

# 받음각 모델
def theta(r, psi):
    return theta_0 + theta_tw * r + theta_1c * np.cos(psi) + theta_1s * np.sin(psi)

# section별 Cn 계산
def compute_sectional_Cn(psi_vals, r_vals):
    Cn_matrix = np.zeros((len(psi_vals), len(r_vals)))
    for i, psi in enumerate(psi_vals):
        for j, r in enumerate(r_vals):
            local_theta = theta(r, psi)
            local_beta = beta(psi)
            alpha = local_theta - local_beta
            cl = 2 * np.pi * alpha
           
            dCn = cl * c_R * r * (r * R) / R**2
            Cn_matrix[i, j] = dCn
    return Cn_matrix

# 계산
Cn_sectional = compute_sectional_Cn(psi_vals, r_vals)

# 피치 각도(theta) 함수
def theta_pitch(target_r, psi):
    return theta_0 + theta_tw * (target_r-0.75) + theta_1c * np.cos(psi) + theta_1s * np.sin(psi)

# # Effective AOA 계산 함수
# def compute_effective_aoa(psi_vals, r_vals):
#     aoa_matrix = np.zeros((len(psi_vals), len(r_vals)))
#     for i, psi in enumerate(psi_vals):
#         for j, r in enumerate(r_vals):
#             local_theta = theta_pitch(r, psi)
#             local_beta = beta(psi)
#             aoa = local_theta - local_beta
#             aoa_matrix[i, j] = np.degrees(aoa)  # degree로 저장
#     return aoa_matrix

# 계산
# aoa_matrix = compute_effective_aoa(psi_vals, r_vals)

# r/R = 0.6에 대한 정확한 Cn 곡선 구하기 (보간 사용)
target_r = 0.6
Cn_r_06 = np.zeros_like(psi_vals)
for i, psi in enumerate(psi_vals):
    Cn_r_06[i] = np.interp(target_r, r_vals, Cn_sectional[i, :])

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(np.degrees(psi_vals), Cn_r_06, marker='o')
plt.xlabel("Azimuth Angle ψ (degrees)")
plt.ylabel("$C_n$")
plt.grid(True)
plt.tight_layout()

# aoa_masked = np.copy(aoa_matrix)
# for j, r in enumerate(r_vals):
#     if r < R_cut:
#         aoa_masked[:, j] = np.nan
        
# aoa_masked = ma.masked_invalid(aoa_masked)
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
# R_grid, Psi_grid = np.meshgrid(r_vals, psi_vals)

# # 도넛 중심 만들기
# ax.set_rmin(R_cut)

# # 등고선 및 회색조 컬러맵 적용
# contour = ax.contourf(Psi_grid, R_grid, aoa_masked, levels=np.linspace(-10, 10, 21), cmap='gray')
# ax.contour(Psi_grid, R_grid, aoa_masked, levels=np.linspace(-10, 10, 21), colors='black', linewidths=0.5)

# # 컬러바 설정 (DEG 레이블)
# cbar = plt.colorbar(contour, ax=ax, orientation='vertical', ticks=np.arange(-11, 11, 1))
# cbar.set_label("DEG", fontsize=12, fontweight='bold')  # 레이블 설정
# cbar.ax.tick_params(labelsize=10)  # 숫자 크기 조정
# cbar.ax.set_yticklabels([str(i) for i in range(-11, 11)])  # 숫자 직접 기입
# cbar.outline.set_edgecolor("black")  # 테두리 색 설정
# cbar.outline.set_linewidth(1)  # 테두리 두께 설정

# # 방향 및 위치 설정
# ax.set_theta_zero_location('S')   # ψ = 0°를 오른쪽으로
# ax.set_theta_direction(1)         # 시계 방향으로 증가
# ax.set_xticks(np.radians([0, 90, 180, 270]))  # 0°, 90°, 180°, 270° 표시
# ax.set_xticklabels(["ψ=0°", "90°", "180°", "270°"])
# ax.set_ylim(0.2, 1.0)
# plt.tight_layout()


plt.show()
