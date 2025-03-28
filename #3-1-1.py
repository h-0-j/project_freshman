import numpy as np
import matplotlib.pyplot as plt

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
n_psi = 36
n_r = 14
psi_vals = np.linspace(0, 2 * np.pi, n_psi)
r_vals = np.linspace(0.2, 1, n_r)

# 🔹 Trimmed Values
theta_0 = np.radians(5.62)
theta_1c = np.radians(0.64)
theta_1s = np.radians(-4.84)
x = np.array([theta_0, theta_1c, theta_1s])

def compute_aero_coefficients(x):
    theta_0, theta_1c, theta_1s = x
    C_n_total = np.zeros((n_psi, n_r))

    C_T = 0.00464
    lambda_h = np.sqrt(C_T/2)
    lambda_0 = lambda_h*(np.sqrt(1/4*(mu/lambda_h)**4 + 1) - 1/2*(mu/lambda_h)**2)**1/2  # Linear inflow

    for j, r_R in enumerate(r_vals):
        if r_R < 1e-6:
            continue
        r = r_R * R
        theta = theta_0 + theta_tw * (r_R - 0.75) + \
                theta_1c * np.cos(psi_vals) + theta_1s * np.sin(psi_vals)

        beta = beta_1c * np.cos(psi_vals) + beta_1s * np.sin(psi_vals)
        beta_dot = - Omega * (beta_1c * np.sin(psi_vals) - beta_1s * np.cos(psi_vals))

        x_angle = np.arctan2(mu , (1e-6 + lambda_0 ))
        k_x = 4/3 * ((1 - np.cos(x_angle) - 1.08 * mu**2) / np.maximum(np.abs(np.sin(x_angle)), 0.05))
        k_y = -2 * mu
        lambda_i = ( 1 + k_x*np.cos(psi_vals)/r_R + k_y*np.sin(psi_vals)/r_R ) * lambda_0


        U_T = Omega * r + mu * U_tip * np.sin(psi_vals)
        U_P = (lambda_i + (r * beta_dot) / Omega + mu * beta * np.cos(psi_vals)) * U_tip
        U_eff = np.sqrt(U_T**2 + U_P**2)

        phi = np.arctan(U_P / (U_T + 1e-6))
        alpha = theta - phi

        Cl_alpha = 2 * np.pi
        Cl = Cl_alpha * alpha
        Cd0 = 0.011
        Cd = Cd0 + (Cl**2) / (np.pi * 0.7 * 6)

        dpsi = 2*np.pi /n_psi
        dr = (r_vals[1] - r_vals[0]) * r_R
        dL = 0.5 * rho * chord * Cl * U_eff**2 * dr
        dD = 0.5 * rho * chord * Cd * U_eff**2 * dr
        dT = (dL * np.cos(phi) - dD * np.sin(phi))*dpsi

        area_section = chord * dr
        C_n_total[:, j] = dT / (0.5 * rho * U_tip**2 * area_section)
        for i, psi in enumerate(psi_vals):
            print(f"psi = {np.degrees(psi):.1f} deg, theta = {np.degrees(theta[i]):.4f} deg")


    C_n_total = np.nan_to_num(C_n_total, nan=0.0)
    return C_T, C_n_total, theta


# 🔹 Linear inflow 모델 결과 계산
_, C_n_total, _ = compute_aero_coefficients(x)

## project에서 가져온 점.
x_60 = [
    346.37454753286346, 337.10488272275956, 326.50521667992786, 316.56449968061145,
    303.9785729494694, 290.72293882308117, 276.8043212712786, 264.87599878968547,
    255.59557562785068, 242.99889054497777, 235.04174464603904, 221.77669696188633,
    213.80879271121677, 205.18328421099818, 194.56344625867115, 189.90776954714062,
    179.2812076249818, 171.31330337431223, 161.35510405343314, 154.04883843421158,
    144.75496733271325, 134.79811280580054, 119.53201169970752, 104.27263456344626,
    85.0367016686652, 65.81959588941312, 45.9516098304439, 34.036735288514336,
    23.453206773278936, 13.529972095525196, 3.6174957695023124
]
y_60 = [
    0.6275862068965518, 0.6531440162271805, 0.6701825557809331, 0.6787018255578094,
    0.701419878296146, 0.7099391480730224, 0.7184584178498986, 0.7298174442190669,
    0.7326572008113591, 0.7326572008113591, 0.7298174442190669, 0.7184584178498986,
    0.6929006085192698, 0.6787018255578094, 0.6531440162271805, 0.6219066937119676,
    0.5821501014198784, 0.5565922920892495, 0.5281947261663286, 0.4997971602434078,
    0.4742393509127789, 0.4486815415821501, 0.41176470588235303, 0.3890466531440162,
    0.3691683569979718, 0.3890466531440162, 0.43448275862068964, 0.4742393509127789,
    0.5253549695740365, 0.57079107505071, 0.6389452332657202
]

x_75 = [
    361.6359680614237, 350.93820469986326, 342.5778169052198, 334.67789014890474,
    324.44643288821857, 316.07630332266547, 297.4727681422443, 288.1641813123969,
    280.7130254693189, 269.06706307255456, 257.41915232160835, 247.14677962310174,
    236.40810082372087, 230.79878913395657, 219.10411788264418, 213.01486161273215,
    208.7849846837713, 201.7494777328373, 193.78720364270072, 182.0886356830244,
    177.8626554624274, 171.312938154187, 166.62454854107958, 160.52944720862183,
    155.83910924133252, 150.2122623639309, 142.71629437466848, 135.2320165104977,
    125.40646637104601, 119.32110680949788, 107.16792287403891, 98.30031354070076,
    79.66560469336872, 67.09872021994755, 56.85752118835171, 48.51272022716367,
    42.0156084818352, 35.05608734399637, 25.776725826877907, 16.509054434850995,
    13.754081621607952, 9.625519110107263, 7.358284293739658, 2.7653640355466536
]

y_75 = [
    0.5027475402028452, 0.5224944706429928, 0.5463654174348833, 0.5660474027356335,
    0.5857835089858815, 0.6026908936090374, 0.6379416720486943, 0.6506925677504085,
    0.6578293169575365, 0.6664561963075079, 0.6736903632237322, 0.6641795083652946,
    0.6546794776967566, 0.645060380939323, 0.618869449445621, 0.5995121898418584,
    0.5759334961772569, 0.5468488979170649, 0.521964085337913, 0.49298772897671705,
    0.4721944601796093, 0.4570261620669872, 0.43902914232727275, 0.4154937454222696,
    0.3961040132488083, 0.37395050458765233, 0.34905486781860084, 0.33251550565203103,
    0.30906670226622446, 0.2924948675299558, 0.2718856099611411, 0.26652041983424557,
    0.2794877993339514, 0.2964926016662035, 0.3092651457477169, 0.34427779200958286,
    0.36671272960812806, 0.39194391626406677, 0.4255854984719851, 0.46758335528238504,
    0.4982879739642151, 0.5471303268544542, 0.5931439581176011, 0.6433898476314868
]

x_91 = [
    0.8767724417870966, 6.576780669756715, 10.004881928636067, 16.844299388387583,
    23.68766627355255, 27.110830750665073, 33.38844244535255, 38.52269548284467,
    45.93379227119385, 52.776171800005486, 60.18924330106141, 71.58333561888044,
    80.12594278818462, 92.64265927977836, 104.01502975782343, 115.37456460327472,
    121.05087627876357, 128.9892213598091, 138.07191245440333, 143.7393379227119,
    153.38383478237017, 158.48155563478787, 162.4497408189572, 164.14108225226954,
    170.3801870491758, 176.6192918460821, 180.0128356325937, 186.24799100408654,
    191.92430267957542, 201.57176160829374, 207.8009928416664, 216.31496667672303,
    220.85186912043002, 228.23433257453166, 236.1865006445243, 246.97929294314469,
    255.5070897671484, 267.4511395737911, 282.24963659800875, 294.2045473245385,
    305.5956775732974, 319.8284194070376, 328.376950714462, 335.2143934615068,
    339.7730177449878, 346.6173719865061, 351.74767559858475, 357.44867118290773,
    360.30114368777595
]

y_91 = [
    0.36529771536710465, 0.3472861413565179, 0.32509037053289824, 0.30430981048243344,
    0.27797372535037446, 0.26272236087874723, 0.23221524368503343, 0.21003263761279178,
    0.18508982200159096, 0.16014261813993036, 0.13242203998793245, 0.10473217958915015,
    0.08813142810125885, 0.08128356325937303, 0.08414909080935795, 0.10507007487452358,
    0.12039165135350105, 0.15378623734949692, 0.17746743095362172, 0.20528893886618582,
    0.23869668961355972, 0.26790269054606297, 0.2859888648144594, 0.3068352486218151,
    0.3304945009736432, 0.35415375332547117, 0.3805688269657992, 0.40978360439922124,
    0.42510518087819876, 0.45434628781437714, 0.4918943528701902, 0.5155711582238558,
    0.5336617207427115, 0.5489964619730673, 0.5629467101834839, 0.5810855434573929,
    0.5853180110254794, 0.5840212830147282, 0.5674688022818903, 0.5508943802967555,
    0.5273711637091689, 0.506647650914676, 0.4817136118043939, 0.463710814294726,
    0.45124598886481465, 0.4235210224623571, 0.4068939414717096, 0.3874934861907245,
    0.37501549600943485
]

x_99 = [-0.6332077996352723, 2.769798534600902, 7.8927604104844065, 12.439813965533993,
              17.55306406534946, 22.098175265185446, 26.649113530662234, 31.192282375284606,
              32.92194969299996, 38.60139633758135, 44.288612403017126, 48.28306589979607,
              51.70355343095467, 63.5917385158248, 73.21319506641777, 77.18045559020624,
              86.22600381996526, 92.44445403632204, 104.27631081999763, 108.7689784290663,
              116.07708992025553, 121.69243884278791, 124.47486268627728, 129.51236093276214,
              135.12188278965374, 140.17297752263383, 143.5351943973843, 146.31761824087363,
              153.04593670080178, 158.0950890785683, 163.14424145633478, 167.0590583893559,
              169.27334333286575, 174.3069568689234, 178.76854679457435, 185.50851938578413,
              194.46277692050376, 201.18138360436384, 207.9174714851464, 214.65161701071534,
              223.61947103193015, 235.4416160395377, 247.83578465755195, 257.9904177142795,
              272.0948300978731, 280.01186994852765, 294.1531870811797, 304.34278253175216,
              312.8299036375997, 317.3614183509404, 321.3306212299425, 327.0042408088831,
              332.67203332218276, 336.65094797725294, 341.76031336664107, 344.5971231561115,
              346.86773640081583, 348.007898911202, 351.41478995586533, 353.6873455557834,
              354.82556571095597]

y_99 = [0.29036079248092705, 0.2795515857172146, 0.25189217770392025, 0.23145590314121994,
              0.20982049400567598, 0.19058901921852567, 0.16774314510472527, 0.14971647009312516,
              0.12684226996579306, 0.10400772625740518, 0.07635398344681721, 0.04868324502811017,
              0.027030840284447133, 0.003053814030279156, -0.014921874156963777, -0.025725415717970046,
              -0.03647797045461898, -0.04364445187814969, -0.03268228464136569, -0.019384165488664173,
              -0.0024433209957808177, 0.014480527888983574, 0.038604849413516606, 0.0639566315244251,
              0.08449487973583969, 0.10141306341789769, 0.11590465194073651, 0.14002897346526966,
              0.16660255095984716, 0.18472553441745532, 0.20284851787506342, 0.22457457025390898,
              0.2511028261268358, 0.27886420778884435, 0.3114391233503469, 0.33078390219162407,
              0.37665693690582813, 0.409254513278156, 0.43100889167053336, 0.4539680698384607,
              0.49140750612381434, 0.5083936722383485, 0.5205663044533888, 0.5218730778776531,
              0.5232195077208619, 0.5124556225788002, 0.4909108566865578, 0.470531234150921,
              0.4561586148849154, 0.4453607385266156, 0.43335239719005947, 0.4141322528083219,
              0.39852650775323456, 0.380494167538928, 0.3612683579544842, 0.35165828576361535,
              0.3432473481455903, 0.3360298798977026, 0.32281107358289, 0.31319533618931494,
              0.3071826677169773]

# 🔹 Plot at r/R = 0.60,0.75,0.91,0.99
for r_target in [0.6]:
    Cn_interp_lin = np.zeros_like(psi_vals)
    for i, psi in enumerate(psi_vals):
        Cn_interp_lin[i] = np.interp(r_target, r_vals, C_n_total[i, :])

    print(f"[DEBUG] Cn min: {Cn_interp_lin.min():.3f}, max: {Cn_interp_lin.max():.3f}")

    plt.plot(360 - np.degrees(psi_vals), Cn_interp_lin, label="Linear inflow", linestyle='--', linewidth=2)


plt.scatter(x_60,y_60,label="project")
plt.grid(True)
plt.xlabel("Azimuth, deg")
plt.ylabel("$C_n$")
plt.legend()
plt.title("Cn at r/R = 60% (Linear Inflow Model)")
plt.show()

for r_target in [0.75]:
    Cn_interp_lin = np.zeros_like(psi_vals)
    for i, psi in enumerate(psi_vals):
        Cn_interp_lin[i] = np.interp(r_target, r_vals, C_n_total[i, :])

    print(f"[DEBUG] Cn min: {Cn_interp_lin.min():.3f}, max: {Cn_interp_lin.max():.3f}")

    plt.plot(360 - np.degrees(psi_vals), Cn_interp_lin, label="Linear inflow", linestyle='--', linewidth=2)

plt.scatter(x_75,y_75,label="project")
plt.grid(True)
plt.xlabel("Azimuth, deg")
plt.ylabel("$C_n$")
plt.legend()
plt.title("Cn at r/R = 75% (Linear Inflow Model)")
plt.show()

for r_target in [0.91]:
    Cn_interp_lin = np.zeros_like(psi_vals)
    for i, psi in enumerate(psi_vals):
        Cn_interp_lin[i] = np.interp(r_target, r_vals, C_n_total[i, :])

    print(f"[DEBUG] Cn min: {Cn_interp_lin.min():.3f}, max: {Cn_interp_lin.max():.3f}")

    plt.plot(360 - np.degrees(psi_vals), Cn_interp_lin, label="Linear inflow", linestyle='--', linewidth=2)
plt.scatter(x_91,y_91,label="project")
plt.grid(True)
plt.xlabel("Azimuth, deg")
plt.ylabel("$C_n$")
plt.legend()
plt.title("Cn at r/R = 91% (Linear Inflow Model)")
plt.show()

for r_target in [0.99]:
    Cn_interp_lin = np.zeros_like(psi_vals)
    for i, psi in enumerate(psi_vals):
        Cn_interp_lin[i] = np.interp(r_target, r_vals, C_n_total[i, :])

    print(f"[DEBUG] Cn min: {Cn_interp_lin.min():.3f}, max: {Cn_interp_lin.max():.3f}")

    plt.plot(360 - np.degrees(psi_vals), Cn_interp_lin, label="Linear inflow", linestyle='--', linewidth=2)
plt.scatter(x_99,y_99,label="project")
plt.grid(True)
plt.xlabel("Azimuth, deg")
plt.ylabel("$C_n$")
plt.legend()
plt.title("Cn at r/R = 99% (Linear Inflow Model)")
plt.show()