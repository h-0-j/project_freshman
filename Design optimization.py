import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
import matplotlib.pyplot as plt

# ✅ Ackley function (vectorized)
def ackley(x):
    d = x.shape[1]
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1) / d))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / d)
    return term1 + term2 + np.e + 20

# ✅ Gradient-based wrapper
class Problem_GB:
    def __init__(self, benchmark_fun, num_var):
        self.num_var = num_var
        self.benchmark_fun = benchmark_fun

    def obj_func(self, x):
        return self.benchmark_fun(x[np.newaxis])[0]

    def minimize(self, seed=0):
        np.random.seed(seed)
        bnds = Bounds(-5 * np.ones(self.num_var), 5 * np.ones(self.num_var))
        random_init = np.random.uniform(-5, 5, self.num_var)
        res = minimize(self.obj_func, random_init, bounds=bnds, method='L-BFGS-B',
                       tol=1e-8, options={'maxiter': int(1e4)})
        return res.fun, np.linalg.norm(res.x), res.nfev

# ✅ pymoo용 문제 정의 (gradient-free)
class MyProblem(Problem):
    def __init__(self, benchmark_fun, num_var):
        super().__init__(n_var=num_var, n_obj=1, n_ieq_constr=0,
                         xl=-5 * np.ones(num_var), xu=5 * np.ones(num_var))
        self.benchmark_fun = benchmark_fun

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.benchmark_fun(x)

# ✅ 실험 자동화
dims = [2, 5, 10, 20, 50]
seeds = range(5)
records = []

for d in dims:
    for seed in seeds:
        # --- Gradient-based
        gb = Problem_GB(ackley, d)
        fval_gb, dist_gb, nfev_gb = gb.minimize(seed=seed)
        records.append({
            "method": "gradient-based", "dim": d, "seed": seed,
            "fval": fval_gb, "dist_to_opt": dist_gb, "n_eval": nfev_gb
        })

        # --- Gradient-free (GA)
        gf = MyProblem(ackley, d)
        algorithm = GA(
            pop_size=200,
            n_offsprings=200,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        termination = get_termination("n_gen", 200)
        res = pymoo_minimize(gf, algorithm, termination, seed=seed, verbose=False)
        records.append({
    "method": "gradient-free", "dim": d, "seed": seed,
    "fval": res.F[0],
    "dist_to_opt": np.linalg.norm(res.X),
    "n_eval": res.algorithm.evaluator.n_eval
        })

# ✅ DataFrame 저장 및 시각화
df = pd.DataFrame(records)
df.to_csv("ackley_optimization_results.csv", index=False)

# 📊 성능 그래프
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="dim", y="fval", hue="method", marker="o", errorbar="sd")
plt.title("Final Objective Value vs Input Dimension")
plt.xlabel("Input Dimension")
plt.ylabel("Objective Function Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("ackley_obj_vs_dim.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="dim", y="dist_to_opt", hue="method", marker="s", errorbar="sd")
plt.title("Distance to Optimum vs Input Dimension")
plt.xlabel("Input Dimension")
plt.ylabel("|| x_opt - x_true ||")
plt.grid(True)
plt.tight_layout()
plt.savefig("ackley_dist_vs_dim.png")
plt.show()
