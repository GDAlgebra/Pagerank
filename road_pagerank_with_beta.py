import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# 定义符号
b = sp.symbols('b')

# 函数：生成解和关系
def solve_system(k):
    # 定义变量
    variables = [sp.symbols(f'x{i+1}') for i in range(k)]

    # 定义方程组
    eqns = []
    if k == 1:
        eqns.append(variables[-1] - 1 + 1 / (1 - b))  # 特殊情况：k=1 时只计算这一项
    else:
        eqns.append(variables[0] - b * variables[1])  # x1 = b * x2
        for i in range(1, k - 1):
            eqns.append(variables[i] - b / 2 * (variables[i - 1] + variables[i + 1]))
        eqns.append(variables[-1] + b / (2 * (1 - b)) - b / 2 * variables[-2])  # xk

    # 解方程组
    solutions = sp.solve(eqns, variables)

    # 提取每个解
    solutions_lambdified = [sp.lambdify(b, solutions[var], "numpy") for var in variables]

    return solutions_lambdified

# 设置参数
n = 4  # 总变量数

# 生成 b 的取值范围
b_vals = np.linspace(0.995, 0.999, 100)

# 循环计算不同的 n1 和 n2
for n1 in range(1, n - 1):
    n2 = n - n1

    # 计算 n1 个变量的解
    solutions_n1 = solve_system(n1)
    solution_n1_vals = [sol(b_vals) for sol in solutions_n1]

    # 计算 n2 个变量的解
    solutions_n2 = solve_system(n2)
    solution_n2_vals = [sol(b_vals) for sol in solutions_n2]

    # 提取指定变量的值
    x_n1_vals = solution_n1_vals[n1 - 1]  # 对应 x_n1
    y_n2_1_vals = solution_n2_vals[n2 - 2]  # 对应 y_(n2-1)

    # 绘制当前 n1 和 n2 的关系图
    plt.figure(figsize=(8, 6))
    plt.plot(b_vals, x_n1_vals, label=f"x_{n1} (n1={n1}, n2={n2})", linewidth=2)
    plt.plot(b_vals, y_n2_1_vals, label=f"y_{n2-1} (n1={n1}, n2={n2})", linewidth=2, linestyle='--')

    # 添加图例和标签
    plt.legend()
    plt.xlabel("b")
    plt.ylabel("Value")
    plt.title(f"Relationship between x_{n1} and y_{n2-1} (n1={n1}, n2={n2})")
    plt.grid(True)
    plt.show()
