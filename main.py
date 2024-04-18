import numpy as np
import pandas as pd
from src import calculate as calc  
from tabulate import tabulate

# 加载数据
x = np.loadtxt("data/x.txt", delimiter=",")
y = np.loadtxt("data/y.txt", delimiter=",")
index = np.loadtxt("data/index.txt", delimiter=",", dtype=bool)
names = np.loadtxt("data/names.txt", delimiter=",", dtype=str)

# 实例化Calculate类
analysis = calc.calculate(x, y, index, names)

# 执行交叉验证并打印结果
cv_result = analysis.cross_validation(10)
cv_error, cv_variables = cv_result
cv_df = pd.DataFrame({
    "交叉验证最小误差": [cv_error],
    "最优变量组合": [", ".join(cv_variables)]
})
print("交叉验证结果:")
print(tabulate(cv_df, headers='keys', tablefmt='psql', showindex=False))

# 实例化Calculate类进行最优子集回归分析
osr_instance = calc.calculate(x, y, index, names)
osr_result = osr_instance.osr()
osr_variables, osr_b, osr_rss, osr_err_test, osr_num_vars = osr_result
osr_df = pd.DataFrame({
    "变量组合": [", ".join(osr_variables)],
    "回归系数": [osr_b],
    "训练误差": [osr_rss],
    "测试误差": [osr_err_test],
    "变量数量": [osr_num_vars]
})
print("最优子集回归结果:")
print(tabulate(osr_df, headers='keys', tablefmt='psql', showindex=False))
