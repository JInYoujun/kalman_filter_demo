# -*- coding: utf-8 -*-
import json
import math

import numpy as np
import matplotlib.pyplot as plt

"""
X(k) = AX(k-1) + BU(k) + w(k-1)
Z(k) = HX(k) + e(k)
p(w) = N(0, Q)
p(e) = N(0, R)
"""


def kf_predict(X0, P0, A, Q, B, U1):
    # 一步预测
    """
    设状态量有xn个
    - X0为前一时刻状态量，shape=(xn,1)
    - P0为初始状态不确定度， shape=(xn,xn)
    - A为状态转移矩阵，shape=(xn,xn)
    - Q为递推噪声协方差矩阵，shape=(xn,xn)
    - B、U1是外部输入部分

    返回的结果为
    - X10为一步预测的状态量结果，shape=(xn,1)
    - P10为一步预测的协方差，shape=(xn,xn)
    """
    X10 = np.dot(A, X0) + np.dot(B, U1)
    P10 = np.dot(np.dot(A, P0), A.T) + Q
    return (X10, P10)


def kf_update(X10, P10, Z, H, R):
    """
    设状态量有xn个
    - X10为一步预测的状态量结果，shape=(xn,1)
    - P10为一步预测的协方差，shape=(xn,xn)
    - Z为观测值，shape=(xn,1)
    - H为观测系数矩阵，shape=(xn,xn)
    - R为观测噪声协方差，shape=(xn,xn)

    返回的结果为
    - X1为一步预测的状态量结果，shape=(xn,1)
    - P1为一步预测的协方差，shape=(xn,xn)
    - K为卡尔曼增益，不需要返回，但是可以看一下它的值来判断是否正常运行
    """
    # 测量更新
    V = Z - np.dot(H, X10)
    K = np.dot(np.dot(P10, H.T), np.linalg.pinv(np.dot(np.dot(H, P10), H.T) + R))
    X1 = X10 + np.dot(K, V)
    P1 = np.dot(np.eye(K.shape[0]) - np.dot(K, H), P10)
    return (X1, P1, K)


"""
匀速轨迹白噪声建模
离散化状态方程：
x(k) = x(k-1)+vx(k)*t+0.5*t^2*ax(k)
y(k) = y(k-1)+vy(k)*t+0.5*t^2*ay(k)
v(k) = v(k-1)+t*a(k)
a(k) = a(k-1)

观测方程：
z(k) = x(k) + e

"""

gps_data_file = open('./gpsdata.json', 'r', errors="ignore")
gps_data = json.loads(''.join(gps_data_file.readlines()))

n = len(gps_data['gpsdata'])  # 数据量
nx = 4  # 变量数量 x, y, v, a
t = np.linspace(0, 3, n)  # 时间序列
index = 0
last_gps = {}
for gps in gps_data['gpsdata']:
    if index == 0:
        last_gps = gps
        index += 1
        continue
    t[index] = gps['time'] - last_gps['time']
    last_gps = gps
    index += 1

# 真实函数关系
a_true = np.ones(n) * 9.8 + np.random.normal(0, 1, size=n)
v_true = np.cumsum(a_true * t)
x_true = np.cumsum(v_true * t)
y_true = np.cumsum(v_true * t)
hdg_true = np.ones(n)
index = 0
for gps in gps_data['gpsdata']:
    a_true[index] = gps['acc']
    v_true[index] = gps['spd']
    x_true[index] = gps['x']
    y_true[index] = gps['y']
    hdg_true[index] = gps['hdg']
    index += 1

X_true = np.concatenate([x_true, y_true, v_true, a_true]).reshape([nx, -1])

# 观测噪声协方差！！！！！！！！！！！！！！！！！！！！（可调整）
R = np.diag([10 ** 2])

# 仿真观测值
e = np.random.normal(0, 1, n)
x_obs = x_true + e
y_obs = y_true + e

# 计算系数
A = np.array([1, 0, math.cos(math.pi / 180 * (360 + 90 - hdg_true[0])) * (t[1] - t[0]),
              0.5 * math.cos(math.pi / 180 * (360 + 90 - hdg_true[0])) * (t[1] - t[0]) * (t[1] - t[0]) ** 2,
              0, 1, math.sin(math.pi / 180 * (360 + 90 - hdg_true[0])) * (t[1] - t[0]),
              0.5 * math.sin(math.pi / 180 * (360 + 90 - hdg_true[0])) * (t[1] - t[0]) * (t[1] - t[0]) ** 2,
              0, 0, 1, t[1] - t[0],
              0, 0, 0, 1]).reshape([nx, nx])
B = 0.03
U1 = -0.5

# 状态假设（观测）初始值
x0 = x_true[0]
y0 = y_true[0]
v0 = v_true[0]
a0 = a_true[0]
X0 = np.array([x0, y0, v0, a0]).reshape(nx, 1)

# 初始状态不确定度！！！！！！！！！！！！！！！！（可调整）
P0 = np.diag([0 ** 2, 0 ** 2, 0 ** 2, 0.2 ** 2])

# 状态递推噪声协方差！！！！！！！！！！！！！！！！！！（可调整）
Q = np.diag([0.0 ** 2, 0.0 ** 2, 0 ** 2, 1 ** 2])

###开始处理
X1_np = np.copy(X0)
P1_list = [P0]
X10_np = np.copy(X0)
P10_list = [P0]

for i in range(n):
    Zi = np.array([x_obs[i], y_obs[i]]).reshape([2, 1])
    Hi = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]).reshape([2, nx])
    if (i == 0):
        continue
    else:
        Xi = X1_np[:, i - 1].reshape([nx, 1])
        Pi = P1_list[i - 1]
        A = np.array([1, 0, math.cos(math.pi / 180 * (360 + 90 - hdg_true[i])) * (t[i] - t[i - 1]),
                      0.5 * math.cos(math.pi / 180 * (360 + 90 - hdg_true[i])) * (t[i] - t[i - 1]) * (
                                  t[i] - t[i - 1]) ** 2,
                      0, 1, math.sin(math.pi / 180 * (360 + 90 - hdg_true[i])) * (t[i] - t[i - 1]),
                      0.5 * math.sin(math.pi / 180 * (360 + 90 - hdg_true[i])) * (t[i] - t[i - 1]) * (
                                  t[i] - t[i - 1]) ** 2,
                      0, 0, 1, t[1] - t[0],
                      0, 0, 0, 1]).reshape([nx, nx])
        X10, P10 = kf_predict(Xi, Pi, A, Q, B, U1)

        X10_np = np.concatenate([X10_np, X10], axis=1)
        P10_list.append(P10)

        X1, P1, K = kf_update(X10, P10, Zi, Hi, R)
        X1_np = np.concatenate([X1_np, X1], axis=1)
        P1_list.append(P1)

# 结束，绘图
print('filter ', X1_np)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x_true, y_true, 'k-', label="Truth")
ax1.plot(X1_np[0, :], X1_np[1, :], 'go--', label="Kalman Filter")
ax1.scatter(x_obs, y_obs, label="Observation", marker='*')

plt.legend()
plt.show()
