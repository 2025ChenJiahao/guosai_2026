"""
调试脚本：验证问题1的遮蔽条件计算
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from config import *
from physics import missile_pos, cloud_center, shielding_distance, shield_distance_to_target

# 问题1参数
v_u = 120.0
theta = np.arctan2(-FY1[1], -FY1[0])  # 朝向假目标
t_d = 1.5
delta_t = 3.6
t_e = t_d + delta_t  # 5.1

# 起爆点
from physics import bomb_detonate_point
P_e = bomb_detonate_point(FY1, v_u, theta, t_d, delta_t)
print(f"起爆点 P_e = {P_e}")
print(f"FY1初始位置 = {FY1}")
print(f"M1初始位置 = {M1}")
print(f"真目标中心 = {TARGET_CENTER}")
print()

# 计算导弹到达时间
t_arrive = np.linalg.norm(M1) / V_M
print(f"M1到达原点时间 = {t_arrive:.2f} s")
print(f"遮蔽时间窗口: [{t_e}, {t_e + T_EFF}]")
print()

# 遍历时间，打印关键距离
print(f"{'t':>8s} | {'M_x':>10s} | {'M_y':>10s} | {'M_z':>10s} | "
      f"{'C_x':>10s} | {'C_y':>10s} | {'C_z':>10s} | "
      f"{'d_seg':>10s} | {'d_mc':>10s} | {'shielded':>8s}")
print("-" * 120)

dt = 0.5  # 粗粒度扫描
t = t_e
while t <= t_e + T_EFF:
    M = missile_pos(M1, t)
    C = cloud_center(P_e, t, t_e)
    d_seg = shield_distance_to_target(M, C, TARGET_CENTER)
    d_mc = np.linalg.norm(M - C)
    shielded = "YES" if d_seg <= R else "NO"

    print(f"{t:8.2f} | {M[0]:10.1f} | {M[1]:10.1f} | {M[2]:10.1f} | "
          f"{C[0]:10.1f} | {C[1]:10.1f} | {C[2]:10.1f} | "
          f"{d_seg:10.1f} | {d_mc:10.1f} | {shielded:>8s}")
    t += dt

# 细粒度扫描最小距离
print("\n\n细粒度扫描 (dt=0.01):")
min_d = float('inf')
min_t = 0
dt = 0.01
t = t_e
while t <= t_e + T_EFF:
    M = missile_pos(M1, t)
    C = cloud_center(P_e, t, t_e)
    d = shield_distance_to_target(M, C, TARGET_CENTER)
    if d < min_d:
        min_d = d
        min_t = t
    t += dt

print(f"最小遮蔽距离 = {min_d:.2f} m, 发生在 t = {min_t:.3f} s")
print(f"是否满足遮蔽条件 (d <= {R}m): {'YES' if min_d <= R else 'NO'}")

# 也检查导弹到云团的距离
print("\n\n导弹到云团中心距离扫描:")
min_d_mc = float('inf')
min_t_mc = 0
t = t_e
while t <= t_e + T_EFF:
    M = missile_pos(M1, t)
    C = cloud_center(P_e, t, t_e)
    d_mc = np.linalg.norm(M - C)
    if d_mc < min_d_mc:
        min_d_mc = d_mc
        min_t_mc = t
    t += dt

print(f"最小导弹-云团距离 = {min_d_mc:.2f} m, 发生在 t = {min_t_mc:.3f} s")
