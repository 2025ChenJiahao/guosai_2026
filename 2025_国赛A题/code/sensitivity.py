"""
灵敏度分析：考察关键参数变化对遮蔽时长的影响
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from config import *
from physics import bomb_detonate_point, missile_pos, cloud_center, shield_distance_to_target


def compute_shield_with_params(v_m=300.0, v_c=3.0, R_eff=10.0, T_eff=20.0):
    """使用指定参数计算问题2最优策略的遮蔽时长"""
    # 问题2最优参数
    theta = np.radians(8.20)
    v_u = 91.39
    t_d = 0.0
    delta_t = 0.9
    t_e = t_d + delta_t

    P_u0 = FY1
    P_m0 = M1
    P_e = bomb_detonate_point(P_u0, v_u, theta, t_d, delta_t)

    if P_e[2] < 0:
        return 0.0

    total = 0.0
    dt = 0.01
    t = t_e
    while t <= t_e + T_eff:
        # 使用自定义v_m计算导弹位置
        d_m = -P_m0 / np.linalg.norm(P_m0)
        P_m = P_m0 + v_m * t * d_m

        # 使用自定义v_c计算云团位置
        P_c = P_e + np.array([0, 0, -v_c]) * (t - t_e)

        d = shield_distance_to_target(P_m, P_c, TARGET_CENTER)
        if d <= R_eff:
            total += dt
        t += dt
    return total


def sensitivity_R():
    """有效半径R的灵敏度分析"""
    R_values = np.arange(5, 16, 1)
    durations = [compute_shield_with_params(R_eff=r) for r in R_values]
    return R_values, durations


def sensitivity_Teff():
    """有效遮蔽时间T_eff的灵敏度分析"""
    T_values = np.arange(10, 31, 2)
    durations = [compute_shield_with_params(T_eff=t) for t in T_values]
    return T_values, durations


def sensitivity_vc():
    """云团下沉速度v_c的灵敏度分析"""
    vc_values = np.arange(1.0, 5.5, 0.5)
    durations = [compute_shield_with_params(v_c=v) for v in vc_values]
    return vc_values, durations


def sensitivity_vm():
    """导弹速度v_m的灵敏度分析"""
    vm_values = np.arange(250, 351, 10)
    durations = [compute_shield_with_params(v_m=v) for v in vm_values]
    return vm_values, durations


if __name__ == "__main__":
    print("=" * 60)
    print("灵敏度分析结果")
    print("=" * 60)

    print("\n--- 有效半径 R 的灵敏度 ---")
    R_vals, R_durs = sensitivity_R()
    for r, d in zip(R_vals, R_durs):
        print(f"  R = {r:2d} m -> 遮蔽时长 = {d:.3f} s")

    print("\n--- 有效遮蔽时间 T_eff 的灵敏度 ---")
    T_vals, T_durs = sensitivity_Teff()
    for t, d in zip(T_vals, T_durs):
        print(f"  T_eff = {t:2d} s -> 遮蔽时长 = {d:.3f} s")

    print("\n--- 云团下沉速度 v_c 的灵敏度 ---")
    vc_vals, vc_durs = sensitivity_vc()
    for v, d in zip(vc_vals, vc_durs):
        print(f"  v_c = {v:.1f} m/s -> 遮蔽时长 = {d:.3f} s")

    print("\n--- 导弹速度 v_m 的灵敏度 ---")
    vm_vals, vm_durs = sensitivity_vm()
    for v, d in zip(vm_vals, vm_durs):
        print(f"  v_m = {v:3d} m/s -> 遮蔽时长 = {d:.3f} s")
