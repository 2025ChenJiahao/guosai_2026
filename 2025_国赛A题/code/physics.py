"""
物理模型：导弹运动、无人机运动、烟幕弹运动、遮蔽条件计算
"""
import numpy as np
from config import *


# ============ 导弹运动 ============
def missile_pos(P_m0, t):
    """计算导弹在时刻t的位置

    Args:
        P_m0: 导弹初始位置 (3,)
        t: 时刻 (s)

    Returns:
        导弹位置 (3,)
    """
    d_m = -P_m0 / np.linalg.norm(P_m0)
    return P_m0 + V_M * t * d_m


# ============ 无人机运动 ============
def uav_pos(P_u0, v_u, theta, t):
    """计算无人机在时刻t的位置

    Args:
        P_u0: 无人机初始位置 (3,)
        v_u: 飞行速度 (m/s)
        theta: 飞行方向角 (rad)，水平面内
        t: 时刻 (s)

    Returns:
        无人机位置 (3,)
    """
    d_u = np.array([np.cos(theta), np.sin(theta), 0.0])
    return P_u0 + v_u * t * d_u


# ============ 烟幕弹运动 ============
def bomb_drop_point(P_u0, v_u, theta, t_d):
    """计算投放点位置

    Args:
        P_u0: 无人机初始位置 (3,)
        v_u: 飞行速度 (m/s)
        theta: 飞行方向角 (rad)
        t_d: 投放时刻 (s)

    Returns:
        投放点位置 (3,)
    """
    d_u = np.array([np.cos(theta), np.sin(theta), 0.0])
    return P_u0 + v_u * t_d * d_u


def bomb_detonate_point(P_u0, v_u, theta, t_d, delta_t):
    """计算起爆点位置

    Args:
        P_u0: 无人机初始位置 (3,)
        v_u: 飞行速度 (m/s)
        theta: 飞行方向角 (rad)
        t_d: 投放时刻 (s)
        delta_t: 投放到起爆的时间间隔 (s)

    Returns:
        起爆点位置 (3,)
    """
    P_d = bomb_drop_point(P_u0, v_u, theta, t_d)
    d_u = np.array([np.cos(theta), np.sin(theta), 0.0])
    v0 = v_u * d_u  # 初速度（等于无人机速度）
    g_vec = np.array([0.0, 0.0, -G])
    return P_d + v0 * delta_t + 0.5 * g_vec * delta_t**2


def bomb_pos(P_u0, v_u, theta, t_d, t):
    """计算烟幕弹在时刻t的位置（投放后到起爆前）

    Args:
        P_u0: 无人机初始位置 (3,)
        v_u: 飞行速度 (m/s)
        theta: 飞行方向角 (rad)
        t_d: 投放时刻 (s)
        t: 当前时刻 (s), t >= t_d

    Returns:
        烟幕弹位置 (3,)
    """
    P_d = bomb_drop_point(P_u0, v_u, theta, t_d)
    d_u = np.array([np.cos(theta), np.sin(theta), 0.0])
    v0 = v_u * d_u
    g_vec = np.array([0.0, 0.0, -G])
    dt = t - t_d
    return P_d + v0 * dt + 0.5 * g_vec * dt**2


# ============ 云团运动 ============
def cloud_center(P_e, t, t_e):
    """计算云团中心在时刻t的位置

    Args:
        P_e: 起爆点位置 (3,)
        t: 当前时刻 (s)
        t_e: 起爆时刻 (s)

    Returns:
        云团中心位置 (3,)
    """
    return P_e + np.array([0.0, 0.0, -V_C]) * (t - t_e)


# ============ 遮蔽条件计算 ============
def shielding_distance(P_m, P_c, T_target):
    """计算云团中心到导弹-目标视线段的距离

    Args:
        P_m: 导弹位置 (3,)
        P_c: 云团中心位置 (3,)
        T_target: 目标位置 (3,)

    Returns:
        距离 (m)
    """
    v = T_target - P_m
    w = P_c - P_m
    vv = np.dot(v, v)
    if vv < 1e-10:
        return np.linalg.norm(P_c - P_m)
    s = np.dot(w, v) / vv
    s = max(0.0, min(1.0, s))
    Q = P_m + s * v
    return np.linalg.norm(P_c - Q)


def sample_cylinder_surface(C_center, r, h, n_side=16, n_height=3, n_top=8):
    """在圆柱体表面均匀采样点

    Args:
        C_center: 下底面圆心 (3,)
        r: 半径 (m)
        h: 高度 (m)
        n_side: 侧面角度采样数
        n_height: 侧面高度采样层数
        n_top: 底面角度采样数

    Returns:
        采样点数组 (N, 3)
    """
    points = []
    # 侧面
    for i in range(n_side):
        angle = 2 * np.pi * i / n_side
        for j in range(n_height):
            z = h * j / (n_height - 1)
            points.append([
                C_center[0] + r * np.cos(angle),
                C_center[1] + r * np.sin(angle),
                C_center[2] + z
            ])
    # 上底面
    for i in range(n_top):
        angle = 2 * np.pi * i / n_top
        for rho in [0, r / 2, r]:
            points.append([
                C_center[0] + rho * np.cos(angle),
                C_center[1] + rho * np.sin(angle),
                C_center[2] + h
            ])
    # 下底面
    for i in range(n_top):
        angle = 2 * np.pi * i / n_top
        for rho in [0, r / 2, r]:
            points.append([
                C_center[0] + rho * np.cos(angle),
                C_center[1] + rho * np.sin(angle),
                C_center[2]
            ])
    return np.array(points)


# 预计算真目标采样点
TARGET_POINTS = sample_cylinder_surface(TARGET_BOTTOM, TARGET_R, TARGET_H)


def is_shielded(P_m, P_c, target_points=None, R_eff=None):
    """判断当前时刻是否有效遮蔽

    Args:
        P_m: 导弹位置 (3,)
        P_c: 云团中心位置 (3,)
        target_points: 目标采样点列表，默认使用圆柱体采样点
        R_eff: 有效遮蔽半径，默认使用 R

    Returns:
        bool: 是否有效遮蔽
    """
    if target_points is None:
        target_points = TARGET_POINTS
    if R_eff is None:
        R_eff = R

    for T in target_points:
        d = shielding_distance(P_m, P_c, T)
        if d > R_eff:
            return False
    return True


def shield_distance_to_target(P_m, P_c, target=None):
    """计算到目标中心的遮蔽距离（简化版，用中心点近似）

    Args:
        P_m: 导弹位置 (3,)
        P_c: 云团中心位置 (3,)
        target: 目标位置，默认使用真目标中心

    Returns:
        距离 (m)
    """
    if target is None:
        target = TARGET_CENTER
    return shielding_distance(P_m, P_c, target)
