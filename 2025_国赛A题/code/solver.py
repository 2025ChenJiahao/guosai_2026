"""
求解器：计算遮蔽时长、PSO优化、结果输出
"""
import numpy as np
from config import *
from physics import (
    missile_pos, bomb_detonate_point, cloud_center,
    shielding_distance, is_shielded, shield_distance_to_target,
    bomb_drop_point, TARGET_POINTS, TARGET_CENTER
)


# ============ 遮蔽时长计算 ============
def compute_shield_duration_single(P_u0, v_u, theta, t_d, delta_t, P_m0,
                                    use_cylinder=True, dt=0.01):
    """计算单枚弹对单枚导弹的有效遮蔽时长"""
    t_e = t_d + delta_t
    P_e = bomb_detonate_point(P_u0, v_u, theta, t_d, delta_t)

    if P_e[2] < 0:
        return 0.0, []

    total = 0.0
    intervals = []
    in_shield = False
    shield_start = 0.0

    t = t_e
    while t <= t_e + T_EFF:
        P_m = missile_pos(P_m0, t)
        P_c = cloud_center(P_e, t, t_e)

        if use_cylinder:
            shielded = is_shielded(P_m, P_c, TARGET_POINTS, R)
        else:
            d = shield_distance_to_target(P_m, P_c, TARGET_CENTER)
            shielded = (d <= R)

        if shielded:
            if not in_shield:
                shield_start = t
                in_shield = True
            total += dt
        else:
            if in_shield:
                intervals.append((shield_start, t))
                in_shield = False

        t += dt

    if in_shield:
        intervals.append((shield_start, t_e + T_EFF))

    return total, intervals


def compute_shield_duration_multi(bomb_params_list, P_m0, use_cylinder=True, dt=0.01):
    """计算多枚弹对单枚导弹的总有效遮蔽时长（考虑并集）"""
    all_intervals = []

    for params in bomb_params_list:
        P_u0, v_u, theta, t_d, delta_t = params
        _, intervals = compute_shield_duration_single(
            P_u0, v_u, theta, t_d, delta_t, P_m0, use_cylinder, dt
        )
        all_intervals.extend(intervals)

    merged = merge_intervals(all_intervals)
    total = sum(end - start for start, end in merged)

    return total, merged


def merge_intervals(intervals):
    """合并时间区间"""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# ============ 问题1求解 ============
def solve_problem1():
    """求解问题1：FY1以120m/s朝向假目标飞行，1.5s后投放，3.6s后起爆"""
    P_u0 = FY1
    v_u = 120.0
    theta = np.arctan2(-FY1[1], -FY1[0])
    t_d = 1.5
    delta_t = 3.6

    P_d = bomb_drop_point(P_u0, v_u, theta, t_d)
    P_e = bomb_detonate_point(P_u0, v_u, theta, t_d, delta_t)
    t_e = t_d + delta_t

    print("=" * 60)
    print("问题1求解结果")
    print("=" * 60)
    print(f"FY1飞行方向角: {np.degrees(theta):.2f} 度")
    print(f"FY1飞行方向向量: ({np.cos(theta):.4f}, {np.sin(theta):.4f}, 0)")
    print(f"投放点: ({P_d[0]:.2f}, {P_d[1]:.2f}, {P_d[2]:.2f})")
    print(f"起爆点: ({P_e[0]:.2f}, {P_e[1]:.2f}, {P_e[2]:.2f})")
    print(f"起爆时刻: {t_e:.2f} s")

    dt = 0.001
    duration_cylinder, intervals_cyl = compute_shield_duration_single(
        P_u0, v_u, theta, t_d, delta_t, M1, use_cylinder=True, dt=dt
    )
    duration_point, intervals_pt = compute_shield_duration_single(
        P_u0, v_u, theta, t_d, delta_t, M1, use_cylinder=False, dt=dt
    )

    print(f"\n使用点目标近似: 有效遮蔽时长 = {duration_point:.3f} s")
    print(f"使用圆柱体目标: 有效遮蔽时长 = {duration_cylinder:.3f} s")
    if intervals_cyl:
        print(f"遮蔽时间区间: {intervals_cyl}")

    return duration_cylinder, intervals_cyl


# ============ 问题2求解 ============
def problem2_objective(params):
    """问题2的目标函数"""
    theta, v_u, t_d, delta_t = params

    if v_u < V_U_MIN or v_u > V_U_MAX:
        return 0.0
    if t_d < 0 or delta_t < 0.1:
        return 0.0
    if t_d + delta_t + T_EFF > M1_ARRIVE:
        return 0.0

    P_e = bomb_detonate_point(FY1, v_u, theta, t_d, delta_t)
    if P_e[2] < 0:
        return 0.0

    duration, _ = compute_shield_duration_single(
        FY1, v_u, theta, t_d, delta_t, M1, use_cylinder=False, dt=0.05
    )
    return duration


def solve_problem2(n_particles=200, max_iter=500):
    """求解问题2：优化FY1投放策略"""
    from pso import PSO

    # 基于问题1的结果生成种子点
    # 问题1: theta=pi, v_u=120, t_d=1.5, delta_t=3.6 -> 1.435s
    seed_theta = np.arctan2(-FY1[1], -FY1[0])  # pi
    seeds = [
        [seed_theta, 120.0, 1.5, 3.6],      # 问题1的参数
        [seed_theta, 110.0, 1.0, 3.0],       # 略微变化
        [seed_theta, 130.0, 2.0, 4.0],
        [seed_theta, 100.0, 0.5, 2.5],
        [seed_theta, 140.0, 1.0, 3.6],
        [seed_theta, 90.0,  2.0, 3.6],
    ]

    lb = [0, V_U_MIN, 0, 0.1]
    ub = [2 * np.pi, V_U_MAX, 30, 10]

    optimizer = PSO(n_particles, 4, lb, ub, problem2_objective, seed_points=seeds)
    best_params, best_val, convergence = optimizer.optimize(max_iter)

    theta, v_u, t_d, delta_t = best_params

    duration, intervals = compute_shield_duration_single(
        FY1, v_u, theta, t_d, delta_t, M1, use_cylinder=True, dt=0.001
    )

    print("=" * 60)
    print("问题2求解结果")
    print("=" * 60)
    print(f"飞行方向角: {np.degrees(theta):.2f} 度")
    print(f"飞行速度: {v_u:.2f} m/s")
    print(f"投放时刻: {t_d:.3f} s")
    print(f"起爆延迟: {delta_t:.3f} s")
    print(f"起爆时刻: {t_d + delta_t:.3f} s")

    P_d = bomb_drop_point(FY1, v_u, theta, t_d)
    P_e = bomb_detonate_point(FY1, v_u, theta, t_d, delta_t)
    print(f"投放点: ({P_d[0]:.2f}, {P_d[1]:.2f}, {P_d[2]:.2f})")
    print(f"起爆点: ({P_e[0]:.2f}, {P_e[1]:.2f}, {P_e[2]:.2f})")
    print(f"有效遮蔽时长: {duration:.3f} s")
    print(f"遮蔽时间区间: {intervals}")

    return best_params, duration, intervals, convergence


# ============ 问题3求解 ============
def problem3_objective(params):
    """问题3的目标函数：FY1投放3枚弹"""
    theta, v_u, t_d1, dt1, t_d2, dt2, t_d3, dt3 = params

    if v_u < V_U_MIN or v_u > V_U_MAX:
        return 0.0
    if any(x < 0 for x in [t_d1, dt1, t_d2, dt2, t_d3, dt3]):
        return 0.0
    if dt1 < 0.1 or dt2 < 0.1 or dt3 < 0.1:
        return 0.0
    if t_d2 < t_d1 + DT_MIN or t_d3 < t_d2 + DT_MIN:
        return 0.0

    for t_d, dt in [(t_d1, dt1), (t_d2, dt2), (t_d3, dt3)]:
        P_e = bomb_detonate_point(FY1, v_u, theta, t_d, dt)
        if P_e[2] < 0:
            return 0.0

    params_list = [
        (FY1, v_u, theta, t_d1, dt1),
        (FY1, v_u, theta, t_d2, dt2),
        (FY1, v_u, theta, t_d3, dt3),
    ]

    duration, _ = compute_shield_duration_multi(params_list, M1, use_cylinder=False, dt=0.05)
    return duration


def solve_problem3(n_particles=200, max_iter=500):
    """求解问题3：FY1投放3枚弹"""
    from pso import PSO

    seed_theta = np.arctan2(-FY1[1], -FY1[0])
    # 种子：3枚弹接力遮蔽
    seeds = [
        [seed_theta, 120.0, 1.0, 3.6, 3.0, 3.6, 5.0, 3.6],
        [seed_theta, 100.0, 0.5, 3.0, 2.5, 3.0, 4.5, 3.0],
        [seed_theta, 140.0, 1.0, 4.0, 3.0, 4.0, 5.0, 4.0],
        [seed_theta, 110.0, 1.5, 3.6, 4.0, 3.6, 6.5, 3.6],
    ]

    lb = [0, V_U_MIN, 0, 0.1, 0, 0.1, 0, 0.1]
    ub = [2 * np.pi, V_U_MAX, 30, 10, 30, 10, 30, 10]

    optimizer = PSO(n_particles, 8, lb, ub, problem3_objective, seed_points=seeds)
    best_params, best_val, convergence = optimizer.optimize(max_iter)

    theta, v_u = best_params[0], best_params[1]
    bombs = [
        (best_params[2], best_params[3]),
        (best_params[4], best_params[5]),
        (best_params[6], best_params[7]),
    ]

    params_list = [(FY1, v_u, theta, t_d, dt) for t_d, dt in bombs]
    duration, intervals = compute_shield_duration_multi(
        params_list, M1, use_cylinder=True, dt=0.001
    )

    print("=" * 60)
    print("问题3求解结果")
    print("=" * 60)
    print(f"飞行方向角: {np.degrees(theta):.2f} 度")
    print(f"飞行速度: {v_u:.2f} m/s")
    for i, (t_d, dt) in enumerate(bombs):
        P_d = bomb_drop_point(FY1, v_u, theta, t_d)
        P_e = bomb_detonate_point(FY1, v_u, theta, t_d, dt)
        print(f"第{i+1}枚弹: 投放={t_d:.3f}s, 起爆延迟={dt:.3f}s, "
              f"投放点=({P_d[0]:.1f},{P_d[1]:.1f},{P_d[2]:.1f}), "
              f"起爆点=({P_e[0]:.1f},{P_e[1]:.1f},{P_e[2]:.1f})")
    print(f"总有效遮蔽时长: {duration:.3f} s")
    print(f"遮蔽时间区间: {intervals}")

    return best_params, duration, intervals, convergence


# ============ 问题4求解 ============
def problem4_objective(params):
    """问题4的目标函数：FY1、FY2、FY3各投放1枚弹"""
    theta1, v1, td1, dt1, theta2, v2, td2, dt2, theta3, v3, td3, dt3 = params

    for v in [v1, v2, v3]:
        if v < V_U_MIN or v > V_U_MAX:
            return 0.0
    for x in [td1, dt1, td2, dt2, td3, dt3]:
        if x < 0:
            return 0.0
    if dt1 < 0.1 or dt2 < 0.1 or dt3 < 0.1:
        return 0.0

    uavs = [(FY1, v1, theta1), (FY2, v2, theta2), (FY3, v3, theta3)]
    tds = [td1, td2, td3]
    dts = [dt1, dt2, dt3]

    for (P_u0, v_u, theta), t_d, dt in zip(uavs, tds, dts):
        P_e = bomb_detonate_point(P_u0, v_u, theta, t_d, dt)
        if P_e[2] < 0:
            return 0.0

    params_list = [
        (FY1, v1, theta1, td1, dt1),
        (FY2, v2, theta2, td2, dt2),
        (FY3, v3, theta3, td3, dt3),
    ]

    duration, _ = compute_shield_duration_multi(params_list, M1, use_cylinder=False, dt=0.05)
    return duration


def solve_problem4(n_particles=200, max_iter=500):
    """求解问题4：3架无人机各投放1枚弹"""
    from pso import PSO

    # FY1的种子（基于问题1）
    theta1_seed = np.arctan2(-FY1[1], -FY1[0])
    # FY2和FY3需要朝M1方向飞，计算方向角
    # M1在(20000,0,2000)，FY2在(12000,1400,1400)
    theta2_seed = np.arctan2(M1[1] - FY2[1], M1[0] - FY2[0])
    # FY3在(6000,-3000,700)
    theta3_seed = np.arctan2(M1[1] - FY3[1], M1[0] - FY3[0])

    seeds = [
        [theta1_seed, 120.0, 1.5, 3.6, theta2_seed, 120.0, 5.0, 3.6, theta3_seed, 120.0, 10.0, 3.6],
        [theta1_seed, 100.0, 1.0, 3.0, theta2_seed, 100.0, 4.0, 3.0, theta3_seed, 100.0, 8.0, 3.0],
        [theta1_seed, 140.0, 2.0, 4.0, theta2_seed, 140.0, 6.0, 4.0, theta3_seed, 140.0, 12.0, 4.0],
    ]

    lb = [0, V_U_MIN, 0, 0.1] * 3
    ub = [2 * np.pi, V_U_MAX, 30, 10] * 3

    optimizer = PSO(n_particles, 12, lb, ub, problem4_objective, seed_points=seeds)
    best_params, best_val, convergence = optimizer.optimize(max_iter)

    uav_params = [
        (FY1, best_params[0], best_params[1], best_params[2], best_params[3]),
        (FY2, best_params[4], best_params[5], best_params[6], best_params[7]),
        (FY3, best_params[8], best_params[9], best_params[10], best_params[11]),
    ]

    params_list = [(P_u0, v_u, theta, t_d, dt) for P_u0, theta, v_u, t_d, dt in uav_params]
    duration, intervals = compute_shield_duration_multi(
        params_list, M1, use_cylinder=True, dt=0.001
    )

    print("=" * 60)
    print("问题4求解结果")
    print("=" * 60)
    for i, (P_u0, theta, v_u, t_d, dt) in enumerate(uav_params):
        P_d = bomb_drop_point(P_u0, v_u, theta, t_d)
        P_e = bomb_detonate_point(P_u0, v_u, theta, t_d, dt)
        print(f"FY{i+1}: 方向={np.degrees(theta):.2f}度, 速度={v_u:.2f}m/s, "
              f"投放={t_d:.3f}s, 起爆延迟={dt:.3f}s")
        print(f"  投放点: ({P_d[0]:.1f}, {P_d[1]:.1f}, {P_d[2]:.1f})")
        print(f"  起爆点: ({P_e[0]:.1f}, {P_e[1]:.1f}, {P_e[2]:.1f})")
    print(f"总有效遮蔽时长: {duration:.3f} s")
    print(f"遮蔽时间区间: {intervals}")

    return best_params, duration, intervals, convergence


# ============ 问题5求解 ============
def solve_problem5_single_missile(P_m0, uav_assignments, n_particles=200, max_iter=500):
    """求解问题5的子问题：优化对单枚导弹的遮蔽策略"""
    from pso import PSO

    n_dims = 0
    lb_list = []
    ub_list = []
    seeds = []

    # 构建种子点
    seed_row = []
    for P_u0, max_bombs in uav_assignments:
        lb_list.extend([0, V_U_MIN])
        ub_list.extend([2 * np.pi, V_U_MAX])
        n_dims += 2

        # 计算朝向导弹的方向角
        theta_seed = np.arctan2(P_m0[1] - P_u0[1], P_m0[0] - P_u0[0])
        seed_row.extend([theta_seed, 120.0])

        for k in range(max_bombs):
            lb_list.extend([0, 0.1])
            ub_list.extend([30, 10])
            n_dims += 2
            seed_row.extend([1.0 + k * 2.0, 3.6])  # 间隔投放

    seeds.append(seed_row)
    # 添加变体
    seed2 = seed_row.copy()
    for i in range(len(seed2)):
        if i % 4 >= 2:  # 投放时刻和起爆延迟
            seed2[i] = seed2[i] * 0.8
    seeds.append(seed2)

    def objective(params):
        bomb_params = []
        idx = 0
        for P_u0, max_bombs in uav_assignments:
            theta = params[idx]
            v_u = params[idx + 1]
            idx += 2
            if v_u < V_U_MIN or v_u > V_U_MAX:
                return 0.0
            prev_td = -DT_MIN
            for _ in range(max_bombs):
                t_d = params[idx]
                dt = params[idx + 1]
                idx += 2
                if t_d < prev_td + DT_MIN or dt < 0.1:
                    return 0.0
                prev_td = t_d
                P_e = bomb_detonate_point(P_u0, v_u, theta, t_d, dt)
                if P_e[2] < 0:
                    return 0.0
                bomb_params.append((P_u0, v_u, theta, t_d, dt))

        duration, _ = compute_shield_duration_multi(bomb_params, P_m0, use_cylinder=False, dt=0.05)
        return duration

    optimizer = PSO(n_particles, n_dims, lb_list, ub_list, objective, seed_points=seeds)
    best_params, best_val, convergence = optimizer.optimize(max_iter)

    return best_params, best_val


def solve_problem5(n_particles=200, max_iter=500):
    """求解问题5：5架无人机干扰3枚导弹"""
    assignments = {
        1: [(FY1, 3), (FY2, 3)],
        2: [(FY4, 3)],
        3: [(FY3, 3), (FY5, 3)],
    }

    total_duration = 0
    all_results = {}

    for missile_id, uav_list in assignments.items():
        print(f"\n优化对M{missile_id}的遮蔽策略...")
        P_m0 = MISSILES[missile_id]
        best_params, best_val = solve_problem5_single_missile(
            P_m0, uav_list, n_particles, max_iter
        )
        all_results[missile_id] = (uav_list, best_params, best_val)
        total_duration += best_val
        print(f"  对M{missile_id}的遮蔽时长: {best_val:.3f} s")

    print("=" * 60)
    print("问题5求解结果")
    print("=" * 60)
    print(f"总遮蔽时长: {total_duration:.3f} s")

    return all_results, total_duration


if __name__ == "__main__":
    print("开始求解...")
    print()

    solve_problem1()
    print()
