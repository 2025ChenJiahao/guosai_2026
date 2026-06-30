"""
可视化：绘制各种图表用于论文
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from config import *
from physics import (
    missile_pos, uav_pos, bomb_detonate_point, cloud_center,
    bomb_drop_point, shielding_distance, shield_distance_to_target
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def plot_3d_trajectory_single(P_u0, v_u, theta, t_d, delta_t, P_m0,
                               title="3D Trajectory", save_path=None):
    """绘制单枚弹的3D轨迹图

    Args:
        P_u0: 无人机初始位置
        v_u: 飞行速度
        theta: 飞行方向角
        t_d: 投放时刻
        delta_t: 起爆延迟
        P_m0: 导弹初始位置
        title: 图标题
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 时间范围
    t_arrive = np.linalg.norm(P_m0) / V_M
    t_e = t_d + delta_t

    # 导弹轨迹
    t_m = np.linspace(0, t_arrive, 500)
    traj_m = np.array([missile_pos(P_m0, t) for t in t_m])
    ax.plot(traj_m[:, 0], traj_m[:, 1], traj_m[:, 2],
            'r-', linewidth=2, label='Missile')
    ax.scatter(*P_m0, c='red', s=50, marker='^')
    ax.text(P_m0[0], P_m0[1], P_m0[2], '  M1', fontsize=10)

    # 无人机轨迹（到投放点）
    t_u = np.linspace(0, t_d, 100)
    traj_u = np.array([uav_pos(P_u0, v_u, theta, t) for t in t_u])
    ax.plot(traj_u[:, 0], traj_u[:, 1], traj_u[:, 2],
            'b-', linewidth=2, label='UAV')
    ax.scatter(*P_u0, c='blue', s=50, marker='o')
    ax.text(P_u0[0], P_u0[1], P_u0[2], '  FY1', fontsize=10)

    # 烟幕弹轨迹（投放到起爆）
    t_b = np.linspace(t_d, t_e, 100)
    from physics import bomb_pos
    traj_b = np.array([bomb_pos(P_u0, v_u, theta, t_d, t) for t in t_b])
    ax.plot(traj_b[:, 0], traj_b[:, 1], traj_b[:, 2],
            'g--', linewidth=1.5, label='Bomb')

    # 云团轨迹
    P_e = bomb_detonate_point(P_u0, v_u, theta, t_d, delta_t)
    t_c = np.linspace(t_e, min(t_e + T_EFF, t_arrive), 200)
    traj_c = np.array([cloud_center(P_e, t, t_e) for t in t_c])
    ax.plot(traj_c[:, 0], traj_c[:, 1], traj_c[:, 2],
            'g-', linewidth=2, label='Cloud')
    ax.scatter(*P_e, c='green', s=80, marker='*')

    # 标记投放点
    P_d = bomb_drop_point(P_u0, v_u, theta, t_d)
    ax.scatter(*P_d, c='cyan', s=60, marker='D', label='Drop point')

    # 目标
    ax.scatter(*TARGET_CENTER, c='gold', s=150, marker='*', label='Real target')
    ax.text(TARGET_CENTER[0], TARGET_CENTER[1], TARGET_CENTER[2], '  Target', fontsize=10)
    ax.scatter(*DECOY, c='black', s=100, marker='x', label='Decoy')

    # 真目标圆柱体（简化绘制）
    theta_cyl = np.linspace(0, 2 * np.pi, 30)
    z_cyl = np.array([0, TARGET_H])
    for z in z_cyl:
        x_cyl = TARGET_BOTTOM[0] + TARGET_R * np.cos(theta_cyl)
        y_cyl = TARGET_BOTTOM[1] + TARGET_R * np.sin(theta_cyl)
        z_arr = np.full_like(theta_cyl, TARGET_BOTTOM[2] + z)
        ax.plot(x_cyl, y_cyl, z_arr, 'gold', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_distance_time(P_u0, v_u, theta, t_d, delta_t, P_m0,
                        save_path=None):
    """绘制遮蔽距离-时间曲线

    Args:
        P_u0: 无人机初始位置
        v_u: 飞行速度
        theta: 飞行方向角
        t_d: 投放时刻
        delta_t: 起爆延迟
        P_m0: 导弹初始位置
        save_path: 保存路径
    """
    t_e = t_d + delta_t
    P_e = bomb_detonate_point(P_u0, v_u, theta, t_d, delta_t)

    dt = 0.01
    times = np.arange(t_e, t_e + T_EFF, dt)
    distances = []
    shielded = []

    for t in times:
        P_m = missile_pos(P_m0, t)
        P_c = cloud_center(P_e, t, t_e)
        d = shield_distance_to_target(P_m, P_c)
        distances.append(d)
        shielded.append(d <= R)

    distances = np.array(distances)
    shielded = np.array(shielded)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 上图：距离曲线
    ax1.plot(times, distances, 'b-', linewidth=1.5, label='Distance')
    ax1.axhline(y=R, color='r', linestyle='--', linewidth=1, label=f'R = {R} m')
    ax1.fill_between(times, 0, R, where=distances <= R,
                      alpha=0.3, color='green', label='Shielded')
    ax1.set_ylabel('Distance (m)', fontsize=12)
    ax1.set_title('Shielding Distance vs Time', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # 下图：遮蔽状态
    ax2.fill_between(times, 0, shielded.astype(int),
                      alpha=0.5, color='green', step='post')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Shielded', fontsize=12)
    ax2.set_title('Shielding Status', fontsize=14)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No', 'Yes'])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_convergence(convergence, title="PSO Convergence", save_path=None):
    """绘制PSO收敛曲线

    Args:
        convergence: 每代最优适应度列表
        title: 图标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(convergence) + 1), convergence, 'b-', linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Fitness (Shield Duration)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_timeline(intervals_list, labels=None, title="Shielding Timeline",
                  save_path=None):
    """绘制遮蔽时间甘特图

    Args:
        intervals_list: [[(start, end), ...], ...] 每枚弹的遮蔽区间
        labels: 各弹标签
        title: 图标题
        save_path: 保存路径
    """
    n = len(intervals_list)
    if labels is None:
        labels = [f'Bomb {i+1}' for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, 4))

    colors = plt.cm.Set2(np.linspace(0, 1, n))

    for i, intervals in enumerate(intervals_list):
        for start, end in intervals:
            ax.barh(i, end - start, left=start, height=0.6,
                    color=colors[i], edgecolor='black', linewidth=0.5)

    # 绘制并集
    all_intervals = []
    for intervals in intervals_list:
        all_intervals.extend(intervals)
    if all_intervals:
        all_intervals.sort(key=lambda x: x[0])
        merged = [all_intervals[0]]
        for start, end in all_intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # 在底部绘制并集
        for start, end in merged:
            ax.barh(n, end - start, left=start, height=0.6,
                    color='lightcoral', edgecolor='red', linewidth=1,
                    alpha=0.7)

    ax.set_yticks(range(n + 1))
    ax.set_yticklabels(labels + ['Union'])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sensitivity(param_name, param_values, shield_durations,
                     baseline_val=None, baseline_dur=None,
                     save_path=None):
    """绘制灵敏度分析图

    Args:
        param_name: 参数名称
        param_values: 参数值列表
        shield_durations: 对应的遮蔽时长列表
        baseline_val: 基准参数值
        baseline_dur: 基准遮蔽时长
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(param_values, shield_durations, 'bo-', linewidth=2, markersize=8)

    if baseline_val is not None and baseline_dur is not None:
        ax.axvline(x=baseline_val, color='r', linestyle='--', linewidth=1,
                    label=f'Baseline = {baseline_val}')
        ax.axhline(y=baseline_dur, color='gray', linestyle=':', linewidth=1)

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Shield Duration (s)', fontsize=12)
    ax.set_title(f'Sensitivity Analysis: {param_name}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_3d_multi_uav(uav_params_list, P_m0, title="Multi-UAV Trajectory",
                       save_path=None):
    """绘制多架无人机的3D轨迹图

    Args:
        uav_params_list: [(P_u0, v_u, theta, t_d, delta_t), ...]
        P_m0: 导弹初始位置
        title: 图标题
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['blue', 'green', 'orange', 'purple', 'brown']

    # 导弹轨迹
    t_arrive = np.linalg.norm(P_m0) / V_M
    t_m = np.linspace(0, t_arrive, 500)
    traj_m = np.array([missile_pos(P_m0, t) for t in t_m])
    ax.plot(traj_m[:, 0], traj_m[:, 1], traj_m[:, 2],
            'r-', linewidth=2, label='Missile')

    # 各无人机轨迹
    for i, (P_u0, v_u, theta, t_d, delta_t) in enumerate(uav_params_list):
        c = colors[i % len(colors)]

        # 无人机轨迹
        t_u = np.linspace(0, t_d, 100)
        traj_u = np.array([uav_pos(P_u0, v_u, theta, t) for t in t_u])
        ax.plot(traj_u[:, 0], traj_u[:, 1], traj_u[:, 2],
                color=c, linewidth=1.5, linestyle='-', label=f'FY{i+1}')

        # 云团轨迹
        P_e = bomb_detonate_point(P_u0, v_u, theta, t_d, delta_t)
        t_e = t_d + delta_t
        t_c = np.linspace(t_e, min(t_e + T_EFF, t_arrive), 200)
        traj_c = np.array([cloud_center(P_e, t, t_e) for t in t_c])
        ax.plot(traj_c[:, 0], traj_c[:, 1], traj_c[:, 2],
                color=c, linewidth=2, linestyle='--')

    # 目标
    ax.scatter(*TARGET_CENTER, c='gold', s=150, marker='*', label='Target')
    ax.scatter(*DECOY, c='black', s=100, marker='x', label='Decoy')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 测试问题1的可视化
    theta = np.arctan2(-FY1[1], -FY1[0])  # 朝向假目标
    v_u = 120.0
    t_d = 1.5
    delta_t = 3.6

    plot_3d_trajectory_single(
        FY1, v_u, theta, t_d, delta_t, M1,
        title="Problem 1: FY1 Trajectory",
        save_path="figures/q1_3d_trajectory.png"
    )

    plot_distance_time(
        FY1, v_u, theta, t_d, delta_t, M1,
        save_path="figures/q1_distance_time.png"
    )
