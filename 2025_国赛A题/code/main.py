"""
主程序：运行所有问题的求解
"""
import os
import sys
import numpy as np

# 确保可以导入同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from physics import *
from solver import (
    solve_problem1, solve_problem2, solve_problem3,
    solve_problem4, solve_problem5,
    compute_shield_duration_single
)
from output import save_result1, save_result2, save_result3

# 创建输出目录
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


def main():
    print("=" * 70)
    print("2025年全国大学生数学建模竞赛A题：烟幕干扰弹投放策略")
    print("=" * 70)
    print()

    # ==================== 问题1 ====================
    print("=" * 70)
    print("问题1：FY1投放1枚弹干扰M1（给定参数）")
    print("=" * 70)
    duration1, intervals1 = solve_problem1()
    print()

    # 可视化
    from visualization import plot_3d_trajectory_single, plot_distance_time

    theta_q1 = np.arctan2(-FY1[1], -FY1[0])
    plot_3d_trajectory_single(
        FY1, 120.0, theta_q1, 1.5, 3.6, M1,
        title="Problem 1: 3D Trajectory",
        save_path="figures/q1_3d_trajectory.png"
    )
    plot_distance_time(
        FY1, 120.0, theta_q1, 1.5, 3.6, M1,
        save_path="figures/q1_distance_time.png"
    )

    # ==================== 问题2 ====================
    print()
    print("=" * 70)
    print("问题2：优化FY1投放1枚弹的策略")
    print("=" * 70)
    params2, duration2, intervals2, conv2 = solve_problem2(
        n_particles=200, max_iter=500
    )
    print()

    # 可视化
    from visualization import plot_convergence
    plot_convergence(conv2, title="Problem 2: PSO Convergence",
                     save_path="figures/q2_convergence.png")

    theta2, v_u2, t_d2, dt2 = params2
    plot_3d_trajectory_single(
        FY1, v_u2, theta2, t_d2, dt2, M1,
        title="Problem 2: Optimal Trajectory",
        save_path="figures/q2_3d_trajectory.png"
    )

    # ==================== 问题3 ====================
    print()
    print("=" * 70)
    print("问题3：FY1投放3枚弹干扰M1")
    print("=" * 70)
    params3, duration3, intervals3, conv3 = solve_problem3(
        n_particles=200, max_iter=500
    )
    print()

    # 保存结果
    theta3, v_u3 = params3[0], params3[1]
    bombs3 = [
        (params3[2], params3[3]),
        (params3[4], params3[5]),
        (params3[6], params3[7]),
    ]
    save_result1(theta3, v_u3, bombs3, duration3)

    # 可视化
    from visualization import plot_timeline
    bomb_intervals3 = []
    for t_d, dt in bombs3:
        _, intervals = compute_shield_duration_single(
            FY1, v_u3, theta3, t_d, dt, M1, use_cylinder=False, dt=0.01
        )
        bomb_intervals3.append(intervals)

    plot_timeline(bomb_intervals3,
                  labels=[f'Bomb {i+1}' for i in range(3)],
                  title="Problem 3: Shielding Timeline",
                  save_path="figures/q3_timeline.png")
    plot_convergence(conv3, title="Problem 3: PSO Convergence",
                     save_path="figures/q3_convergence.png")

    # ==================== 问题4 ====================
    print()
    print("=" * 70)
    print("问题4：FY1、FY2、FY3各投放1枚弹干扰M1")
    print("=" * 70)
    params4, duration4, intervals4, conv4 = solve_problem4(
        n_particles=200, max_iter=500
    )
    print()

    # 保存结果
    uav_params4 = [
        (FY1, params4[0], params4[1], params4[2], params4[3]),
        (FY2, params4[4], params4[5], params4[6], params4[7]),
        (FY3, params4[8], params4[9], params4[10], params4[11]),
    ]
    save_result2(uav_params4, duration4)

    # 可视化
    from visualization import plot_3d_multi_uav
    multi_uav_params = [
        (FY1, params4[1], params4[0], params4[2], params4[3]),
        (FY2, params4[5], params4[4], params4[6], params4[7]),
        (FY3, params4[9], params4[8], params4[10], params4[11]),
    ]
    plot_3d_multi_uav(multi_uav_params, M1,
                       title="Problem 4: Multi-UAV Trajectory",
                       save_path="figures/q4_3d_trajectory.png")

    # ==================== 问题5 ====================
    print()
    print("=" * 70)
    print("问题5：5架无人机干扰M1、M2、M3")
    print("=" * 70)
    all_results5, total5 = solve_problem5(
        n_particles=200, max_iter=500
    )
    print()

    # 保存结果
    save_result3(all_results5)

    # ==================== 汇总 ====================
    print()
    print("=" * 70)
    print("结果汇总")
    print("=" * 70)
    print(f"问题1 有效遮蔽时长: {duration1:.3f} s")
    print(f"问题2 有效遮蔽时长: {duration2:.3f} s")
    print(f"问题3 有效遮蔽时长: {duration3:.3f} s")
    print(f"问题4 有效遮蔽时长: {duration4:.3f} s")
    print(f"问题5 总遮蔽时长:   {total5:.3f} s")
    print()
    print("结果已保存到 results/ 目录")
    print("图表已保存到 figures/ 目录")


if __name__ == "__main__":
    main()
