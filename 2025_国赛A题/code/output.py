"""
结果输出：将求解结果保存到Excel文件
"""
import numpy as np
import pandas as pd
from config import *
from physics import bomb_drop_point, bomb_detonate_point


def save_result1(theta, v_u, bombs, duration, save_path="results/result1.xlsx"):
    """保存问题3结果（FY1投放3枚弹）

    Args:
        theta: 飞行方向角 (rad)
        v_u: 飞行速度 (m/s)
        bombs: [(t_d1, dt1), (t_d2, dt2), (t_d3, dt3)]
        duration: 总遮蔽时长 (s)
        save_path: 保存路径
    """
    rows = []
    for i, (t_d, dt) in enumerate(bombs):
        P_d = bomb_drop_point(FY1, v_u, theta, t_d)
        P_e = bomb_detonate_point(FY1, v_u, theta, t_d, dt)
        rows.append({
            '无人机编号': 'FY1',
            '无人机运动方向': round(np.degrees(theta), 2),
            '无人机运动速度(m/s)': round(v_u, 2),
            '烟幕干扰弹编号': i + 1,
            '烟幕干扰弹投放点的x坐标(m)': round(P_d[0], 2),
            '烟幕干扰弹投放点的y坐标(m)': round(P_d[1], 2),
            '烟幕干扰弹投放点的z坐标(m)': round(P_d[2], 2),
            '烟幕干扰弹起爆点的x坐标(m)': round(P_e[0], 2),
            '烟幕干扰弹起爆点的y坐标(m)': round(P_e[1], 2),
            '烟幕干扰弹起爆点的z坐标(m)': round(P_e[2], 2),
            '有效干扰时长(s)': round(duration, 2),
        })

    df = pd.DataFrame(rows)
    df.to_excel(save_path, index=False)
    print(f"结果已保存到 {save_path}")


def save_result2(uav_params, duration, save_path="results/result2.xlsx"):
    """保存问题4结果（3架无人机各投放1枚弹）

    Args:
        uav_params: [(FY, theta, v_u, t_d, dt), ...]
        duration: 总遮蔽时长 (s)
        save_path: 保存路径
    """
    rows = []
    for i, (P_u0, theta, v_u, t_d, dt) in enumerate(uav_params):
        P_d = bomb_drop_point(P_u0, v_u, theta, t_d)
        P_e = bomb_detonate_point(P_u0, v_u, theta, t_d, dt)
        rows.append({
            '无人机编号': f'FY{i+1}',
            '无人机运动方向': round(np.degrees(theta), 2),
            '无人机运动速度(m/s)': round(v_u, 2),
            '烟幕干扰弹编号': 1,
            '烟幕干扰弹投放点的x坐标(m)': round(P_d[0], 2),
            '烟幕干扰弹投放点的y坐标(m)': round(P_d[1], 2),
            '烟幕干扰弹投放点的z坐标(m)': round(P_d[2], 2),
            '烟幕干扰弹起爆点的x坐标(m)': round(P_e[0], 2),
            '烟幕干扰弹起爆点的y坐标(m)': round(P_e[1], 2),
            '烟幕干扰弹起爆点的z坐标(m)': round(P_e[2], 2),
            '有效干扰时长(s)': round(duration, 2),
        })

    df = pd.DataFrame(rows)
    df.to_excel(save_path, index=False)
    print(f"结果已保存到 {save_path}")


def save_result3(all_results, save_path="results/result3.xlsx"):
    """保存问题5结果（5架无人机干扰3枚导弹）

    Args:
        all_results: {missile_id: (uav_list, best_params, best_val), ...}
        save_path: 保存路径
    """
    rows = []

    for missile_id, (uav_list, best_params, best_val) in all_results.items():
        idx = 0
        for j, (P_u0, max_bombs) in enumerate(uav_list):
            theta = best_params[idx]
            v_u = best_params[idx + 1]
            idx += 2

            for k in range(max_bombs):
                t_d = best_params[idx]
                dt = best_params[idx + 1]
                idx += 2

                if t_d < 0 or dt < 0:
                    continue

                P_d = bomb_drop_point(P_u0, v_u, theta, t_d)
                P_e = bomb_detonate_point(P_u0, v_u, theta, t_d, dt)

                # 确定无人机编号
                if np.allclose(P_u0, FY1):
                    uav_name = 'FY1'
                elif np.allclose(P_u0, FY2):
                    uav_name = 'FY2'
                elif np.allclose(P_u0, FY3):
                    uav_name = 'FY3'
                elif np.allclose(P_u0, FY4):
                    uav_name = 'FY4'
                elif np.allclose(P_u0, FY5):
                    uav_name = 'FY5'
                else:
                    uav_name = f'UAV{j+1}'

                rows.append({
                    '干扰导弹': f'M{missile_id}',
                    '无人机编号': uav_name,
                    '无人机运动方向': round(np.degrees(theta), 2),
                    '无人机运动速度(m/s)': round(v_u, 2),
                    '烟幕干扰弹编号': k + 1,
                    '烟幕干扰弹投放点的x坐标(m)': round(P_d[0], 2),
                    '烟幕干扰弹投放点的y坐标(m)': round(P_d[1], 2),
                    '烟幕干扰弹投放点的z坐标(m)': round(P_d[2], 2),
                    '烟幕干扰弹起爆点的x坐标(m)': round(P_e[0], 2),
                    '烟幕干扰弹起爆点的y坐标(m)': round(P_e[1], 2),
                    '烟幕干扰弹起爆点的z坐标(m)': round(P_e[2], 2),
                    '有效干扰时长(s)': round(best_val, 2),
                })

    df = pd.DataFrame(rows)
    df.to_excel(save_path, index=False)
    print(f"结果已保存到 {save_path}")
