"""
配置文件：定义所有常量和初始参数
"""
import numpy as np

# ============ 物理常量 ============
V_M = 300.0       # 导弹速度 (m/s)
V_C = 3.0         # 云团下沉速度 (m/s)
G = 9.8           # 重力加速度 (m/s^2)
R = 10.0          # 有效遮蔽半径 (m)
T_EFF = 20.0      # 有效遮蔽时间 (s)
DT_MIN = 1.0      # 同一无人机两枚弹最小投放间隔 (s)

# ============ 无人机速度范围 ============
V_U_MIN = 70.0    # 无人机最小速度 (m/s)
V_U_MAX = 140.0   # 无人机最大速度 (m/s)

# ============ 初始位置 ============
# 导弹初始位置
M1 = np.array([20000.0, 0.0, 2000.0])
M2 = np.array([19000.0, 600.0, 2100.0])
M3 = np.array([18000.0, -600.0, 1900.0])

# 无人机初始位置
FY1 = np.array([17800.0, 0.0, 1800.0])
FY2 = np.array([12000.0, 1400.0, 1400.0])
FY3 = np.array([6000.0, -3000.0, 700.0])
FY4 = np.array([11000.0, 2000.0, 1800.0])
FY5 = np.array([13000.0, -2000.0, 1300.0])

# 真目标参数
TARGET_CENTER = np.array([0.0, 200.0, 5.0])   # 真目标中心点
TARGET_BOTTOM = np.array([0.0, 200.0, 0.0])   # 真目标下底面圆心
TARGET_R = 7.0      # 真目标圆柱半径 (m)
TARGET_H = 10.0     # 真目标圆柱高度 (m)

# 假目标（原点）
DECOY = np.array([0.0, 0.0, 0.0])

# ============ 导弹方向向量（预计算） ============
def calc_missile_dir(P_m0):
    """计算导弹方向向量（指向原点）"""
    d = -P_m0 / np.linalg.norm(P_m0)
    return d

M1_DIR = calc_missile_dir(M1)
M2_DIR = calc_missile_dir(M2)
M3_DIR = calc_missile_dir(M3)

# 导弹速度向量
M1_VEL = V_M * M1_DIR
M2_VEL = V_M * M2_DIR
M3_VEL = V_M * M3_DIR

# 导弹到达原点的时间
M1_ARRIVE = np.linalg.norm(M1) / V_M
M2_ARRIVE = np.linalg.norm(M2) / V_M
M3_ARRIVE = np.linalg.norm(M3) / V_M

# 无人机和导弹的字典，方便索引
MISSILES = {1: M1, 2: M2, 3: M3}
MISSILE_DIRS = {1: M1_DIR, 2: M2_DIR, 3: M3_DIR}
MISSILE_VELS = {1: M1_VEL, 2: M2_VEL, 3: M3_VEL}
MISSILE_ARRIVE = {1: M1_ARRIVE, 2: M2_ARRIVE, 3: M3_ARRIVE}

UAVS = {1: FY1, 2: FY2, 3: FY3, 4: FY4, 5: FY5}
