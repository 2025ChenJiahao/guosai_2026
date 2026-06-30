"""
粒子群优化算法（PSO）实现
"""
import numpy as np


class PSO:
    """粒子群优化器

    用于求解连续优化问题，支持边界约束和引导初始化。

    Args:
        n_particles: 粒子数量
        n_dims: 决策变量维度
        lb: 各维度下界 (n_dims,)
        ub: 各维度上界 (n_dims,)
        fitness_func: 适应度函数，输入参数向量，输出适应度值（越大越好）
        w_max: 最大惯性权重
        w_min: 最小惯性权重
        c1: 个体学习因子
        c2: 社会学习因子
        seed_points: 引导种子点列表，PSO会将部分粒子初始化在这些点附近
    """

    def __init__(self, n_particles, n_dims, lb, ub, fitness_func,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, seed_points=None):
        self.n = n_particles
        self.d = n_dims
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.fitness = fitness_func

        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

        # 初始化粒子位置
        self.positions = np.random.uniform(self.lb, self.ub, (n_particles, n_dims))

        # 如果提供了种子点，将部分粒子初始化在种子点附近
        if seed_points is not None and len(seed_points) > 0:
            n_seeds = min(len(seed_points), n_particles // 3)
            for i in range(n_seeds):
                seed = np.array(seed_points[i], dtype=float)
                # 在种子点附近添加小扰动
                noise_scale = (self.ub - self.lb) * 0.05  # 5%的范围
                noise = np.random.uniform(-noise_scale, noise_scale)
                self.positions[i] = np.clip(seed + noise, self.lb, self.ub)

        # 初始化速度
        v_range = (self.ub - self.lb) * 0.1
        self.velocities = np.random.uniform(-v_range, v_range, (n_particles, n_dims))

        # 个体最优
        self.pbest_pos = self.positions.copy()
        self.pbest_val = np.full(n_particles, -np.inf)

        # 全局最优
        self.gbest_pos = None
        self.gbest_val = -np.inf

    def optimize(self, max_iter, verbose=True):
        """运行PSO优化

        Args:
            max_iter: 最大迭代次数
            verbose: 是否打印进度

        Returns:
            best_params: 最优参数向量
            best_fitness: 最优适应度值
            convergence: 每代最优适应度列表
        """
        convergence = []

        for iteration in range(max_iter):
            # 线性递减惯性权重
            w = self.w_max - (self.w_max - self.w_min) * iteration / max_iter

            # 计算适应度并更新个体最优和全局最优
            for i in range(self.n):
                fit = self.fitness(self.positions[i])

                # 更新个体最优
                if fit > self.pbest_val[i]:
                    self.pbest_val[i] = fit
                    self.pbest_pos[i] = self.positions[i].copy()

                # 更新全局最优
                if fit > self.gbest_val:
                    self.gbest_val = fit
                    self.gbest_pos = self.positions[i].copy()

            convergence.append(self.gbest_val)

            # 生成随机数
            r1 = np.random.random((self.n, self.d))
            r2 = np.random.random((self.n, self.d))

            # 更新速度
            self.velocities = (
                w * self.velocities
                + self.c1 * r1 * (self.pbest_pos - self.positions)
                + self.c2 * r2 * (self.gbest_pos - self.positions)
            )

            # 速度限制
            v_max = (self.ub - self.lb) * 0.2
            self.velocities = np.clip(self.velocities, -v_max, v_max)

            # 更新位置
            self.positions += self.velocities

            # 边界约束
            self.positions = np.clip(self.positions, self.lb, self.ub)

            # 打印进度
            if verbose and (iteration + 1) % 50 == 0:
                print(f"  迭代 {iteration + 1}/{max_iter}: 最优适应度 = {self.gbest_val:.6f}")

        if verbose:
            print(f"  优化完成: 最优适应度 = {self.gbest_val:.6f}")

        return self.gbest_pos.copy(), self.gbest_val, convergence
