import numpy as np

class CircularCarFollowingModel:
    def __init__(self, 
                 # 基础物理参数
                 track_length=2000.0,
                 initial_velocities=None,
                 initial_positions=None,
                 d=40.0,  # 期望跟车距离阈值
                 
                 # 基础速度参数
                 base_velocity=20.0,  # v：系统基础速度
                 
                 # Lambda公式的速度调整系数（这些控制速度响应强度）
                 # F1车的四种模式系数
                 a11=0.5,   # 稀疏模式(λ1=1,λ2=1)的速度调整
                 a10=-1.0,  # 前松后紧(λ1=1,λ2=0)的速度调整  
                 a01=1.5,   # 前紧后松(λ1=0,λ2=1)的速度调整
                 a00=-2.0,  # 密集模式(λ1=0,λ2=0)的速度调整
                 
                 # F2车的系数 
                 b1=1.0,    # λ2=1时的速度调整
                 b0=-1.5,   # λ2=0时的速度调整
                 
                 # L车的系数
                 c1=-0.3,   # λ1=1时的速度调整
                 c0=0.5,    # λ1=0时的速度调整
                 
                 # 响应参数
                 response_factor=0.3,  # 速度响应系数（控制加速度大小）
                 
                 # 仿真参数
                 dt=2.0,
                 t_max=300.0):
        """
        基于Lambda公式的车辆跟随模型
        
        工作原理：
        1. 根据当前距离计算λ1, λ2
        2. 用你的公式计算目标速度
        3. 模型自动计算达到目标速度所需的加速度
        4. 更新车辆状态
        
        参数说明：
        - a, b, c系数：控制在不同λ状态下的速度调整幅度
        - response_factor：控制车辆对速度差异的响应强度（影响加速度大小）
        """
        
        # 基础参数
        self.track_length = track_length
        
        # 处理d参数（可能是单个值或元组）
        if isinstance(d, tuple):
            self.d = (d[0] + d[1]) / 2.0  # 使用平均值作为阈值
            self.min_distance = d[0]      # 使用元组的最小值
            self.max_distance = d[1]      # 使用元组的最大值
        else:
            self.d = float(d)
            self.min_distance = d - 10.0
            self.max_distance = d + 10.0
        
        # Lambda公式参数
        self.base_velocity = base_velocity
        self.lambda_params = {
            'v': base_velocity,
            'a11': a11, 'a10': a10, 'a01': a01, 'a00': a00,
            'b1': b1, 'b0': b0,
            'c1': c1, 'c0': c0
        }
        
        # 响应参数
        self.response_factor = response_factor
        
        # 初始化车辆状态
        if initial_velocities is None:
            initial_velocities = [base_velocity, base_velocity, base_velocity]
        if initial_positions is None:
            initial_positions = [1000.0, 960.0, 930.0]
            
        self.x0 = float(initial_positions[0])  # L车位置
        self.y1 = float(initial_positions[1])  # F1车位置  
        self.y2 = float(initial_positions[2])  # F2车位置
        
        self.v0 = float(initial_velocities[0])  # L车速度
        self.v1 = float(initial_velocities[1])  # F1车速度
        self.v2 = float(initial_velocities[2])  # F2车速度
        
        # 仿真参数
        self.dt = dt
        self.t_max = t_max
        self.time = np.arange(0, self.t_max, self.dt)
        
        # 计算初始距离
        self.x1 = self.circular_distance(self.x0, self.y1)
        self.x2 = self.circular_distance(self.y1, self.y2)
        
        # 历史记录
        self.history = {
            'time': self.time,
            'x0': [self.x0], 'y1': [self.y1], 'y2': [self.y2],
            'x1': [self.x1], 'x2': [self.x2], 
            'v0': [self.v0], 'v1': [self.v1], 'v2': [self.v2],
            'lambda1': [], 'lambda2': [], 'mode': [],
            'target_v0': [], 'target_v1': [], 'target_v2': [],  # 记录目标速度
            'accel_0': [], 'accel_1': [], 'accel_2': []         # 记录实际加速度
        }
    
    def circular_distance(self, pos1, pos2):
        """计算环形轨道距离"""
        diff = abs(pos1 - pos2)
        return min(diff, self.track_length - diff)
    
    def heaviside_step(self, x):
        """Heaviside阶跃函数：x > 0 返回1，否则返回0"""
        return 1 if x > 0 else 0
    
    def calculate_target_velocities(self, lambda1, lambda2):
        """
        使用你的Lambda公式计算目标速度
        
        y1 = v + a11*λ1*λ2 + a10*λ1*(1-λ2) + a01*(1-λ1)*λ2 + a00*(1-λ1)*(1-λ2)
        y2 = v + b1*λ2 + b0*(1-λ2)
        y0 = v + c1*λ1 + c0*(1-λ1)
        """
        v = self.lambda_params['v']
        
        # F1车目标速度（考虑四种λ组合）
        target_v1 = (v + 
                     self.lambda_params['a11'] * lambda1 * lambda2 +
                     self.lambda_params['a10'] * lambda1 * (1 - lambda2) +
                     self.lambda_params['a01'] * (1 - lambda1) * lambda2 +
                     self.lambda_params['a00'] * (1 - lambda1) * (1 - lambda2))
        
        # F2车目标速度（只依赖λ2）
        target_v2 = (v + 
                     self.lambda_params['b1'] * lambda2 + 
                     self.lambda_params['b0'] * (1 - lambda2))
        
        # L车目标速度（只依赖λ1）
        target_v0 = (v + 
                     self.lambda_params['c1'] * lambda1 + 
                     self.lambda_params['c0'] * (1 - lambda1))
        
        return target_v0, target_v1, target_v2
    
    def calculate_accelerations(self, target_velocities, current_velocities):
        """
        根据目标速度和当前速度自动计算所需加速度
        这里是模型的核心：加速度由当前状况自动决定！
        """
        target_v0, target_v1, target_v2 = target_velocities
        current_v0, current_v1, current_v2 = current_velocities
        
        # 计算速度差异
        delta_v0 = target_v0 - current_v0
        delta_v1 = target_v1 - current_v1  
        delta_v2 = target_v2 - current_v2
        
        # 根据速度差异计算加速度（这里就是自动产生的加速度！）
        accel_0 = delta_v0 * self.response_factor
        accel_1 = delta_v1 * self.response_factor
        accel_2 = delta_v2 * self.response_factor
        
        return accel_0, accel_1, accel_2
    
    def apply_safety_constraints(self, accelerations):
        """应用安全约束（防止碰撞和过度加速）"""
        accel_0, accel_1, accel_2 = accelerations
        
        # 加速度限制
        max_accel = 3.0   # 最大加速度
        max_decel = -4.0  # 最大减速度
        
        accel_0 = np.clip(accel_0, max_decel, max_accel)
        accel_1 = np.clip(accel_1, max_decel, max_accel)
        accel_2 = np.clip(accel_2, max_decel, max_accel)
        
        # 紧急制动逻辑
        emergency_distance = self.d * 0.6
        
        if self.x1 < emergency_distance:
            # F1与L车太近，F1紧急制动
            accel_1 = min(accel_1, -3.0)
            
        if self.x2 < emergency_distance:
            # F2与F1太近，F2紧急制动
            accel_2 = min(accel_2, -3.0)
        
        return accel_0, accel_1, accel_2
    
    def add_realistic_noise(self, accelerations):
        """添加现实驾驶的随机性"""
        accel_0, accel_1, accel_2 = accelerations
        
        # 添加小的随机波动（模拟驾驶员的不完美控制）
        noise_std = 0.2
        accel_0 += np.random.normal(0, noise_std)
        accel_1 += np.random.normal(0, noise_std)
        accel_2 += np.random.normal(0, noise_std)
        
        # L车作为扰动源，偶尔有额外的随机变化
        if np.random.random() < 0.05:  # 5%概率
            accel_0 += np.random.normal(0, 1.0)
        
        return accel_0, accel_1, accel_2
    
    def run_simulation(self):
        """运行仿真：每一步都由当前状况自动决定加速度"""
        print("运行基于Lambda公式的智能车辆跟随仿真...")
        print(f"Lambda参数: a11={self.lambda_params['a11']}, a10={self.lambda_params['a10']}, a01={self.lambda_params['a01']}, a00={self.lambda_params['a00']}")
        
        for t_idx in range(len(self.time) - 1):
            # 第1步：根据当前距离计算Lambda
            lambda1 = self.heaviside_step(self.x1 - self.d)
            lambda2 = self.heaviside_step(self.x2 - self.d)
            
            # 第2步：使用Lambda公式计算目标速度
            target_velocities = self.calculate_target_velocities(lambda1, lambda2)
            
            # 第3步：根据目标速度自动计算所需加速度
            current_velocities = (self.v0, self.v1, self.v2)
            accelerations = self.calculate_accelerations(target_velocities, current_velocities)
            
            # 第4步：应用安全约束
            safe_accelerations = self.apply_safety_constraints(accelerations)
            
            # 第5步：添加现实噪声
            final_accelerations = self.add_realistic_noise(safe_accelerations)
            
            # 第6步：更新速度（使用自动计算的加速度）
            self.v0 += final_accelerations[0] * self.dt
            self.v1 += final_accelerations[1] * self.dt  
            self.v2 += final_accelerations[2] * self.dt
            
            # 速度限制
            min_speed, max_speed = 5.0, 35.0
            self.v0 = np.clip(self.v0, min_speed, max_speed)
            self.v1 = np.clip(self.v1, min_speed, max_speed)
            self.v2 = np.clip(self.v2, min_speed, max_speed)
            
            # 第7步：更新位置
            self.x0 = (self.x0 + self.v0 * self.dt) % self.track_length
            self.y1 = (self.y1 + self.v1 * self.dt) % self.track_length
            self.y2 = (self.y2 + self.v2 * self.dt) % self.track_length
            
            # 第8步：重新计算距离
            self.x1 = self.circular_distance(self.x0, self.y1)
            self.x2 = self.circular_distance(self.y1, self.y2)
            
            # 第9步：记录所有数据
            mode = f"{lambda1}{lambda2}"
            
            self.history['x0'].append(self.x0)
            self.history['y1'].append(self.y1)
            self.history['y2'].append(self.y2)
            self.history['x1'].append(self.x1)
            self.history['x2'].append(self.x2)
            self.history['v0'].append(self.v0)
            self.history['v1'].append(self.v1)
            self.history['v2'].append(self.v2)
            self.history['lambda1'].append(lambda1)
            self.history['lambda2'].append(lambda2)
            self.history['mode'].append(mode)
            self.history['target_v0'].append(target_velocities[0])
            self.history['target_v1'].append(target_velocities[1])
            self.history['target_v2'].append(target_velocities[2])
            self.history['accel_0'].append(final_accelerations[0])
            self.history['accel_1'].append(final_accelerations[1])
            self.history['accel_2'].append(final_accelerations[2])
        
        print("仿真完成！系统根据交通状况自动生成了所有加速度行为")
        return self.history
    
    def print_simulation_summary(self):
        """打印仿真摘要"""
        if not self.history['mode']:
            print("请先运行仿真")
            return
            
        modes = self.history['mode']
        accels = {
            'L车': self.history['accel_0'],
            'F1车': self.history['accel_1'], 
            'F2车': self.history['accel_2']
        }
        
        print("\n=== 仿真摘要 ===")
        print("Lambda模式分布:")
        for mode in ['00', '01', '10', '11']:
            count = modes.count(mode)
            percentage = count / len(modes) * 100
            print(f"  模式{mode}: {percentage:.1f}%")
        
        print("\n自动生成的加速度统计:")
        for car, accel_list in accels.items():
            avg_accel = np.mean(accel_list)
            std_accel = np.std(accel_list)
            max_accel = max(accel_list)
            min_accel = min(accel_list)
            print(f"  {car}: 平均{avg_accel:.2f} m/s², 标准差{std_accel:.2f}, 范围[{min_accel:.1f}, {max_accel:.1f}]")

# 便捷的场景创建函数
def create_scenario(name="default", **kwargs):
    """创建不同的测试场景"""
    scenarios = {
        "default": {},
        "aggressive": {
            "a11": 1.0, "a10": -2.5, "a01": 3.0, "a00": -3.5,
            "b1": 2.0, "b0": -2.5, "response_factor": 0.4
        },
        "conservative": {
            "a11": 0.2, "a10": -0.5, "a01": 0.7, "a00": -1.0,
            "b1": 0.5, "b0": -0.8, "response_factor": 0.2
        },
        "phantom_jam": {
            "initial_velocities": [22.0, 20.0, 18.0],
            "a11": 0.8, "a10": -1.8, "a01": 2.2, "a00": -2.8,
            "response_factor": 0.35
        }
    }
    
    params = scenarios.get(name, {})
    params.update(kwargs)  # 允许覆盖参数
    
    return CircularCarFollowingModel(**params)

if __name__ == "__main__":
    print("智能Lambda车辆跟随模型测试")
    
    # 创建默认模型
    model = CircularCarFollowingModel()
    print("✓ 默认模型创建成功")
    
    # 测试不同场景
    scenarios = ["default", "aggressive", "conservative", "phantom_jam"]
    for scenario in scenarios:
        test_model = create_scenario(scenario)
        print(f"✓ {scenario}场景模型创建成功")
    
    print("所有模型测试完成！加速度将由交通状况自动决定")