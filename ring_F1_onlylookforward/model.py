import numpy as np
import matplotlib.pyplot as plt

class CircularCarFollowingModel:
    def __init__(self, 
                 # 基础物理参数
                 track_length=2000.0,
                 initial_velocities=None,
                 initial_positions=None,
                 d=40.0,  # 期望跟车距离阈值
                 
                 # 基础速度参数
                 base_velocity=20.0,  # v：系统基础速度
                 
                 # F1车的新简化公式系数（只看前车）
                 a11=0.5,   # λ1=1时的速度调整系数
                 a0=1.0,    # λ1=0时的速度调整系数（原来的负号已包含在公式中）
                 
                 # F2车的系数（保持不变）
                 b1=1.0,    # λ2=1时的速度调整
                 b0=-1.5,   # λ2=0时的速度调整
                 
                 # L车的系数（保持不变）
                 c1=-0.3,   # λ1=1时的速度调整
                 c0=0.5,    # λ1=0时的速度调整
                 
                 # 响应参数
                 response_factor=0.3,  # 速度响应系数（控制加速度大小）
                 
                 # 仿真参数
                 dt=2.0,
                 t_max=300.0):
        """
        修改后的车辆跟随模型 - F1车只看前车
        
        新的F1车公式：v1' = v + a11*λ1 - a0*(1-λ1)
        """
        
        # 基础参数
        self.track_length = track_length
        
        # 处理d参数
        if isinstance(d, tuple):
            self.d = (d[0] + d[1]) / 2.0
            self.min_distance = d[0]
            self.max_distance = d[1]
        else:
            self.d = float(d)
            self.min_distance = d - 10.0
            self.max_distance = d + 10.0
        
        # 修改后的Lambda公式参数
        self.base_velocity = base_velocity
        self.lambda_params = {
            'v': base_velocity,
            'a11': a11,  # F1车：前车远时的调整
            'a0': a0,    # F1车：前车近时的调整
            'b1': b1, 'b0': b0,  # F2车参数
            'c1': c1, 'c0': c0   # L车参数
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
            'target_v0': [], 'target_v1': [], 'target_v2': [],
            'accel_0': [], 'accel_1': [], 'accel_2': []
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
        使用修改后的公式计算目标速度
        F1车新公式：v1' = v + a11*λ1 - a0*(1-λ1)
        """
        v = self.lambda_params['v']
        
        # F1车目标速度（新的简化公式，只看前车）
        target_v1 = (v + 
                     self.lambda_params['a11'] * lambda1 - 
                     self.lambda_params['a0'] * (1 - lambda1))
        
        # F2车目标速度（保持不变）
        target_v2 = (v + 
                     self.lambda_params['b1'] * lambda2 + 
                     self.lambda_params['b0'] * (1 - lambda2))
        
        # L车目标速度（保持不变）
        target_v0 = (v + 
                     self.lambda_params['c1'] * lambda1 + 
                     self.lambda_params['c0'] * (1 - lambda1))
        
        return target_v0, target_v1, target_v2
    
    def calculate_accelerations(self, target_velocities, current_velocities):
        """根据目标速度和当前速度计算加速度"""
        target_v0, target_v1, target_v2 = target_velocities
        current_v0, current_v1, current_v2 = current_velocities
        
        delta_v0 = target_v0 - current_v0
        delta_v1 = target_v1 - current_v1  
        delta_v2 = target_v2 - current_v2
        
        accel_0 = delta_v0 * self.response_factor
        accel_1 = delta_v1 * self.response_factor
        accel_2 = delta_v2 * self.response_factor
        
        return accel_0, accel_1, accel_2
    
    def apply_safety_constraints(self, accelerations):
        """应用安全约束"""
        accel_0, accel_1, accel_2 = accelerations
        
        max_accel = 3.0
        max_decel = -4.0
        
        accel_0 = np.clip(accel_0, max_decel, max_accel)
        accel_1 = np.clip(accel_1, max_decel, max_accel)
        accel_2 = np.clip(accel_2, max_decel, max_accel)
        
        emergency_distance = self.d * 0.6
        
        if self.x1 < emergency_distance:
            accel_1 = min(accel_1, -3.0)
            
        if self.x2 < emergency_distance:
            accel_2 = min(accel_2, -3.0)
        
        return accel_0, accel_1, accel_2
    
    def run_simulation(self):
        """运行仿真"""
        print("🚗 运行修改后的车辆跟随仿真...")
        print(f"📋 F1车新公式: v1' = v + {self.lambda_params['a11']}*λ1 - {self.lambda_params['a0']}*(1-λ1)")
        print(f"👁️  F1车现在只看前车L，不再考虑后车F2")
        print(f"🎯 目标距离阈值: {self.d}m")
        print("-" * 50)
        
        for t_idx in range(len(self.time) - 1):
            # 计算Lambda
            lambda1 = self.heaviside_step(self.x1 - self.d)
            lambda2 = self.heaviside_step(self.x2 - self.d)
            
            # 计算目标速度
            target_velocities = self.calculate_target_velocities(lambda1, lambda2)
            
            # 计算加速度
            current_velocities = (self.v0, self.v1, self.v2)
            accelerations = self.calculate_accelerations(target_velocities, current_velocities)
            
            # 应用安全约束
            safe_accelerations = self.apply_safety_constraints(accelerations)
            
            # 更新速度
            self.v0 += safe_accelerations[0] * self.dt
            self.v1 += safe_accelerations[1] * self.dt  
            self.v2 += safe_accelerations[2] * self.dt
            
            # 速度限制
            min_speed, max_speed = 5.0, 35.0
            self.v0 = np.clip(self.v0, min_speed, max_speed)
            self.v1 = np.clip(self.v1, min_speed, max_speed)
            self.v2 = np.clip(self.v2, min_speed, max_speed)
            
            # 更新位置
            self.x0 = (self.x0 + self.v0 * self.dt) % self.track_length
            self.y1 = (self.y1 + self.v1 * self.dt) % self.track_length
            self.y2 = (self.y2 + self.v2 * self.dt) % self.track_length
            
            # 重新计算距离
            self.x1 = self.circular_distance(self.x0, self.y1)
            self.x2 = self.circular_distance(self.y1, self.y2)
            
            # 记录数据
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
            self.history['accel_0'].append(safe_accelerations[0])
            self.history['accel_1'].append(safe_accelerations[1])
            self.history['accel_2'].append(safe_accelerations[2])
        
        print("✅ 仿真完成！")
        return self.history
    
    def print_simulation_summary(self):
        """打印仿真摘要"""
        if not self.history['mode']:
            print("❌ 请先运行仿真")
            return
            
        modes = self.history['mode']
        lambda1_states = self.history['lambda1']
        
        print("\n" + "="*60)
        print("📊 F1车行为分析摘要（只看前车模式）")
        print("="*60)
        
        # 统计F1车的λ1状态
        lambda1_1_count = lambda1_states.count(1)  # 前车远
        lambda1_0_count = lambda1_states.count(0)  # 前车近
        total_steps = len(lambda1_states)
        
        print(f"\n🎯 F1车行为模式分布:")
        print(f"   前车远 (λ1=1): {lambda1_1_count/total_steps*100:.1f}% - 目标速度 = v + {self.lambda_params['a11']} = {self.lambda_params['v'] + self.lambda_params['a11']:.1f} m/s")
        print(f"   前车近 (λ1=0): {lambda1_0_count/total_steps*100:.1f}% - 目标速度 = v - {self.lambda_params['a0']} = {self.lambda_params['v'] - self.lambda_params['a0']:.1f} m/s")
        
        print(f"\n🔄 完整系统模式分布（λ1λ2）:")
        for mode in ['00', '01', '10', '11']:
            count = modes.count(mode)
            percentage = count / len(modes) * 100
            mode_desc = {
                '00': '前近后近', '01': '前近后远', 
                '10': '前远后近', '11': '前远后远'
            }
            print(f"   模式{mode} ({mode_desc[mode]}): {percentage:.1f}%")
        
        # 加速度统计
        accels = {
            'L车': self.history['accel_0'],
            'F1车': self.history['accel_1'], 
            'F2车': self.history['accel_2']
        }
        
        print(f"\n⚡ 自动生成的加速度统计:")
        for car, accel_list in accels.items():
            avg_accel = np.mean(accel_list)
            std_accel = np.std(accel_list)
            max_accel = max(accel_list)
            min_accel = min(accel_list)
            print(f"   {car}: 平均{avg_accel:+.2f} m/s², 标准差{std_accel:.2f}, 范围[{min_accel:+.1f}, {max_accel:+.1f}]")

    def plot_results(self):
        """简化的绘图功能"""
        try:
            fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            
            time = self.history['time'][:len(self.history['x0'])]
            
            # 1. 位置
            axs[0].plot(time, self.history['x0'], 'r-', label='Lead Car (L)', linewidth=2)
            axs[0].plot(time, self.history['y1'], 'g-', label='Following Car 1 (F1)', linewidth=2)
            axs[0].plot(time, self.history['y2'], 'b-', label='Following Car 2 (F2)', linewidth=2)
            axs[0].set_ylabel('Position [m]')
            axs[0].set_title('🚗 Vehicle Positions over Time')
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)
            
            # 2. 距离
            axs[1].plot(time, self.history['x1'], 'g-', label='Distance L-F1', linewidth=2)
            axs[1].plot(time, self.history['x2'], 'b-', label='Distance F1-F2', linewidth=2)
            axs[1].axhline(y=self.d, color='r', linestyle='--', label=f'Threshold (d={self.d}m)', linewidth=2)
            axs[1].set_ylabel('Distance [m]')
            axs[1].set_title('📏 Inter-vehicle Distances')
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)
            
            # 3. 速度
            axs[2].plot(time, self.history['v0'], 'r-', label='Lead Car (L)', linewidth=2)
            axs[2].plot(time, self.history['v1'], 'g-', label='Following Car 1 (F1)', linewidth=2)
            axs[2].plot(time, self.history['v2'], 'b-', label='Following Car 2 (F2)', linewidth=2)
            axs[2].set_ylabel('Velocity [m/s]')
            axs[2].set_title('🏃 Vehicle Velocities over Time')
            axs[2].legend()
            axs[2].grid(True, alpha=0.3)
            
            # 4. F1车的λ1状态
            lambda1_states = self.history['lambda1']
            time_lambda = time[:len(lambda1_states)]
            axs[3].step(time_lambda, lambda1_states, 'g-', linewidth=3, where='post')
            axs[3].set_yticks([0, 1])
            axs[3].set_yticklabels(['前车近 (λ1=0)', '前车远 (λ1=1)'])
            axs[3].set_ylabel('F1车状态')
            axs[3].set_xlabel('Time [s]')
            axs[3].set_title('🎯 F1车前车距离状态 (λ1)')
            axs[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print("📈 图表显示完成")
            
        except Exception as e:
            print(f"⚠️  绘图失败: {e}")

def main():
    """主程序 - 测试修改后的F1车模型"""
    print("🚗" + "="*50)
    print("   F1车只看前车的车辆跟随模型测试")
    print("="*50 + "🚗")
    
    # 测试不同的参数组合
    test_scenarios = [
        {
            "name": "默认参数",
            "params": {"a11": 1.0, "a0": 1.5}
        },
        {
            "name": "激进F1车",
            "params": {"a11": 2.0, "a0": 2.5, "response_factor": 0.4}
        },
        {
            "name": "保守F1车", 
            "params": {"a11": 0.3, "a0": 0.8, "response_factor": 0.2}
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧪 测试场景 {i}: {scenario['name']}")
        print("-" * 40)
        
        # 创建模型
        model = CircularCarFollowingModel(**scenario['params'])
        
        # 运行仿真
        model.run_simulation()
        
        # 显示结果
        model.print_simulation_summary()
        
        # 绘制图表（仅第一个场景）
        if i == 1:
            model.plot_results()
        
        print(f"✅ 场景 {i} 完成")

if __name__ == "__main__":
    main()