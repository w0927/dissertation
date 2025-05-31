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
                 
                 # 修正后的参数 - 满足稳定性条件
                 # F1车的简化公式系数（只看前车）
                 a11=0.5,   # λ1=1时的速度调整系数
                 a0=0.3,    # λ1=0时的速度调整系数（从1.0降低到0.3）
                 
                 # F2车的系数（修正符号和数值）
                 b1=1.0,    # λ2=1时的速度调整
                 b0=1.5,    # λ2=0时的速度调整（改为正值，因为公式中是-b0）
                 
                 # L车的系数（修正符号和数值）
                 c1=0.3,    # λ1=1时的速度调整（改为正值，因为公式中是-c1）
                 c0=0.5,    # λ1=0时的速度调整
                 
                 # 响应参数
                 response_factor=0.3,  # 速度响应系数（控制加速度大小）
                 
                 # 仿真参数
                 dt=2.0,
                 t_max=300.0):
        """
        修正参数的车辆跟随模型 - 最小修改版本
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
        
        # 修正后的参数（满足稳定性条件）
        self.base_velocity = base_velocity
        self.lambda_params = {
            'v': base_velocity,
            'a11': a11, 'a0': a0,    # F1车参数
            'b1': b1, 'b0': b0,      # F2车参数
            'c1': c1, 'c0': c0       # L车参数
        }
        
        # 响应参数
        self.response_factor = response_factor
        
        # 初始化车辆位置
        if initial_positions is None:
            # 默认等间距初始化：L和F1间距=F1和F2间距=d
            self.x0 = 1000.0  # L车位置
            self.y1 = (self.x0 - self.d) % self.track_length  # F1车位置，距离L车d米
            self.y2 = (self.y1 - self.d) % self.track_length  # F2车位置，距离F1车d米
        else:
            # 自定义位置
            self.x0 = float(initial_positions[0])
            self.y1 = float(initial_positions[1])
            self.y2 = float(initial_positions[2])
        
        # 所有车初始速度相同
        if initial_velocities is None:
            initial_velocities = [base_velocity, base_velocity, base_velocity]
        
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
        
        print(f"🔧 初始化设置:")
        if initial_positions is None:
            print(f"   等间距初始化 (间距={self.d}m)")
        else:
            print(f"   自定义位置初始化")
        print(f"   L车位置: {self.x0:.1f}m, 速度: {self.v0:.1f}m/s")
        print(f"   F1车位置: {self.y1:.1f}m, 速度: {self.v1:.1f}m/s, 距离L车: {self.x1:.1f}m")
        print(f"   F2车位置: {self.y2:.1f}m, 速度: {self.v2:.1f}m/s, 距离F1车: {self.x2:.1f}m")
        print(f"   目标距离: {self.d}m")
    


    def circular_distance(self, pos1, pos2):
        """计算环形轨道距离"""
        diff = abs(pos1 - pos2)
        return min(diff, self.track_length - diff)
    
    def heaviside_step(self, x):
        """Heaviside阶跃函数：x > 0 返回1，否则返回0"""
        return 1 if x > 0 else 0
    
    def calculate_target_velocities(self, lambda1, lambda2):
        """
        使用修正后的公式计算目标速度
        F1车：v1' = v + a11*λ1 - a0*(1-λ1)
        F2车：v2' = v + b1*λ2 - b0*(1-λ2)  [修正符号]
        L车： v0' = v - c1*λ1 + c0*(1-λ1)  [修正符号]
        """
        v = self.lambda_params['v']
        
        # F1车目标速度（简化公式，只看前车）
        target_v1 = (v + 
                     self.lambda_params['a11'] * lambda1 - 
                     self.lambda_params['a0'] * (1 - lambda1))
        
        # F2车目标速度（修正符号）
        target_v2 = (v + 
                     self.lambda_params['b1'] * lambda2 - 
                     self.lambda_params['b0'] * (1 - lambda2))
        
        # L车目标速度（修正符号）
        target_v0 = (v - 
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
        print("🚗 运行修正参数的车辆跟随仿真...")
        print(f"📐 修正后的公式:")
        print(f"   F1: v1' = v + {self.lambda_params['a11']}*λ1 - {self.lambda_params['a0']}*(1-λ1)")
        print(f"   F2: v2' = v + {self.lambda_params['b1']}*λ2 - {self.lambda_params['b0']}*(1-λ2)")
        print(f"   L:  v0' = v - {self.lambda_params['c1']}*λ1 + {self.lambda_params['c0']}*(1-λ1)")
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
        
        # 简单的最终状态报告
        final_x1 = self.history['x1'][-1]
        final_x2 = self.history['x2'][-1]
        print(f"📊 最终状态:")
        print(f"   L-F1距离: {final_x1:.1f}m (目标: {self.d}m, 偏差: {abs(final_x1-self.d):.1f}m)")
        print(f"   F1-F2距离: {final_x2:.1f}m (目标: {self.d}m, 偏差: {abs(final_x2-self.d):.1f}m)")
        
        return self.history
    

def create_custom_spacing_positions(L_position, L_to_F1_distance, F1_to_F2_distance, track_length=2000):
    """
    根据指定的车间距创建初始位置
    
    Args:
        L_position: L车的位置
        L_to_F1_distance: L车到F1车的距离
        F1_to_F2_distance: F1车到F2车的距离
        track_length: 轨道长度
    
    Returns:
        [L车位置, F1车位置, F2车位置]
    """
    F1_position = (L_position - L_to_F1_distance + track_length) % track_length
    F2_position = (F1_position - F1_to_F2_distance + track_length) % track_length
    
    print(f"🎯 自定义间距设置:")
    print(f"   L车位置: {L_position}m")
    print(f"   F1车位置: {F1_position}m (距离L车: {L_to_F1_distance}m)")
    print(f"   F2车位置: {F2_position}m (距离F1车: {F1_to_F2_distance}m)")
    
    return [L_position, F1_position, F2_position]

def main():
    """主程序 - 测试修正参数的车辆跟随模型"""
    print("🚗" + "="*50)
    print("   修正参数的车辆跟随模型测试")
    print("="*50 + "🚗")
    
    # 创建模型 - 可以自定义间距
    # 方式1：使用默认等间距（d=40m）
    # model = CircularCarFollowingModel(d=40.0, a11=0.5, a0=0.3, b1=1.0, b0=1.5, c1=0.3, c0=0.5)
    
    # 方式2：自定义间距 - 使用便捷函数
    custom_positions = create_custom_spacing_positions(
        L_position=1000,      # L车位置
        L_to_F1_distance=50,  # L到F1的距离
        F1_to_F2_distance=50 #F1到F2的距离
    )
    
    model = CircularCarFollowingModel(
        d=40.0,  # 目标距离
        initial_positions=custom_positions,  # 使用自定义位置
        # 修正后的参数（满足稳定性条件）
        a11=0.5, a0=0.3,    # F1车参数
        b1=1.0, b0=1.5,     # F2车参数  
        c1=0.3, c0=0.5      # L车参数
    )
    
    # 方式3：直接指定位置
    # model = CircularCarFollowingModel(
    #     d=40.0,
    #     initial_positions=[1000, 950, 920],  # [L车, F1车, F2车]位置
    #     a11=0.5, a0=0.3, b1=1.0, b0=1.5, c1=0.3, c0=0.5
    # )
    
    # 验证参数是否满足稳定性条件
    print("\n🔍 验证稳定性条件:")
    params = model.lambda_params
    
    # 检查正值条件
    pos_cond = {
        "b0-c1": params['b0'] - params['c1'],
        "a01+c0": -params['a0'] + params['c0'],  # a01 ≈ -a0
        "a00+c0": -params['a0'] + params['c0'],  # a00 ≈ -a0
        "b0+c0": params['b0'] + params['c0']
    }
    
    print("   正值条件（需要>0）:")
    for name, val in pos_cond.items():
        status = "✅" if val > 0 else "❌"
        print(f"     {name} = {val:.2f} {status}")
    
    # 检查负值条件
    neg_cond = {
        "-a11-c1": -params['a11'] - params['c1'],
        "-b1-c1": -params['b1'] - params['c1'],
        "c0-b1": params['c0'] - params['b1']
    }
    
    print("   负值条件（需要<0）:")
    for name, val in neg_cond.items():
        status = "✅" if val < 0 else "❌"
        print(f"     {name} = {val:.2f} {status}")
    
    # 运行仿真
    model.run_simulation()
    
    # 绘制结果
    model.plot_results()
    
    print("✅ 测试完成")

if __name__ == "__main__":
    main()