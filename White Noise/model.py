import numpy as np
import matplotlib.pyplot as plt

class CircularCarFollowingModel:
    def __init__(self, 
                 # 基础物理参数
                 track_length=6000.0,
                 initial_velocities=None,
                 initial_positions=None,
                 d=40.0,  # 期望跟车距离阈值
                 
                 # 修正后的参数 - 满足稳定性条件
                 # F1车的简化公式系数（只看前车）
                 a11=0.5,   # λ1=1时的速度调整系数
                 a0=0.3,    # λ1=0时的速度调整系数
                 
                 # F2车的系数（修正符号和数值）
                 b1=1.0,    # λ2=1时的速度调整
                 b0=1.5,    # λ2=0时的速度调整
                 
                 # L车的系数（修正符号和数值）
                 c1=0.3,    # λ1=1时的速度调整
                 c0=0.5,    # λ1=0时的速度调整
                 
                 # 响应参数
                 response_factor=0.3,  # 速度响应系数（控制加速度大小）
                 
                 # 仿真参数
                 dt=2.0,
                 t_max=500.0,  # 增加总时间以观察稳定后的噪声效果
                 
                 # 延迟噪声参数
                 enable_L_noise=False,    # 是否启用L车噪声
                 noise_std=1.0,           # 噪声标准差 (m/s)
                 noise_start_time=150.0,  # 新增：噪声开始时间 (秒)
                 noise_seed=None):        # 随机种子

        
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
        
        # 修正后的参数（去掉base_velocity）
        self.lambda_params = {
            'a11': a11, 'a0': a0,    # F1车参数
            'b1': b1, 'b0': b0,      # F2车参数
            'c1': c1, 'c0': c0       # L车参数
        }
        
        # 响应参数
        self.response_factor = response_factor
        
        # 延迟噪声设置
        self.enable_L_noise = enable_L_noise
        self.noise_std = noise_std
        self.noise_start_time = noise_start_time  # 新增
        self.noise_active = False  # 新增：噪声是否已激活
        
        # 设置随机种子
        if noise_seed is not None:
            np.random.seed(noise_seed)
        
        # 初始化车辆位置
        if initial_positions is None:
            # 默认等间距初始化
            self.x0 = 3000.0  # 调整到轨道中间
            self.y1 = (self.x0 - self.d) % self.track_length
            self.y2 = (self.y1 - self.d) % self.track_length
        else:
            self.x0 = float(initial_positions[0])
            self.y1 = float(initial_positions[1])
            self.y2 = float(initial_positions[2])
        
        # 初始速度设置
        if initial_velocities is None:
            initial_velocities = [60.0, 60.0, 60.0]  # 统一使用60 m/s
        
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
            'accel_0': [], 'accel_1': [], 'accel_2': [],
            'L_noise': [],
            'L_speed_without_noise': [],
            'noise_active_flag': []  # 新增：记录噪声是否激活
        }
        
        print(f"🔧 初始化设置:")
        if initial_positions is None:
            print(f"   等间距初始化 (间距={self.d}m)")
        else:
            print(f"   自定义位置初始化")
        print(f"   轨道长度: {self.track_length}m")
        print(f"   L车位置: {self.x0:.1f}m, 速度: {self.v0:.1f}m/s")
        print(f"   F1车位置: {self.y1:.1f}m, 速度: {self.v1:.1f}m/s, 距离L车: {self.x1:.1f}m")
        print(f"   F2车位置: {self.y2:.1f}m, 速度: {self.v2:.1f}m/s, 距离F1车: {self.x2:.1f}m")
        print(f"   目标距离: {self.d}m")
        print(f"   ⭐ 无base_velocity，车辆保持当前速度除非受lambda规则调整")
        
        if self.enable_L_noise:
            print(f"🔊 延迟噪声设置:")
            print(f"   噪声标准差: {self.noise_std:.2f} m/s")
            print(f"   噪声开始时间: {self.noise_start_time:.1f} 秒")
            print(f"   稳定阶段时长: {self.noise_start_time:.1f} 秒")
        else:
            print(f"🔇 未启用噪声")

    def generate_white_noise(self, current_time):
        # 检查是否应该激活噪声
        if self.enable_L_noise and current_time >= self.noise_start_time:
            if not self.noise_active:
                self.noise_active = True
                print(f"\n🔊 {current_time:.1f}秒: 开始引入L车噪声！")
                print("-" * 40)
            
            # 生成噪声
            noise = np.random.normal(0, self.noise_std)
            max_noise = 3 * self.noise_std
            noise = np.clip(noise, -max_noise, max_noise)
            return noise
        else:
            return 0.0

    def circular_distance(self, pos1, pos2):
        """计算环形轨道距离"""
        diff = abs(pos1 - pos2)
        return min(diff, self.track_length - diff)
    
    def heaviside_step(self, x):
        """Heaviside阶跃函数"""
        return 1 if x > 0 else 0
    
    def calculate_target_velocities(self, lambda1, lambda2):
        
        # F1车目标速度：当前速度 + lambda调整
        target_v1 = (self.v1 + 
                     self.lambda_params['a11'] * lambda1 - 
                     self.lambda_params['a0'] * (1 - lambda1))
        
        # F2车目标速度：当前速度 + lambda调整
        target_v2 = (self.v2 + 
                     self.lambda_params['b1'] * lambda2 - 
                     self.lambda_params['b0'] * (1 - lambda2))
        
        # L车目标速度：当前速度 + lambda调整
        target_v0 = (self.v0 - 
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
    
    def assess_stability(self, start_time=100.0, end_time=None):
        """
        评估系统稳定性
        
        Args:
            start_time: 开始评估的时间
            end_time: 结束评估的时间，None表示到噪声开始前
        """
        if end_time is None:
            end_time = self.noise_start_time - 10.0  # 噪声开始前10秒
        
        # 找到对应的时间索引
        start_idx = int(start_time / self.dt)
        end_idx = int(end_time / self.dt)
        
        if start_idx >= len(self.history['x1']) or end_idx >= len(self.history['x1']):
            return False
        
        # 提取稳定阶段的数据
        distances_x1 = self.history['x1'][start_idx:end_idx]
        distances_x2 = self.history['x2'][start_idx:end_idx]
        velocities_v0 = self.history['v0'][start_idx:end_idx]
        velocities_v1 = self.history['v1'][start_idx:end_idx]
        velocities_v2 = self.history['v2'][start_idx:end_idx]
        
        # 计算稳定性指标
        x1_std = np.std(distances_x1)
        x2_std = np.std(distances_x2)
        v0_std = np.std(velocities_v0)
        v1_std = np.std(velocities_v1)
        v2_std = np.std(velocities_v2)
        
        # 计算与目标的偏差
        x1_deviation = abs(np.mean(distances_x1) - self.d)
        x2_deviation = abs(np.mean(distances_x2) - self.d)
        
        print(f"\n📊 稳定性评估 ({start_time:.0f}s - {end_time:.0f}s):")
        print(f"   距离标准差: L-F1={x1_std:.2f}m, F1-F2={x2_std:.2f}m")
        print(f"   速度标准差: L={v0_std:.2f}, F1={v1_std:.2f}, F2={v2_std:.2f} m/s")
        print(f"   距离偏差: L-F1={x1_deviation:.2f}m, F1-F2={x2_deviation:.2f}m")
        
        # 稳定性判断标准
        distance_stable = x1_std < 5.0 and x2_std < 5.0
        velocity_stable = v0_std < 2.0 and v1_std < 2.0 and v2_std < 2.0
        target_reached = x1_deviation < 10.0 and x2_deviation < 10.0
        
        is_stable = distance_stable and velocity_stable and target_reached
        
        print(f"   稳定性状态: {'✅ 系统稳定' if is_stable else '❌ 系统不稳定'}")
        
        return is_stable
    
    def run_simulation(self):
        """运行仿真"""
        print("🚗 运行延迟噪声的车辆跟随仿真...")
        print(f"📐 公式（相对于当前速度）:")
        print(f"   F1: v1' = v1 + {self.lambda_params['a11']}*λ1 - {self.lambda_params['a0']}*(1-λ1)")
        print(f"   F2: v2' = v2 + {self.lambda_params['b1']}*λ2 - {self.lambda_params['b0']}*(1-λ2)")
        print(f"   L:  v0' = v0 - {self.lambda_params['c1']}*λ1 + {self.lambda_params['c0']}*(1-λ1)")
        print(f"🎯 目标距离阈值: {self.d}m")
        print(f"⏰ 总仿真时间: {self.t_max}s")
        if self.enable_L_noise:
            print(f"🔊 噪声将在 {self.noise_start_time}s 时引入")
        print("-" * 50)
        
        # 调试统计
        total_noise_applied = 0
        noise_count = 0
        max_noise = 0
        min_noise = 0
        
        for t_idx in range(len(self.time) - 1):
            current_time = self.time[t_idx]
            
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
            # F1和F2车正常更新
            self.v1 += safe_accelerations[1] * self.dt  
            self.v2 += safe_accelerations[2] * self.dt
            
            # L车特殊处理：先记录无噪声的速度变化，然后添加噪声
            v0_without_noise = self.v0 + safe_accelerations[0] * self.dt
            
            # 生成噪声并应用到L车（考虑延迟）
            current_noise = self.generate_white_noise(current_time)
            self.v0 = v0_without_noise + current_noise
            
            # 噪声统计
            if current_noise != 0:
                total_noise_applied += abs(current_noise)
                noise_count += 1
                max_noise = max(max_noise, current_noise)
                min_noise = min(min_noise, current_noise)
                
                # 刚开始引入噪声时输出调试信息
                if noise_count <= 10:
                    print(f"⏱️ {current_time:.1f}s: L车速度{v0_without_noise:.1f}→{self.v0:.1f} (噪声: {current_noise:+.2f})")
            
            # 记录数据
            self.history['L_noise'].append(current_noise)
            self.history['L_speed_without_noise'].append(v0_without_noise)
            self.history['noise_active_flag'].append(self.noise_active)
            
            # 速度限制
            min_speed, max_speed = 5.0, 80.0
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
            
            # 在噪声引入前评估稳定性
            if self.enable_L_noise and abs(current_time - (self.noise_start_time - 20.0)) < self.dt:
                print(f"\n⏰ {current_time:.1f}s: 即将在{self.noise_start_time:.1f}s引入噪声...")
                self.assess_stability(start_time=50.0, end_time=current_time)
        
        print("✅ 仿真完成！")
        
        # 最终状态报告
        final_x1 = self.history['x1'][-1]
        final_x2 = self.history['x2'][-1]
        final_v0 = self.history['v0'][-1]
        final_v1 = self.history['v1'][-1]
        final_v2 = self.history['v2'][-1]
        
        print(f"\n📊 最终状态:")
        print(f"   L-F1距离: {final_x1:.1f}m (目标: {self.d}m, 偏差: {abs(final_x1-self.d):.1f}m)")
        print(f"   F1-F2距离: {final_x2:.1f}m (目标: {self.d}m, 偏差: {abs(final_x2-self.d):.1f}m)")
        print(f"   最终速度: L={final_v0:.1f}, F1={final_v1:.1f}, F2={final_v2:.1f} m/s")
        
        # 噪声统计
        if self.enable_L_noise and noise_count > 0:
            noise_std_actual = np.std([n for n in self.history['L_noise'] if n != 0])
            noise_mean = np.mean([n for n in self.history['L_noise'] if n != 0])
            avg_noise_magnitude = total_noise_applied / noise_count
            noise_duration = self.t_max - self.noise_start_time
            avg_speed = np.mean(self.history['v0'])
            
            print(f"\n🔊 噪声详细统计:")
            print(f"   噪声持续时间: {noise_duration:.1f}s")
            print(f"   噪声均值: {noise_mean:.3f} m/s (理论值: 0)")
            print(f"   噪声标准差: {noise_std_actual:.3f} m/s (设定值: {self.noise_std:.3f})")
            print(f"   平均噪声幅度: {avg_noise_magnitude:.3f} m/s")
            print(f"   噪声范围: [{min_noise:.2f}, {max_noise:.2f}] m/s")
            print(f"   平均速度: {avg_speed:.1f} m/s")
            print(f"   相对噪声强度: ±{(self.noise_std/avg_speed*100):.1f}%")
        
        return self.history
    
    def plot_results(self):
        """绘图功能 - 增强版，确保显示噪声对比"""
        try:
            # 强制显示5个子图（包括噪声图）
            fig, axs = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
            
            time = self.history['time'][:len(self.history['x0'])]
            
            # 1. 位置
            axs[0].plot(time, self.history['x0'], 'r-', label='Lead Car (L)', linewidth=2)
            axs[0].plot(time, self.history['y1'], 'g-', label='Following Car 1 (F1)', linewidth=2)
            axs[0].plot(time, self.history['y2'], 'b-', label='Following Car 2 (F2)', linewidth=2)
            if self.enable_L_noise:
                axs[0].axvline(x=self.noise_start_time, color='orange', linestyle='--', 
                              label=f'噪声开始 ({self.noise_start_time}s)', alpha=0.8, linewidth=2)
            axs[0].set_ylabel('Position [m]')
            axs[0].set_title('🚗 Vehicle Positions over Time')
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)
            
            # 2. 距离
            axs[1].plot(time, self.history['x1'], 'g-', label='Distance L-F1', linewidth=2)
            axs[1].plot(time, self.history['x2'], 'b-', label='Distance F1-F2', linewidth=2)
            axs[1].axhline(y=self.d, color='r', linestyle='--', label=f'Threshold (d={self.d}m)', linewidth=2)
            if self.enable_L_noise:
                axs[1].axvline(x=self.noise_start_time, color='orange', linestyle='--', 
                              label=f'噪声开始', alpha=0.8, linewidth=2)
            axs[1].set_ylabel('Distance [m]')
            axs[1].set_title('📏 Inter-vehicle Distances')
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)
            
            # 3. 速度 - 重点显示噪声对比
            axs[2].plot(time, self.history['v0'], 'r-', label='Lead Car (L) - 实际速度', linewidth=2)
            axs[2].plot(time, self.history['v1'], 'g-', label='Following Car 1 (F1)', linewidth=2)
            axs[2].plot(time, self.history['v2'], 'b-', label='Following Car 2 (F2)', linewidth=2)
            
            # 强制显示无噪声的L车速度对比
            if len(self.history['L_speed_without_noise']) > 0:
                time_noise = time[:len(self.history['L_speed_without_noise'])]
                axs[2].plot(time_noise, self.history['L_speed_without_noise'], 'r--', 
                           label='Lead Car (L) - 无噪声速度', linewidth=2, alpha=0.8)
                print(f"✅ 绘制无噪声L车速度对比线，数据点数: {len(self.history['L_speed_without_noise'])}")
            
            if self.enable_L_noise:
                axs[2].axvline(x=self.noise_start_time, color='orange', linestyle='--', 
                              label=f'噪声开始', alpha=0.8, linewidth=2)
                
                # 添加文本说明
                axs[2].text(self.noise_start_time + 20, max(self.history['v0']) * 0.9,
                           f'噪声开始\nσ={self.noise_std}m/s', 
                           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3),
                           fontsize=10)
            
            axs[2].set_ylabel('Velocity [m/s]')
            axs[2].set_title('🏃 Vehicle Velocities over Time (红实线=有噪声, 红虚线=无噪声)')
            axs[2].legend()
            axs[2].grid(True, alpha=0.3)
            
            # 4. Lambda状态
            lambda1_states = self.history['lambda1']
            time_lambda = time[:len(lambda1_states)]
            axs[3].step(time_lambda, lambda1_states, 'g-', linewidth=3, where='post')
            axs[3].set_yticks([0, 1])
            axs[3].set_yticklabels(['前车近 (λ1=0)', '前车远 (λ1=1)'])
            if self.enable_L_noise:
                axs[3].axvline(x=self.noise_start_time, color='orange', linestyle='--', 
                              label=f'噪声开始', alpha=0.8, linewidth=2)
                axs[3].legend()
            axs[3].set_ylabel('F1车状态')
            axs[3].set_title('🎯 F1车前车距离状态 (λ1)')
            axs[3].grid(True, alpha=0.3)
            
            # 5. 噪声图 - 强制显示
            if len(self.history['L_noise']) > 0:
                time_noise = time[:len(self.history['L_noise'])]
                noise_values = self.history['L_noise']
                
                # 绘制噪声
                axs[4].plot(time_noise, noise_values, 'r-', alpha=0.7, linewidth=1, label='实际噪声')
                axs[4].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                if self.enable_L_noise:
                    axs[4].axhline(y=self.noise_std, color='k', linestyle=':', alpha=0.5, 
                                  label=f'±σ ({self.noise_std:.1f})')
                    axs[4].axhline(y=-self.noise_std, color='k', linestyle=':', alpha=0.5)
                    axs[4].axvline(x=self.noise_start_time, color='orange', linestyle='--', 
                                  label=f'噪声开始 ({self.noise_start_time}s)', alpha=0.8, linewidth=2)
                
                # 计算噪声统计
                active_noise = [n for n in noise_values if n != 0]
                if active_noise:
                    actual_std = np.std(active_noise)
                    actual_mean = np.mean(active_noise)
                    axs[4].text(0.02, 0.95, 
                               f'噪声统计:\n均值: {actual_mean:.2f}\n标准差: {actual_std:.2f}\n设定值: {self.noise_std:.1f}',
                               transform=axs[4].transAxes, fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                               verticalalignment='top')
                
                axs[4].set_ylabel('Noise [m/s]')
                axs[4].set_xlabel('Time [s]')
                axs[4].set_title(f'🔊 L车白噪声 (σ={self.noise_std:.1f} m/s)')
                axs[4].legend()
                axs[4].grid(True, alpha=0.3)
                
                print(f"✅ 绘制噪声图，数据点数: {len(noise_values)}")
                print(f"✅ 活跃噪声点数: {len(active_noise)}")
            else:
                # 如果没有噪声数据，显示说明
                axs[4].text(0.5, 0.5, '无噪声数据', transform=axs[4].transAxes, 
                           fontsize=16, ha='center', va='center')
                axs[4].set_ylabel('Noise [m/s]')
                axs[4].set_xlabel('Time [s]')
                axs[4].set_title('🔇 噪声图 (无数据)')
                axs[4].grid(True, alpha=0.3)
                print("❌ 无噪声数据可绘制")
            
            plt.tight_layout()
            plt.show()
            
            print("📈 图表显示完成")
            
            # 额外的调试信息
            print(f"\n🔧 调试信息:")
            print(f"   enable_L_noise: {self.enable_L_noise}")
            print(f"   noise_start_time: {self.noise_start_time}")
            print(f"   noise_std: {self.noise_std}")
            print(f"   L_noise数据长度: {len(self.history['L_noise'])}")
            print(f"   L_speed_without_noise数据长度: {len(self.history['L_speed_without_noise'])}")
            if len(self.history['L_noise']) > 0:
                non_zero_noise = [n for n in self.history['L_noise'] if n != 0]
                print(f"   非零噪声点数: {len(non_zero_noise)}")
                if non_zero_noise:
                    print(f"   噪声范围: [{min(non_zero_noise):.2f}, {max(non_zero_noise):.2f}]")
            
        except Exception as e:
            print(f"⚠️  绘图失败: {e}")
            import traceback
            traceback.print_exc()


def create_custom_spacing_positions(L_position, L_to_F1_distance, F1_to_F2_distance, track_length=6000):
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
    """主程序 - 测试延迟引入噪声的车辆跟随模型"""
    print("🚗" + "="*70)
    print("   延迟引入噪声的车辆跟随模型测试")
    print("="*70 + "🚗")
    
    # 测试场景：先稳定150秒，然后引入不同强度的噪声
    test_scenarios = [
        {"name": "无噪声对照组", "noise_std": 0.0, "noise_start": 150.0},
        {"name": "轻度噪声", "noise_std": 2.0, "noise_start": 150.0},
        {"name": "中度噪声", "noise_std": 5.0, "noise_start": 150.0},
        {"name": "强度噪声", "noise_std": 8.0, "noise_start": 150.0}
    ]
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n{'='*30} 场景 {i+1}: {scenario['name']} {'='*30}")
        
        enable_noise = scenario['noise_std'] > 0
        custom_positions = create_custom_spacing_positions(
            L_position=3000,     # 轨道中间位置
            L_to_F1_distance=50, # 初始间距大于目标值，观察收敛过程
            F1_to_F2_distance=50,
            track_length=6000
        )
        
        model = CircularCarFollowingModel(
            # 基础设置
            track_length=6000.0,   # 较长轨道，动画更慢
            d=40.0,                # 目标距离
            initial_positions=custom_positions,
            initial_velocities=[60.0, 60.0, 60.0],
            t_max=400.0,           # 总仿真时间400秒
            
            # 稳定性参数
            a11=0.5, a0=0.3,
            b1=1.0, b0=1.5,
            c1=0.3, c0=0.5,
            
            # 延迟噪声参数
            enable_L_noise=enable_noise,
            noise_std=scenario['noise_std'],
            noise_start_time=scenario['noise_start'],
            noise_seed=42
        )
        
        # 验证稳定性条件（只在第一次显示）
        if i == 0:
            print("\n🔍 验证稳定性条件:")
            params = model.lambda_params
            
            # 检查正值条件
            pos_cond = {
                "b0-c1": params['b0'] - params['c1'],
                "a01+c0": -params['a0'] + params['c0'],
                "a00+c0": -params['a0'] + params['c0'],
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
        
        # 绘制结果（只绘制最后一个场景，或者你想看所有场景）
        if i == len(test_scenarios) - 1:  # 只绘制最后一个场景
        # if True:  # 取消注释这行，注释上一行，可以看到所有场景的图
            model.plot_results()
        
        print(f"✅ 场景 {i+1} 完成\n")
    
    print("🎉 所有测试完成！")
    print("\n📋 关键特性:")
    print("✅ 系统先稳定150秒，然后引入噪声")
    print("✅ 可以清楚观察噪声对稳定系统的影响")
    print("✅ 增加了稳定性自动评估功能")
    print("✅ 图表中橙色虚线标记噪声引入时间点")
    print("✅ 轨道长度6000m，动画速度适中")
    
    print("\n🎯 观察要点:")
    print("1. 0-150s: 系统如何从初始状态收敛到稳定")
    print("2. 150s时刻: 噪声引入的瞬间影响")
    print("3. 150s之后: F1、F2如何响应L车的噪声扰动")
    print("4. 距离控制: 系统是否还能维持目标距离")
    print("5. 速度图中红线vs红虚线: 噪声的直观效果")
    
    print("\n💡 实验设计:")
    print("- 稳定阶段: 0-150s (观察自然收敛)")
    print("- 噪声阶段: 150-400s (观察扰动响应)")
    print("- 对比分析: 不同噪声强度的系统响应差异")

if __name__ == "__main__":
    main()