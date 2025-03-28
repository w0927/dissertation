import numpy as np

class CircularCarFollowingModel:
    def __init__(self, 
                 track_length=2000.0,  # 圆形轨道总长度
                 initial_velocities=None, 
                 initial_positions=None, 
                 d=30.0,  # 期望车距
                 parameters=None):
        """
        初始化三车跟随模型
        
        Args:
        - track_length: 圆形轨道总长度
        - initial_velocities: 三辆车的初始速度 [L, F1, F2]
        - initial_positions: 三辆车的初始位置 [L, F1, F2]
        - d: 期望车距，可以是单一值或范围元组(min_d, max_d)
        - parameters: 模型额外参数
        """
        # 轨道长度
        self.track_length = track_length
        
        # 默认速度设置
        if initial_velocities is None:
            initial_velocities = [20.0, 20.0, 20.0]
        
        # 默认位置设置
        if initial_positions is None:
            initial_positions = [1000.0, 970.0, 940.0]
        
        # 默认参数
        if parameters is None:
            parameters = {}
        
        # 初始化车辆状态
        self.x0 = float(initial_positions[0])  # 领头车位置
        self.y1 = float(initial_positions[1])  # 第一跟随车位置
        self.y2 = float(initial_positions[2])  # 第二跟随车位置
        
        self.v0 = float(initial_velocities[0])  # 领头车速度
        self.v1 = float(initial_velocities[1])  # 第一跟随车速度
        self.v2 = float(initial_velocities[2])  # 第二跟随车速度
        
        # 处理期望车距参数 - 检查d是否为元组
        if isinstance(d, tuple) and len(d) == 2:
            self.min_distance = float(d[0])  # 最小期望距离
            self.max_distance = float(d[1])  # 最大期望距离
            self.d = (self.min_distance + self.max_distance) / 2.0  # 使用平均值作为参考
        else:
            # 使用单一值
            self.d = float(d)
            self.min_distance = self.d  # 最小允许距离
            self.max_distance = self.d + 20.0  # 最大允许距离
        
        # 存储额外参数
        self.np = parameters.get('np', {})
        
        # 模拟参数 - 调整时间步长以平衡细节和稳定性
        self.dt = 2.0       # 时间步长 [s] (改为2秒，原为5秒)
        self.t_max = 300.0  # 最大模拟时间 [s]
        self.time = np.arange(0, self.t_max, self.dt)
        
        # 初始距离计算
        self.x1 = self.circular_distance(self.x0, self.y1)  # L和F1之间的距离
        self.x2 = self.circular_distance(self.y1, self.y2)  # F1和F2之间的距离
        
        # 存储模拟结果的容器
        self.history = {
            'time': self.time,
            'x0': [self.x0],    # 领头车位置
            'y1': [self.y1],    # 第一跟随车位置
            'y2': [self.y2],    # 第二跟随车位置
            'x1': [self.x1],    # L和F1之间的距离
            'x2': [self.x2],    # F1和F2之间的距离
            'v0': [self.v0],    # 领头车速度
            'v1': [self.v1],    # 第一跟随车速度
            'v2': [self.v2],    # 第二跟随车速度
            'mode': ['00']      # 系统模式
        }
    
    def circular_distance(self, pos1, pos2):
        """
        计算圆形轨道上两点之间的最短距离
        
        Args:
        - pos1: 第一个点的位置
        - pos2: 第二个点的位置
        
        Returns:
        - 两点间的最短距离
        """
        diff = abs(pos1 - pos2)
        return min(diff, self.track_length - diff)
    
    def heaviside_step(self, x):
        """
        Heaviside阶跃函数: 如果x > 0返回1，否则返回0
        
        Args:
        - x: 输入值
        
        Returns:
        - 阶跃函数结果
        """
        return 1 if x > 0 else 0

    def run_simulation(self):
        """
        运行三车跟随模型模拟，增加随机性以模拟幻影拥堵，但保持合理的稳定性
        增强了车辆在距离过大时的加速响应
        
        Returns:
        - 模拟历史记录
        """
        # 导入numpy以支持随机性
        import numpy as np
        
        # 模拟波浪周期的正弦波
        wave_period = 60  # 60秒的周期
        
        for t_idx in range(len(self.time) - 1):
            current_time = self.time[t_idx]
            
            # 引入领头车的速度变化
            # 1. 随机扰动
            if np.random.random() < 0.20:  # 20%的概率随机变速
                self.v0 += np.random.normal(0, 1.5)  # 添加随机速度变化
            
            # 2. 周期性变化（模拟驾驶员无意识的轻踩油门/刹车）
            self.v0 += 0.5 * np.sin(current_time * 2 * np.pi / wave_period)
            
            # 3. 偶尔的减速事件（较为温和的版本）
            if np.random.random() < 0.015:  # 1.5%的概率减速
                self.v0 -= 3.0 + np.random.random() * 2.0  # 3-5 m/s^2 减速
                
            # 计算距离，添加感知误差（但控制在合理范围内）
            perception_error_1 = np.random.normal(0, 0.08)  # 8%的感知误差
            perception_error_2 = np.random.normal(0, 0.08)  # 8%的感知误差
            distance_L_F1 = self.circular_distance(self.x0, self.y1) * (1 + perception_error_1)
            distance_F1_F2 = self.circular_distance(self.y1, self.y2) * (1 + perception_error_2)

            # 计算随机阈值（模拟驾驶员注意力变化）
            threshold_lower_1 = self.min_distance * (1 + np.random.normal(0, 0.05))
            threshold_upper_1 = self.max_distance * (1 + np.random.normal(0, 0.05))
            threshold_lower_2 = self.min_distance * (1 + np.random.normal(0, 0.05))
            threshold_upper_2 = self.max_distance * (1 + np.random.normal(0, 0.05))

            # L和F1距离控制（增强了加速响应）
            if distance_L_F1 < threshold_lower_1:
                # 距离过近，F1需要减速，L需要加速
                accel_1 = -2.0 + np.random.normal(0, 0.6)  # 添加随机反应
                accel_0 = 1.0 + np.random.normal(0, 0.3)
            elif distance_L_F1 > threshold_upper_1:
                # 距离过远，F1需要加速，L可能减速
                # 增强加速响应：从2.0增加到4.0
                accel_1 = 4.0 + np.random.normal(0, 0.5)  # 增加加速度，减少随机性
                accel_0 = -1.0 + np.random.normal(0, 0.3)
                
                # 距离非常远时进一步增强反应
                if distance_L_F1 > threshold_upper_1 * 1.2:  # 超过阈值20%
                    accel_1 += 1.0  # 额外加速
            else:
                # 在安全距离范围内
                accel_1 = np.random.normal(0, 0.3)  # 小的随机波动
                accel_0 = np.random.normal(0, 0.3)

            # F1和F2距离控制（增强了加速响应）
            if distance_F1_F2 < threshold_lower_2:
                # 距离过近，F2需要减速，F1需要加速
                accel_2 = -2.0 + np.random.normal(0, 0.6)
                accel_1 += 1.0 + np.random.normal(0, 0.3)
            elif distance_F1_F2 > threshold_upper_2:
                # 距离过远，F2需要加速
                # 增强加速响应：从2.0增加到4.0
                accel_2 = 4.0 + np.random.normal(0, 0.5)  # 增加加速度，减少随机性
                accel_1 += -1.0 + np.random.normal(0, 0.3)
                
                # 距离非常远时进一步增强反应
                if distance_F1_F2 > threshold_upper_2 * 1.2:  # 超过阈值20%
                    accel_2 += 1.0  # 额外加速
            else:
                # 在安全距离范围内
                accel_2 = np.random.normal(0, 0.3)
                # F1的加速度已经在前面处理过了

            # 模拟驾驶员反应时间延迟（降低延迟概率，从25%降至20%）
            if np.random.random() < 0.20:  # 20%概率出现延迟
                accel_1 *= 0.6  # 减弱响应，但减弱程度降低（从0.4增至0.6）
                accel_2 *= 0.6  # 减弱响应，但减弱程度降低
            
            # 过度反应模拟（但不要太极端）
            if distance_L_F1 < self.min_distance * 0.85:  # 距离小于最小距离的85%
                accel_1 = -2.5 - np.random.random() * 1.5  # 比正常减速更强但不太极端
            if distance_F1_F2 < self.min_distance * 0.85:
                accel_2 = -2.5 - np.random.random() * 1.5

            # 计算布尔距离指示器
            lambda1 = self.heaviside_step(self.x1 - self.d)
            lambda2 = self.heaviside_step(self.x2 - self.d)
            
            # 确定当前模式
            mode = f"{lambda1}{lambda2}"

            # 更新速度（保持原有的速度限制逻辑）
            min_speed = 16.67  # 60 km/h 对应的 m/s
            max_speed = 22.22  # 80 km/h 对应的 m/s

            # 更新速度
            self.v0 += accel_0 * self.dt
            self.v1 += accel_1 * self.dt
            self.v2 += accel_2 * self.dt

            # 限制速度范围（增加跟随车的最大速度限制）
            self.v0 = max(min_speed, min(self.v0, max_speed))
            self.v1 = max(min_speed, min(self.v1, self.v0 * 1.12))  # 允许F1超过领头车的速度上限增至12%
            self.v2 = max(min_speed, min(self.v2, self.v1 * 1.12))  # 允许F2超过F1的速度上限增至12%

            # 更新位置
            self.x0 = (self.x0 + self.v0 * self.dt) % self.track_length
            self.y1 = (self.y1 + self.v1 * self.dt) % self.track_length
            self.y2 = (self.y2 + self.v2 * self.dt) % self.track_length

            # 重新计算距离
            self.x1 = self.circular_distance(self.x0, self.y1)
            self.x2 = self.circular_distance(self.y1, self.y2)

            # 存储历史记录
            self.history['x0'].append(self.x0)
            self.history['y1'].append(self.y1)
            self.history['y2'].append(self.y2)
            self.history['x1'].append(self.x1)
            self.history['x2'].append(self.x2)
            self.history['v0'].append(self.v0)
            self.history['v1'].append(self.v1)
            self.history['v2'].append(self.v2)
            self.history['mode'].append(mode)
        
        return self.history