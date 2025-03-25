import numpy as np

class ThreeCarFollowingModel:
    def __init__(self, initial_velocities=None, initial_positions=None, d=30, parameters=None):
        """
        Initialize the three-car following model with the provided parameters.
        """
        # 设置默认速度
        if initial_velocities is None:
            initial_velocities = [20.0, 20.0, 20.0]
        elif len(initial_velocities) != 3:
            raise ValueError("initial_velocities must contain exactly 3 values")
    
        # 存储初始速度
        self.v0 = initial_velocities[0]  # 领头车初始速度
        self.v1 = initial_velocities[1]  # 第一跟随车初始速度
        self.v2 = initial_velocities[2]  # 第二跟随车初始速度
    
        # 设置默认位置
        if initial_positions is None:
            initial_positions = [200.0, 150.0, 100.0]
        elif len(initial_positions) != 3:
            raise ValueError("initial_positions must contain exactly 3 values")
    
        # 存储初始位置
        self.x0 = initial_positions[0]  # 领头车位置
        self.y1 = initial_positions[1]  # 第一跟随车位置
        self.y2 = initial_positions[2]  # 第二跟随车位置

        # 存储距离阈值
        self.d = d

        # 存储参数 - 确保这部分代码存在
        if parameters is None:
            self.parameters = {
                'a11': 2.0,  # F1 acceleration when both distances are large
                'a10': 1.5,  # F1 acceleration when front gap large, rear gap small
                'a01': 1.0,  # F1 deceleration when front gap small, rear gap large
                'a00': 0.5,  # F1 deceleration when both distances are small
                'b1': 2.0,   # F2 acceleration when distance to F1 is large
                'b0': 1.0,   # F2 deceleration when distance to F1 is small
                'c1': 1.5,   # L deceleration when distance to F1 is large
                'c0': 1.0    # L acceleration when distance to F1 is small
            }
        else:
            self.parameters = parameters  # 将传入的参数赋值给self.parameters
    
        # 计算初始距离
        self.x1 = self.x0 - self.y1  # L和F1之间的距离
        self.x2 = self.y1 - self.y2  # F1和F2之间的距离
    
        # 设置模拟时间参数
        self.dt = 5       # 时间步长 [s]
        self.t_max = 100.0   # 最大模拟时间 [s]
        self.time = np.arange(0, self.t_max, self.dt)

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
            'lambda1': [0],     # 布尔指示器x1 > d
            'lambda2': [0],     # 布尔指示器x2 > d
            'mode': ['00']      # 当前模式(λ1λ2)
        }

    def heaviside_step(self, x):
         """
         Heaviside step function: 1 if x > 0, 0 otherwise.
          """
         return 1 if x > 0 else 0
    
    def run_simulation(self):
     """
     Run the three-car following model simulation with enhanced distance control.
     """
     # initial speed
     v0 = self.v0  # speed of leader
     v1 = self.v1  # speed of F1
     v2 = self.v2  # speeed of F2
     
     # target distance
     target_distance = 30.0  # distance want to maintain
    
     # parameter extraction
     a11 = self.parameters['a11']
     a10 = self.parameters['a10']
     a01 = self.parameters['a01']
     a00 = self.parameters['a00']
     b1 = self.parameters['b1']
     b0 = self.parameters['b0']
     c1 = self.parameters['c1']
     c0 = self.parameters['c0']
    
     # enhanced range control parameters
     distance_kp = 0.15  # distance control ratio factor
     
     # cycle simulator
     for t_idx in range(len(self.time) - 1):
         # calculate the boolean  distance indicator
         lambda1 = self.heaviside_step(self.x1 - self.d)
         lambda2 = self.heaviside_step(self.x2 - self.d)
        
         # determine current mode
         mode = f"{lambda1}{lambda2}"
         
         # foundation acceleration calculation
         base_accel_1 = a11*lambda1*lambda2 + a10*lambda1*(1-lambda2) - a01*(1-lambda1)*lambda2 - a00*(1-lambda1)*(1-lambda2)
         base_accel_2 = b1*lambda2 - b0*(1-lambda2)
         base_accel_0 = -c1*lambda1 + c0*(1-lambda1)
         
         # enhanced distance control-proportional control
         # for F1: adjust the acceleration according to the distance different from the lead car
         distance_error_1 = target_distance - self.x1
         dist_control_1 = distance_kp * distance_error_1  # a positive error indicates that the distance is less than the target value and needs to be slowed down
         
         # For F2: adjust the acceleration according to the distance difference from F1
         distance_error_2 = target_distance - self.x2
         dist_control_2 = distance_kp * distance_error_2
         
         # combined based acceleration and distance control
         accel_1 = base_accel_1 - dist_control_1
         accel_2 = base_accel_2 - dist_control_2
         accel_0 = base_accel_0  # lead car maintain normal behavior
        
         # restricted acceleration range
         max_accel = 3.0  # m/s²
         max_decel = 5.0  # m/s²
         
         accel_0 = max(min(accel_0, max_accel), -max_decel)
         accel_1 = max(min(accel_1, max_accel), -max_decel)
         accel_2 = max(min(accel_2, max_accel), -max_decel)
         
         # update the velocity
         v0 += accel_0 * self.dt
         v1 += accel_1 * self.dt
         v2 += accel_2 * self.dt
         
         # ensure minimum speed
         min_speed = 5.0  # m/s
         v0 = max(v0, min_speed)
         v1 = max(v1, min_speed)
         v2 = max(v2, min_speed)
         
         # update location
         self.x0 += v0 * self.dt
         self.y1 += v1 * self.dt
         self.y2 += v2 * self.dt
         
         # re-calculate the diastance
         self.x1 = self.x0 - self.y1
         self.x2 = self.y1 - self.y2
         
         # ensure minimum safe distance
         min_safe_distance = 5.0  # m
         if self.x1 < min_safe_distance:
             self.y1 = self.x0 - min_safe_distance
             self.x1 = min_safe_distance
             v1 = min(v1, v0)  # match the speed of the vehicle ahead to avoid collision
            
         if self.x2 < min_safe_distance:
             self.y2 = self.y1 - min_safe_distance
             self.x2 = min_safe_distance
             v2 = min(v2, v1)  # match the speed of the vehicle ahead to avoid collision
         
         # store current state
         self.history['x0'].append(self.x0)
         self.history['y1'].append(self.y1)
         self.history['y2'].append(self.y2)
         self.history['x1'].append(self.x1)
         self.history['x2'].append(self.x2)
         self.history['v0'].append(v0)
         self.history['v1'].append(v1)
         self.history['v2'].append(v2)
         self.history['lambda1'].append(lambda1)
         self.history['lambda2'].append(lambda2)
         self.history['mode'].append(mode)
     
     return self.history