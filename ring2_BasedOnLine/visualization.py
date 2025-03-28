import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np
import math
import os
import datetime

class CircularTrackVisualizer:
    def __init__(self, model):
        """
        Initialize the visualizer with a car following model
        
        Args:
        - model: Circular car following model instance
        """
        self.model = model
        
        # 创建输出文件夹
        self.output_folder = self._create_output_folder()
    
    def _create_output_folder(self):
        """创建输出文件夹"""
        # 获取当前时间作为文件夹名称一部分
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"simulation_output_{timestamp}"
        
        # 创建文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Create output folder: {output_folder}")
        
        return output_folder
    
    def plot_results(self, save=False, filename=None):
        """
        Plot comprehensive simulation results with four subplots
        
        Args:
        - save: 是否保存图片 (默认: False)
        - filename: 保存的文件名 (默认: None，会使用自动生成的名称)
        """
        fig, axs = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
        
        # Time vector for plotting
        time = self.model.history['time']
        
        # Plot positions
        axs[0].plot(time, self.model.history['x0'], 'r-', label='Lead Car (L)')
        axs[0].plot(time, self.model.history['y1'], 'g-', label='Following Car 1 (F1)')
        axs[0].plot(time, self.model.history['y2'], 'b-', label='Following Car 2 (F2)')
        axs[0].set_ylabel('Position [m]')
        axs[0].set_title('Vehicle Positions over Time')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot distances
        axs[1].plot(time, self.model.history['x1'], 'g-', label='Distance L-F1')
        axs[1].plot(time, self.model.history['x2'], 'b-', label='Distance F1-F2')
        axs[1].axhline(y=self.model.d, color='r', linestyle='--', label=f'Threshold (d={self.model.d}m)')
        axs[1].axhline(y=self.model.min_distance, color='orange', linestyle=':', label=f'Min Distance ({self.model.min_distance}m)')
        axs[1].axhline(y=self.model.max_distance, color='orange', linestyle=':', label=f'Max Distance ({self.model.max_distance}m)')
        axs[1].set_ylabel('Distance [m]')
        axs[1].set_title('Inter-vehicle Distances')
        axs[1].legend()
        axs[1].grid(True)
        # 设置距离图表的Y轴范围，使波动更加明显
        axs[1].set_ylim([max(0, self.model.min_distance - 10), self.model.max_distance + 10])
        
        # Plot velocities
        axs[2].plot(time, self.model.history['v0'], 'r-', label='Lead Car (L) Velocity')
        axs[2].plot(time, self.model.history['v1'], 'g-', label='Following Car 1 (F1) Velocity')
        axs[2].plot(time, self.model.history['v2'], 'b-', label='Following Car 2 (F2) Velocity')
        axs[2].set_ylabel('Velocity [m/s]')
        axs[2].set_title('Vehicle Velocities over Time')
        axs[2].legend()
        axs[2].grid(True)
        # 设置速度图表的Y轴范围，使波动更加明显
        v_min = min(min(self.model.history['v0']), min(self.model.history['v1']), min(self.model.history['v2']))
        v_max = max(max(self.model.history['v0']), max(self.model.history['v1']), max(self.model.history['v2']))
        v_padding = (v_max - v_min) * 0.1  # 10%的填充
        axs[2].set_ylim([max(0, v_min - v_padding), v_max + v_padding])
        
        # Plot operation mode
        mode_mapping = {'00': 0, '01': 1, '10': 2, '11': 3}
        mode_numeric = [mode_mapping[mode] for mode in self.model.history['mode']]
        
        axs[3].step(time, mode_numeric, 'k-')
        axs[3].set_yticks([0, 1, 2, 3])
        axs[3].set_yticklabels(['00', '01', '10', '11'])
        axs[3].set_ylabel('Mode (λ1λ2)')
        axs[3].set_xlabel('Time [s]')
        axs[3].set_title('System Operation Mode')
        axs[3].grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        if save:
            if filename is None:
                filename = 'simulation_results.png'
            save_path = os.path.join(self.output_folder, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"The picture was saved to: {save_path}")
        
        plt.show()
    
    def animate_vehicles(self, total_frames=500, save=False, filename=None, writer='pillow'):
        """
        Create a smooth animation of the vehicles on a circular track
        
        Args:
        - total_frames: Total number of frames for the animation
        - save: 是否保存动画 (默认: False)
        - filename: 保存的文件名 (默认: None，会使用自动生成的名称)
        - writer: 使用的动画编写器 ('pillow' 或 'ffmpeg'，默认: 'pillow')
        """
        # 轨道半径和圆心
        radius = 200  # 轨道半径
        center_x, center_y = 0, 0
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Circular Track: Three-Car Following Model')
        
        # 绘制圆形轨道
        track_circle = plt.Circle((center_x, center_y), radius, fill=False, color='gray', linestyle='--')
        ax.add_artist(track_circle)
        
        # 设置坐标轴
        ax.set_xlim(center_x - radius - 50, center_x + radius + 50)
        ax.set_ylim(center_y - radius - 50, center_y + radius + 50)
        ax.set_aspect('equal')
        
        # 车辆尺寸
        car_length = 15
        car_width = 8
        
        # 创建车辆
        lead_car = plt.Rectangle((0, 0), car_length, car_width, fc='r', ec='k', angle=0)
        f1_car = plt.Rectangle((0, 0), car_length, car_width, fc='g', ec='k', angle=0)
        f2_car = plt.Rectangle((0, 0), car_length, car_width, fc='b', ec='k', angle=0)
        
        # 添加车辆到图形
        ax.add_patch(lead_car)
        ax.add_patch(f1_car)
        ax.add_patch(f2_car)
        
        # 添加图例
        lead_rect = plt.Rectangle((0, 0), 1, 1, fc='r', ec='k', label='Lead Car (L)')
        f1_rect = plt.Rectangle((0, 0), 1, 1, fc='g', ec='k', label='Following Car 1 (F1)')
        f2_rect = plt.Rectangle((0, 0), 1, 1, fc='b', ec='k', label='Following Car 2 (F2)')
        ax.legend(handles=[lead_rect, f1_rect, f2_rect], loc='upper right')
        
        # 添加信息显示
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        mode_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        speed_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        
        # 添加距离显示
        distance_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)
        distance_status_text = ax.text(0.02, 0.75, '', transform=ax.transAxes)
        
        # 保存前一帧位置，用于连续性检查
        prev_positions = None
        
        # 添加显示当前参数的文本（理想跟随距离和允许范围）
        param_text = ax.text(0.02, 0.70, f'Ideal distance: {self.model.d:.1f}m\n'
                                         f'Allowed range: {self.model.min_distance:.1f}m - {self.model.max_distance:.1f}m',
                            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        # 添加说明文本，解释phantom jam现象
        info_text = ax.text(0.5, 0.02, 'Phantom Traffic Jam Simulation: Minor speed variations amplify down the chain',
                           transform=ax.transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.7))
        
        def convert_to_circular_coords(self, position):
            """
            将线性位置转换为圆形轨道上的坐标
            """
            # 计算角度（假设总轨道长度对应360度）
            angle = (position / self.model.track_length) * (2 * math.pi)
            x = center_x + radius * math.cos(angle - math.pi/2)
            y = center_y + radius * math.sin(angle - math.pi/2)
            return x, y, math.degrees(angle)
        
        def map_frame_to_time(frame):
            """将帧号映射到模拟时间"""
            start_time = self.model.history['time'][0]
            end_time = self.model.history['time'][-1]
            return start_time + (frame / total_frames) * (end_time - start_time)

        def interpolate_position(self, time):
            """
            根据给定时间插值位置，考虑轨道循环问题
            
            Returns:
            - [x0, y1, y2]的插值位置
            - [v0, v1, v2]的插值速度
            - [x1, x2]的插值距离
            """
            times = self.model.history['time']
            
            # 找到最近的两个时间点
            for i in range(len(times) - 1):
                if times[i] <= time <= times[i+1]:
                    # 线性插值
                    t = (time - times[i]) / (times[i+1] - times[i])
                    
                    # 处理位置的循环特性
                    x0_1 = self.model.history['x0'][i]
                    x0_2 = self.model.history['x0'][i+1]
                    y1_1 = self.model.history['y1'][i]
                    y1_2 = self.model.history['y1'][i+1]
                    y2_1 = self.model.history['y2'][i]
                    y2_2 = self.model.history['y2'][i+1]
                    
                    # 检查是否通过轨道循环点
                    if abs(x0_2 - x0_1) > self.model.track_length / 2:
                        # 调整位置以确保连续性
                        if x0_2 < x0_1:
                            x0_2 += self.model.track_length
                        else:
                            x0_1 += self.model.track_length
                    
                    # 为其他车辆执行相同的检查
                    if abs(y1_2 - y1_1) > self.model.track_length / 2:
                        if y1_2 < y1_1:
                            y1_2 += self.model.track_length
                        else:
                            y1_1 += self.model.track_length
                            
                    if abs(y2_2 - y2_1) > self.model.track_length / 2:
                        if y2_2 < y2_1:
                            y2_2 += self.model.track_length
                        else:
                            y2_1 += self.model.track_length
                    
                    # 线性插值
                    x0 = (1-t) * x0_1 + t * x0_2
                    y1 = (1-t) * y1_1 + t * y1_2
                    y2 = (1-t) * y2_1 + t * y2_2
                    
                    # 确保结果在轨道范围内
                    x0 %= self.model.track_length
                    y1 %= self.model.track_length
                    y2 %= self.model.track_length
                    
                    # 插值速度和距离
                    v0 = (1-t) * self.model.history['v0'][i] + t * self.model.history['v0'][i+1]
                    v1 = (1-t) * self.model.history['v1'][i] + t * self.model.history['v1'][i+1]
                    v2 = (1-t) * self.model.history['v2'][i] + t * self.model.history['v2'][i+1]
                    
                    # 插值距离
                    dist_L_F1 = (1-t) * self.model.history['x1'][i] + t * self.model.history['x1'][i+1]
                    dist_F1_F2 = (1-t) * self.model.history['x2'][i] + t * self.model.history['x2'][i+1]
                    
                    return [x0, y1, y2], [v0, v1, v2], [dist_L_F1, dist_F1_F2]
            
            # 如果时间超出范围，返回最后一个状态
            return (
                [self.model.history['x0'][-1], self.model.history['y1'][-1], self.model.history['y2'][-1]],
                [self.model.history['v0'][-1], self.model.history['v1'][-1], self.model.history['v2'][-1]],
                [self.model.history['x1'][-1], self.model.history['x2'][-1]]
            )

        def get_distance_status(distances):
            """获取距离状态描述（是否太近、太远或适中）"""
            status_L_F1 = "OK"
            if distances[0] < self.model.min_distance:
                status_L_F1 = "TOO CLOSE"
            elif distances[0] > self.model.max_distance:
                status_L_F1 = "TOO FAR"
                
            status_F1_F2 = "OK"
            if distances[1] < self.model.min_distance:
                status_F1_F2 = "TOO CLOSE"
            elif distances[1] > self.model.max_distance:
                status_F1_F2 = "TOO FAR"
                
            return status_L_F1, status_F1_F2

        def init():
            """初始化动画"""
            nonlocal prev_positions
            
            # 设置初始位置
            current_time = self.model.history['time'][0]
            initial_pos, initial_vel, initial_dist = interpolate_position(self, current_time)
            prev_positions = initial_pos
            
            lead_x, lead_y, lead_angle = convert_to_circular_coords(self, initial_pos[0])
            f1_x, f1_y, f1_angle = convert_to_circular_coords(self, initial_pos[1])
            f2_x, f2_y, f2_angle = convert_to_circular_coords(self, initial_pos[2])
            
            lead_car.set_xy((lead_x - car_length/2, lead_y - car_width/2))
            lead_car.set_angle(lead_angle)
            f1_car.set_xy((f1_x - car_length/2, f1_y - car_width/2))
            f1_car.set_angle(f1_angle)
            f2_car.set_xy((f2_x - car_length/2, f2_y - car_width/2))
            f2_car.set_angle(f2_angle)
            
            time_text.set_text('')
            mode_text.set_text('')
            speed_text.set_text('')
            distance_text.set_text('')
            distance_status_text.set_text('')
            
            return lead_car, f1_car, f2_car, time_text, mode_text, speed_text, distance_text, distance_status_text
        
        def animate(frame):
            """动画更新函数"""
            nonlocal prev_positions
            
            # 安全地将帧映射到时间
            current_time = map_frame_to_time(frame)
            
            # 安全地获取位置、速度和距离
            current_pos, current_vel, current_dist = interpolate_position(self, current_time)
            
            # 检查与上一帧的连续性 - 如果发生大跳变，尝试修复
            if prev_positions is not None:
                for i in range(3):
                    if abs(current_pos[i] - prev_positions[i]) > self.model.track_length / 2:
                        # 位置大幅跳变，很可能是通过轨道0点，调整为平滑过渡
                        if current_pos[i] < prev_positions[i]:
                            # 从大值到小值，这意味着穿过了0点
                            current_pos[i] += self.model.track_length
                        else:
                            # 从小值到大值，不太可能是自然运动
                            prev_positions[i] += self.model.track_length
            
            # 更新prev_positions用于下一帧
            prev_positions = current_pos.copy()
            
            # 转换位置到圆形坐标
            lead_x, lead_y, lead_angle = convert_to_circular_coords(self, current_pos[0] % self.model.track_length)
            f1_x, f1_y, f1_angle = convert_to_circular_coords(self, current_pos[1] % self.model.track_length)
            f2_x, f2_y, f2_angle = convert_to_circular_coords(self, current_pos[2] % self.model.track_length)
            
            # 更新车辆位置和角度
            lead_car.set_xy((lead_x - car_length/2, lead_y - car_width/2))
            lead_car.set_angle(lead_angle)
            f1_car.set_xy((f1_x - car_length/2, f1_y - car_width/2))
            f1_car.set_angle(f1_angle)
            f2_car.set_xy((f2_x - car_length/2, f2_y - car_width/2))
            f2_car.set_angle(f2_angle)
            
            # 获取距离状态
            status_L_F1, status_F1_F2 = get_distance_status(current_dist)
            
            # 更新时间文本
            time_text.set_text(f'Time: {current_time:.1f} s')
            
            # 更新速度文本
            speed_text.set_text(f'Speeds: L={current_vel[0]:.1f} m/s, F1={current_vel[1]:.1f} m/s, F2={current_vel[2]:.1f} m/s')
            
            # 更新距离文本
            distance_text.set_text(f'Distances: L-F1={current_dist[0]:.1f}m, F1-F2={current_dist[1]:.1f}m')
            
            # 更新距离状态文本，使用颜色编码
            distance_status_text.set_text(f'Status: L-F1: {status_L_F1}   F1-F2: {status_F1_F2}')
            if status_L_F1 == "TOO CLOSE" or status_F1_F2 == "TOO CLOSE":
                distance_status_text.set_color('red')
            elif status_L_F1 == "TOO FAR" or status_F1_F2 == "TOO FAR":
                distance_status_text.set_color('orange')
            else:
                distance_status_text.set_color('green')
            
            # 安全地获取模式
            mode_index = next((i for i, t in enumerate(self.model.history['time']) if t >= current_time), 
                              len(self.model.history['time'])-1)
            mode_text.set_text(f'Mode (λ1λ2): {self.model.history["mode"][mode_index]}')
            
            return lead_car, f1_car, f2_car, time_text, mode_text, speed_text, distance_text, distance_status_text

        # 创建动画，使用较短的间隔创建平滑动画
        ani = FuncAnimation(fig, animate, frames=total_frames, 
                          init_func=init, interval=50, blit=True)
        
        # 保存动画
        if save:
            if filename is None:
                # 根据使用的writer设置默认文件名和扩展名
                if writer == 'pillow':
                    filename = 'vehicle_animation.gif'
                else:  # ffmpeg
                    filename = 'vehicle_animation.mp4'
            
            # 获取完整保存路径
            save_path = os.path.join(self.output_folder, filename)
            
            print(f"Please wait while animation is being saved...")
            
            if writer == 'pillow':
                # 使用Pillow保存为GIF
                writer_obj = PillowWriter(fps=20)
                ani.save(save_path, writer=writer_obj)
            else:
                # 使用ffmpeg保存为MP4（需要安装ffmpeg）
                try:
                    writer_obj = FFMpegWriter(fps=20, metadata=dict(artist='Traffic Simulator'), bitrate=1800)
                    ani.save(save_path, writer=writer_obj)
                except Exception as e:
                    print(f"Failed to save MP4: {e}")
                    print("Try saving as a GIF...")
                    gif_path = os.path.join(self.output_folder, 'vehicle_animation.gif')
                    writer_obj = PillowWriter(fps=20)
                    ani.save(gif_path, writer=writer_obj)
                    save_path = gif_path
            
            print(f"Animation saved to: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return ani