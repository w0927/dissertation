import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

class CircularTrackVisualizer:
    def __init__(self, model):
        self.model = model
        self.output_folder = self._create_output_folder()
    
    def _create_output_folder(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"simulation_output_{timestamp}"
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Create output folder: {output_folder}")
        
        return output_folder
    
    def plot_results(self, save=False, filename=None):
        """绘制仿真结果的四个子图"""
        fig, axs = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
        
        # 获取时间数据
        time = self.model.history['time']
        
        # 1. 绘制位置
        axs[0].plot(time, self.model.history['x0'], 'r-', label='Lead Car (L)')
        axs[0].plot(time, self.model.history['y1'], 'g-', label='Following Car 1 (F1)')
        axs[0].plot(time, self.model.history['y2'], 'b-', label='Following Car 2 (F2)')
        axs[0].set_ylabel('Position [m]')
        axs[0].set_title('Vehicle Positions over Time')
        axs[0].legend()
        axs[0].grid(True)
        
        # 2. 绘制距离
        axs[1].plot(time, self.model.history['x1'], 'g-', label='Distance L-F1')
        axs[1].plot(time, self.model.history['x2'], 'b-', label='Distance F1-F2')
        axs[1].axhline(y=self.model.d, color='r', linestyle='--', label=f'Threshold (d={self.model.d}m)')
        
        # 安全地添加距离阈值线
        if hasattr(self.model, 'min_distance'):
            axs[1].axhline(y=self.model.min_distance, color='orange', linestyle=':', 
                          label=f'Min Distance ({self.model.min_distance}m)')
        if hasattr(self.model, 'max_distance'):
            axs[1].axhline(y=self.model.max_distance, color='orange', linestyle=':', 
                          label=f'Max Distance ({self.model.max_distance}m)')
        
        axs[1].set_ylabel('Distance [m]')
        axs[1].set_title('Inter-vehicle Distances')
        axs[1].legend()
        axs[1].grid(True)
        
        # 自动调整Y轴范围以显示所有数据
        all_distances = self.model.history['x1'] + self.model.history['x2']
        if all_distances:
            min_dist = min(all_distances)
            max_dist = max(all_distances)
            padding = (max_dist - min_dist) * 0.1
            axs[1].set_ylim([max(0, min_dist - padding), max_dist + padding])
        
        # 3. 绘制速度
        axs[2].plot(time, self.model.history['v0'], 'r-', label='Lead Car (L) Velocity')
        axs[2].plot(time, self.model.history['v1'], 'g-', label='Following Car 1 (F1) Velocity')
        axs[2].plot(time, self.model.history['v2'], 'b-', label='Following Car 2 (F2) Velocity')
        axs[2].set_ylabel('Velocity [m/s]')
        axs[2].set_title('Vehicle Velocities over Time')
        axs[2].legend()
        axs[2].grid(True)
        
        # 4. 绘制系统模式 - 修复数组长度问题
        mode_mapping = {'00': 0, '01': 1, '10': 2, '11': 3}
        modes = self.model.history['mode']
        
        # 确保数组长度匹配
        if len(modes) != len(time):
            min_length = min(len(modes), len(time))
            time_for_mode = time[:min_length]
            modes = modes[:min_length]
        else:
            time_for_mode = time
        
        mode_numeric = [mode_mapping.get(mode, 0) for mode in modes]
        
        axs[3].step(time_for_mode, mode_numeric, 'k-')
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
    
    def animate_vehicles(self, save=False, filename=None):
        """高连续性环形轨道动画"""
        print("正在生成高连续性环形轨道动画...")
        
        try:
            import math
            from matplotlib.animation import FuncAnimation, PillowWriter
            
            # 创建环形轨道动画
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # 轨道参数
            radius = 200
            center_x, center_y = 0, 0
            
            # 绘制环形轨道
            track_circle = plt.Circle((center_x, center_y), radius, 
                                    fill=False, color='gray', linestyle='--', linewidth=3)
            ax.add_artist(track_circle)
            
            # 设置坐标轴
            ax.set_xlim(center_x - radius - 50, center_x + radius + 50)
            ax.set_ylim(center_y - radius - 50, center_y + radius + 50)
            ax.set_aspect('equal')
            ax.set_title('Circular Track: Lambda-based Car Following Model')
            
            # 移除坐标轴刻度让视觉更清晰
            ax.set_xticks([])
            ax.set_yticks([])
            
            def position_to_coords(position):
                """将线性位置转换为圆形坐标"""
                angle = (position / self.model.track_length) * (2 * math.pi)
                x = center_x + radius * math.cos(angle - math.pi/2)
                y = center_y + radius * math.sin(angle - math.pi/2)
                return x, y
            
            def interpolate_data(time_target):
                """对给定时间进行平滑插值"""
                times = self.model.history['time']
                
                # 找到最近的两个时间点
                for i in range(len(times) - 1):
                    if times[i] <= time_target <= times[i+1]:
                        # 线性插值因子
                        t = (time_target - times[i]) / (times[i+1] - times[i])
                        
                        # 插值位置（考虑环形特性）
                        def interpolate_position(pos1, pos2):
                            diff = pos2 - pos1
                            if abs(diff) > self.model.track_length / 2:
                                if diff > 0:
                                    pos1 += self.model.track_length
                                else:
                                    pos2 += self.model.track_length
                            result = (1-t) * pos1 + t * pos2
                            return result % self.model.track_length
                        
                        x0 = interpolate_position(self.model.history['x0'][i], self.model.history['x0'][i+1])
                        y1 = interpolate_position(self.model.history['y1'][i], self.model.history['y1'][i+1])
                        y2 = interpolate_position(self.model.history['y2'][i], self.model.history['y2'][i+1])
                        
                        # 插值速度
                        v0 = (1-t) * self.model.history['v0'][i] + t * self.model.history['v0'][i+1]
                        v1 = (1-t) * self.model.history['v1'][i] + t * self.model.history['v1'][i+1]
                        v2 = (1-t) * self.model.history['v2'][i] + t * self.model.history['v2'][i+1]
                        
                        # 插值距离
                        x1 = (1-t) * self.model.history['x1'][i] + t * self.model.history['x1'][i+1]
                        x2 = (1-t) * self.model.history['x2'][i] + t * self.model.history['x2'][i+1]
                        
                        # 获取模式
                        mode = self.model.history['mode'][i]
                        
                        return x0, y1, y2, v0, v1, v2, x1, x2, mode
                
                # 如果超出范围，返回最后的值
                return (self.model.history['x0'][-1], self.model.history['y1'][-1], 
                       self.model.history['y2'][-1], self.model.history['v0'][-1],
                       self.model.history['v1'][-1], self.model.history['v2'][-1],
                       self.model.history['x1'][-1], self.model.history['x2'][-1],
                       self.model.history['mode'][-1])
            
            # 创建车辆图形对象
            lead_car = ax.scatter([], [], s=400, c='red', marker='o', 
                                label='Lead Car (L)', edgecolor='darkred', linewidth=2, zorder=5)
            f1_car = ax.scatter([], [], s=400, c='green', marker='o', 
                              label='Following Car 1 (F1)', edgecolor='darkgreen', linewidth=2, zorder=5)
            f2_car = ax.scatter([], [], s=400, c='blue', marker='o', 
                              label='Following Car 2 (F2)', edgecolor='darkblue', linewidth=2, zorder=5)
            
            # 添加图例
            ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
            
            # 添加信息文本
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            speed_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            distance_text = ax.text(0.02, 0.75, '', transform=ax.transAxes, fontsize=10,
                                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            mode_text = ax.text(0.02, 0.65, '', transform=ax.transAxes, fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            # 添加参数信息
            param_text = ax.text(0.02, 0.05, 
                               f'Target Distance: {self.model.d:.1f}m\n'
                               f'Track Length: {self.model.track_length:.0f}m', 
                               transform=ax.transAxes, fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            # 总帧数和时间范围
            total_frames = 200  # 更多帧数提高连续性
            start_time = self.model.history['time'][0]
            end_time = self.model.history['time'][-1]
            
            def animate_frame(frame):
                """动画帧更新函数"""
                # 计算当前时间
                current_time = start_time + (frame / total_frames) * (end_time - start_time)
                
                # 获取插值数据
                x0, y1, y2, v0, v1, v2, x1, x2, mode = interpolate_data(current_time)
                
                # 转换为圆形坐标
                lead_x, lead_y = position_to_coords(x0)
                f1_x, f1_y = position_to_coords(y1)
                f2_x, f2_y = position_to_coords(y2)
                
                # 更新车辆位置
                lead_car.set_offsets([[lead_x, lead_y]])
                f1_car.set_offsets([[f1_x, f1_y]])
                f2_car.set_offsets([[f2_x, f2_y]])
                
                # 更新文本信息
                time_text.set_text(f'Time: {current_time:.1f}s')
                speed_text.set_text(f'Speeds:\nL: {v0:.1f} m/s\nF1: {v1:.1f} m/s\nF2: {v2:.1f} m/s')
                distance_text.set_text(f'Distances:\nL-F1: {x1:.1f}m\nF1-F2: {x2:.1f}m')
                mode_text.set_text(f'Mode (λ1λ2): {mode}')
                
                return lead_car, f1_car, f2_car, time_text, speed_text, distance_text, mode_text
            
            # 创建动画
            anim = FuncAnimation(fig, animate_frame, frames=total_frames, 
                               interval=50, blit=False, repeat=True)
            
            # 保存动画
            if save:
                try:
                    if filename is None:
                        filename = 'smooth_circular_animation.gif'
                    
                    save_path = os.path.join(self.output_folder, filename)
                    print(f"正在保存高质量动画到: {save_path}")
                    print("这可能需要几分钟时间...")
                    
                    writer = PillowWriter(fps=20)
                    anim.save(save_path, writer=writer, dpi=150)
                    print(f"动画已保存到: {save_path}")
                    
                except Exception as e:
                    print(f"动画保存失败: {e}")
            
            plt.tight_layout()
            plt.show()
            print("高连续性环形轨道动画完成")
            
        except Exception as e:
            print(f"环形轨道动画生成失败: {e}")
            print("跳过动画，继续执行...")