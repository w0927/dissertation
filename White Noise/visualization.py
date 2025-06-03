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
        """绘制仿真结果 - 支持噪声对比显示"""
        
        # 检查是否有噪声功能
        has_noise = (hasattr(self.model, 'enable_L_noise') and 
                    self.model.enable_L_noise and 
                    len(self.model.history.get('L_noise', [])) > 0)
        
        # 根据是否有噪声决定子图数量
        num_plots = 5 if has_noise else 4
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, 4*num_plots), sharex=True)
        
        # 获取时间数据
        time = self.model.history['time']
        
        print(f"🔍 可视化调试信息:")
        print(f"   has_noise: {has_noise}")
        print(f"   子图数量: {num_plots}")
        if has_noise:
            print(f"   噪声开始时间: {getattr(self.model, 'noise_start_time', 'N/A')}")
            print(f"   噪声数据长度: {len(self.model.history.get('L_noise', []))}")
            print(f"   无噪声数据长度: {len(self.model.history.get('L_speed_without_noise', []))}")
        
        # 检查必要的历史数据键，如果不存在则使用替代键
        position_keys = {
            'L': ['x0', 'L_position'],
            'F1': ['y1', 'F1_position'], 
            'F2': ['y2', 'F2_position']
        }
        
        velocity_keys = {
            'L': ['v0', 'L_speed'],
            'F1': ['v1', 'F1_speed'],
            'F2': ['v2', 'F2_speed']
        }
        
        distance_keys = {
            'L_F1': ['x1', 'L_F1_distance'],
            'F1_F2': ['x2', 'F1_F2_distance']
        }
        
        def get_data(key_list):
            """从可能的键列表中获取数据"""
            for key in key_list:
                if key in self.model.history:
                    return self.model.history[key]
            return None
        
        # 1. 绘制位置
        L_pos = get_data(position_keys['L'])
        F1_pos = get_data(position_keys['F1'])
        F2_pos = get_data(position_keys['F2'])
        
        if L_pos is not None:
            axs[0].plot(time, L_pos, 'r-', label='Lead Car (L)')
        if F1_pos is not None:
            axs[0].plot(time, F1_pos, 'g-', label='Following Car 1 (F1)')
        if F2_pos is not None:
            axs[0].plot(time, F2_pos, 'b-', label='Following Car 2 (F2)')
        
        # 添加噪声开始标记
        if has_noise and hasattr(self.model, 'noise_start_time'):
            axs[0].axvline(x=self.model.noise_start_time, color='orange', linestyle='--', 
                          linewidth=2, label=f'噪声开始({self.model.noise_start_time}s)', alpha=0.8)
        
        axs[0].set_ylabel('Position [m]')
        axs[0].set_title('Vehicle Positions over Time')
        axs[0].legend()
        axs[0].grid(True)
        
        # 2. 绘制距离
        L_F1_dist = get_data(distance_keys['L_F1'])
        F1_F2_dist = get_data(distance_keys['F1_F2'])
        
        if L_F1_dist is not None:
            axs[1].plot(time, L_F1_dist, 'g-', label='Distance L-F1')
        if F1_F2_dist is not None:
            axs[1].plot(time, F1_F2_dist, 'b-', label='Distance F1-F2')
            
        axs[1].axhline(y=self.model.d, color='r', linestyle='--', label=f'Threshold (d={self.model.d}m)')
        
        # 安全地添加距离阈值线
        if hasattr(self.model, 'min_distance'):
            axs[1].axhline(y=self.model.min_distance, color='orange', linestyle=':', 
                          label=f'Min Distance ({self.model.min_distance}m)')
        if hasattr(self.model, 'max_distance'):
            axs[1].axhline(y=self.model.max_distance, color='orange', linestyle=':', 
                          label=f'Max Distance ({self.model.max_distance}m)')
        
        # 添加噪声开始标记
        if has_noise and hasattr(self.model, 'noise_start_time'):
            axs[1].axvline(x=self.model.noise_start_time, color='orange', linestyle='--', 
                          linewidth=2, alpha=0.8)
        
        axs[1].set_ylabel('Distance [m]')
        axs[1].set_title('Inter-vehicle Distances')
        axs[1].legend()
        axs[1].grid(True)
        
        # 自动调整Y轴范围以显示所有数据
        all_distances = []
        if L_F1_dist is not None:
            all_distances.extend(L_F1_dist)
        if F1_F2_dist is not None:
            all_distances.extend(F1_F2_dist)
            
        if all_distances:
            min_dist = min(all_distances)
            max_dist = max(all_distances)
            padding = (max_dist - min_dist) * 0.1
            axs[1].set_ylim([max(0, min_dist - padding), max_dist + padding])
        
        # 3. 绘制速度 - 重点改进的部分
        L_vel = get_data(velocity_keys['L'])
        F1_vel = get_data(velocity_keys['F1'])
        F2_vel = get_data(velocity_keys['F2'])
        
        if L_vel is not None:
            axs[2].plot(time, L_vel, 'r-', linewidth=2, 
                       label='Lead Car (L) - Real Velocity')
        if F1_vel is not None:
            axs[2].plot(time, F1_vel, 'g-', 
                       label='Following Car 1 (F1) Velocity')
        if F2_vel is not None:
            axs[2].plot(time, F2_vel, 'b-', 
                       label='Following Car 2 (F2) Velocity')
        
        # 如果有噪声数据，显示无噪声对比
        if has_noise:
            # 添加噪声开始标记
            if hasattr(self.model, 'noise_start_time'):
                axs[2].axvline(x=self.model.noise_start_time, color='orange', linestyle='--', 
                              linewidth=2, alpha=0.8, label='Noise Start')
            
            # 显示无噪声的L车速度
            if 'L_speed_without_noise' in self.model.history and len(self.model.history['L_speed_without_noise']) > 0:
                time_clean = time[:len(self.model.history['L_speed_without_noise'])]
                # 确保数据长度一致且有效
                if len(time_clean) > 0 and len(self.model.history['L_speed_without_noise']) > 0:
                    axs[2].plot(time_clean, self.model.history['L_speed_without_noise'], 'r--', 
                               linewidth=2, alpha=0.8, label='Lead Car (L) - Velocity without Noise')
                    print("✅ 添加无噪声L车速度对比线")
                else:
                    print("❌ 无噪声数据长度不匹配")
            else:
                print("❌ 无L_speed_without_noise数据，可能模型未正确记录")
                # 如果没有无噪声数据，就不显示对比图例
                
            # 添加说明文本
            if hasattr(self.model, 'noise_std') and L_vel is not None:
                axs[2].text(self.model.noise_start_time + 20 if hasattr(self.model, 'noise_start_time') else 50, 
                           max(L_vel) * 0.95,
                           f'Noise: σ={self.model.noise_std}m/s', 
                           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3),
                           fontsize=9)
        
        axs[2].set_ylabel('Velocity [m/s]')
        axs[2].set_title('Vehicle Velocities over Time' + 
                        (' (red solid line = with noise, red dot line = without noise)' if has_noise else ''))
        axs[2].legend()
        axs[2].grid(True)
        
        # 4. 绘制系统模式 - 修复 mode_mapping 问题
        # 定义模式映射
        mode_mapping = {
            '00': 0,
            '01': 1, 
            '10': 2,
            '11': 3
        }
        
        # 初始化 time_for_mode
        time_for_mode = time
        
        # 获取模式数据
        if 'mode' in self.model.history:
            modes = self.model.history['mode']
            
            # 确保数组长度匹配
            if len(modes) != len(time):
                min_length = min(len(modes), len(time))
                time_for_mode = time[:min_length]
                modes = modes[:min_length]
            else:
                time_for_mode = time
            
            # 转换模式为数值
            mode_numeric = [mode_mapping.get(str(mode), 0) for mode in modes]
            
            axs[3].step(time_for_mode, mode_numeric, 'k-', linewidth=2)
            axs[3].set_yticks([0, 1, 2, 3])
            axs[3].set_yticklabels(['00', '01', '10', '11'])
        else:
            # 如果没有模式数据，创建默认模式
            default_mode = np.zeros(len(time))
            axs[3].step(time, default_mode, 'k-', linewidth=2)
            axs[3].set_yticks([0, 1, 2, 3])
            axs[3].set_yticklabels(['00', '01', '10', '11'])
            axs[3].text(0.5, 0.5, 'Data is not available', transform=axs[3].transAxes, 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 添加噪声开始标记
        if has_noise and hasattr(self.model, 'noise_start_time'):
            axs[3].axvline(x=self.model.noise_start_time, color='orange', linestyle='--', 
                          linewidth=2, alpha=0.8)
        
        axs[3].set_ylabel('Mode (λ1λ2)')
        if not has_noise:
            axs[3].set_xlabel('Time [s]')
        axs[3].set_title('System Operation Mode')
        axs[3].grid(True)
        
        # 5. 绘制噪声图（如果有噪声）
        if has_noise:
            noise_data = self.model.history.get('L_noise', [])
            if len(noise_data) > 0:
                time_noise = time[:len(noise_data)]
                axs[4].plot(time_noise, noise_data, 'r-', alpha=0.8, linewidth=1, label='Noise Value')
                axs[4].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                # 添加标准差参考线
                if hasattr(self.model, 'noise_std'):
                    std_val = self.model.noise_std
                    axs[4].axhline(y=std_val, color='k', linestyle=':', alpha=0.5, 
                                  label=f'±σ ({std_val:.1f})')
                    axs[4].axhline(y=-std_val, color='k', linestyle=':', alpha=0.5)
                
                # 添加噪声开始标记
                if hasattr(self.model, 'noise_start_time'):
                    axs[4].axvline(x=self.model.noise_start_time, color='orange', linestyle='--', 
                                  linewidth=2, alpha=0.8, label=f'Noise Start')
                
                # 计算并显示噪声统计
                active_noise = [n for n in noise_data if n != 0]
                if active_noise:
                    actual_std = np.std(active_noise)
                    actual_mean = np.mean(active_noise)
                    axs[4].text(0.02, 0.95, 
                               f'Noise Statistics:\nMean: {actual_mean:.2f}\nVariance: {actual_std:.2f}',
                               transform=axs[4].transAxes, fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                               verticalalignment='top')
                    print(f"✅ 噪声统计: 均值={actual_mean:.2f}, 标准差={actual_std:.2f}")
                
                axs[4].set_ylabel('Noise [m/s]')
                axs[4].set_xlabel('Time [s]')
                axs[4].set_title(f'Noise Waveform of L' + 
                               (f' (σ={self.model.noise_std:.1f} m/s)' if hasattr(self.model, 'noise_std') else ''))
                axs[4].legend()
                axs[4].grid(True)
                
                print("✅ 成功绘制噪声图")
            else:
                # 无噪声数据的情况
                axs[4].text(0.5, 0.5, '无噪声数据', transform=axs[4].transAxes, 
                           fontsize=16, ha='center', va='center')
                axs[4].set_ylabel('Noise [m/s]')
                axs[4].set_xlabel('Time [s]')
                axs[4].set_title('噪声图 (无数据)')
                axs[4].grid(True)
                print("❌ 无噪声数据")
        
        plt.tight_layout()
        
        # 保存图片
        if save:
            if filename is None:
                filename = 'simulation_results.png'
            save_path = os.path.join(self.output_folder, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"The picture was saved to: {save_path}")
        
        plt.show()
        
        print("📈 可视化完成")
    
    def animate_vehicles(self, save=False, filename=None):
        """高连续性环形轨道动画"""
        print("正在生成高连续性环形轨道动画...")
        
        try:
            import math
            from matplotlib.animation import FuncAnimation, PillowWriter
            
            # 创建环形轨道动画
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # 轨道参数 - 调整半径使动画更慢
            radius = 150  # 从200减少到150，使轨道更紧凑
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
            
            # 检查并获取位置数据
            position_keys = {
                'L': ['L_position', 'x0'],
                'F1': ['F1_position', 'y1'], 
                'F2': ['F2_position', 'y2']
            }
            
            velocity_keys = {
                'L': ['L_speed', 'v0'],
                'F1': ['F1_speed', 'v1'],
                'F2': ['F2_speed', 'v2']
            }
            
            distance_keys = {
                'L_F1': ['L_F1_distance', 'x1'],
                'F1_F2': ['F1_F2_distance', 'x2']
            }
            
            def get_data(key_list):
                """从可能的键列表中获取数据"""
                for key in key_list:
                    if key in self.model.history:
                        return self.model.history[key]
                return None
            
            # 获取数据
            L_pos = get_data(position_keys['L'])
            F1_pos = get_data(position_keys['F1'])
            F2_pos = get_data(position_keys['F2'])
            L_vel = get_data(velocity_keys['L'])
            F1_vel = get_data(velocity_keys['F1'])
            F2_vel = get_data(velocity_keys['F2'])
            L_F1_dist = get_data(distance_keys['L_F1'])
            F1_F2_dist = get_data(distance_keys['F1_F2'])
            
            # 调试信息
            print(f"🎬 动画数据检查:")
            print(f"   L_pos: {'✅' if L_pos is not None else '❌'} ({len(L_pos) if L_pos else 0} 点)")
            print(f"   F1_pos: {'✅' if F1_pos is not None else '❌'} ({len(F1_pos) if F1_pos else 0} 点)")
            print(f"   F2_pos: {'✅' if F2_pos is not None else '❌'} ({len(F2_pos) if F2_pos else 0} 点)")
            print(f"   可用键: {list(self.model.history.keys())}")
            
            if L_pos is None or F1_pos is None or F2_pos is None:
                print("❌ 缺少位置数据，无法生成动画")
                print("   请检查模型是否正确保存了位置历史数据")
                return
            
            def position_to_coords(position):
                """将线性位置转换为圆形坐标"""
                # 使用正确的轨道长度属性
                track_length = getattr(self.model, 'L', getattr(self.model, 'track_length', 6000))
                angle = (position / track_length) * (2 * math.pi)
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
                        
                        # 获取轨道长度
                        track_length = getattr(self.model, 'L', getattr(self.model, 'track_length', 6000))
                        
                        # 插值位置（考虑环形特性）
                        def interpolate_position(pos1, pos2):
                            diff = pos2 - pos1
                            if abs(diff) > track_length / 2:
                                if diff > 0:
                                    pos1 += track_length
                                else:
                                    pos2 += track_length
                            result = (1-t) * pos1 + t * pos2
                            return result % track_length
                        
                        x0 = interpolate_position(L_pos[i], L_pos[i+1])
                        y1 = interpolate_position(F1_pos[i], F1_pos[i+1])
                        y2 = interpolate_position(F2_pos[i], F2_pos[i+1])
                        
                        # 插值速度
                        v0 = (1-t) * L_vel[i] + t * L_vel[i+1] if L_vel else 0
                        v1 = (1-t) * F1_vel[i] + t * F1_vel[i+1] if F1_vel else 0
                        v2 = (1-t) * F2_vel[i] + t * F2_vel[i+1] if F2_vel else 0
                        
                        # 插值距离
                        x1 = (1-t) * L_F1_dist[i] + t * L_F1_dist[i+1] if L_F1_dist else 0
                        x2 = (1-t) * F1_F2_dist[i] + t * F1_F2_dist[i+1] if F1_F2_dist else 0
                        
                        # 获取模式
                        mode = self.model.history.get('mode', ['00'] * len(times))[i]
                        
                        return x0, y1, y2, v0, v1, v2, x1, x2, mode
                
                # 如果超出范围，返回最后的值
                return (L_pos[-1], F1_pos[-1], F2_pos[-1], 
                       L_vel[-1] if L_vel else 0,
                       F1_vel[-1] if F1_vel else 0, 
                       F2_vel[-1] if F2_vel else 0,
                       L_F1_dist[-1] if L_F1_dist else 0, 
                       F1_F2_dist[-1] if F1_F2_dist else 0,
                       self.model.history.get('mode', ['00'])[-1])
            
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
            track_length = getattr(self.model, 'L', getattr(self.model, 'track_length', 6000))
            param_text = ax.text(0.02, 0.05, 
                               f'Target Distance: {self.model.d:.1f}m\n'
                               f'Track Length: {track_length:.0f}m', 
                               transform=ax.transAxes, fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            # 总帧数和时间范围 - 调整动画参数
            total_frames = 300  # 增加帧数
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
            
            # 创建动画 - 调整速度
            anim = FuncAnimation(fig, animate_frame, frames=total_frames, 
                               interval=80, blit=False, repeat=True)  # 增加间隔到80ms
            
            # 保存动画
            if save:
                try:
                    if filename is None:
                        filename = 'smooth_circular_animation.gif'
                    
                    save_path = os.path.join(self.output_folder, filename)
                    print(f"正在保存高质量动画到: {save_path}")
                    print("这可能需要几分钟时间...")
                    
                    writer = PillowWriter(fps=15)  # 降低FPS到15
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