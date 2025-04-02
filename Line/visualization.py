import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os

def plot_results(model, save_path=None, colors=None):
    """
    Plot the simulation results with customized colors.
    Each subplot is created in a separate figure.
    
    Parameters:
    -----------
    model: ThreeCarFollowingModel
        The model containing simulation results
    save_path: str, optional
        If provided, figures will be saved to this directory
    colors: dict, optional
        Dictionary with color specifications for different elements
        If None, default colors will be used
    """
    # Default colors will be used from matplotlib
    
    # Time vector for plotting
    time = model.history['time']
    
    # 1. Plot positions
    fig1 = plt.figure(figsize=(12, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(time, model.history['x0'], linestyle='-', label='Lead Car (L)')
    ax1.plot(time, model.history['y1'], linestyle='-', label='Following Car 1 (F1)')
    ax1.plot(time, model.history['y2'], linestyle='-', label='Following Car 2 (F2)')
    ax1.set_ylabel('Position [m]')
    ax1.set_xlabel('Time [s]')
    ax1.set_title('Vehicle Positions over Time')
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    
    if save_path:
        pos_save_path = os.path.join(save_path, 'positions.png')
        os.makedirs(os.path.dirname(pos_save_path), exist_ok=True)
        plt.savefig(pos_save_path, dpi=300, bbox_inches='tight')
        print(f"Position plot saved to {pos_save_path}")
    
    plt.show()
    
    # 2. Plot distances
    fig2 = plt.figure(figsize=(12, 6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(time, model.history['x1'], linestyle='-', label='Distance L-F1')
    ax2.plot(time, model.history['x2'], linestyle='-', label='Distance F1-F2')
    ax2.axhline(y=model.d, linestyle='--', label=f'Threshold (d={model.d}m)')
    ax2.set_ylabel('Distance [m]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title('Inter-vehicle Distances')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    
    if save_path:
        dist_save_path = os.path.join(save_path, 'distances.png')
        os.makedirs(os.path.dirname(dist_save_path), exist_ok=True)
        plt.savefig(dist_save_path, dpi=300, bbox_inches='tight')
        print(f"Distance plot saved to {dist_save_path}")
    
    plt.show()
    
    # 3. Plot velocities
    fig3 = plt.figure(figsize=(12, 6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(time, model.history['v0'], linestyle='-', label='Lead Car (L) Velocity')
    ax3.plot(time, model.history['v1'], linestyle='-', label='Following Car 1 (F1) Velocity')
    ax3.plot(time, model.history['v2'], linestyle='-', label='Following Car 2 (F2) Velocity')
    ax3.set_ylabel('Velocity [m/s]')
    ax3.set_xlabel('Time [s]')
    ax3.set_title('Vehicle Velocities over Time')
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    
    if save_path:
        vel_save_path = os.path.join(save_path, 'velocities.png')
        os.makedirs(os.path.dirname(vel_save_path), exist_ok=True)
        plt.savefig(vel_save_path, dpi=300, bbox_inches='tight')
        print(f"Velocity plot saved to {vel_save_path}")
    
    plt.show()
    
    # 4. Plot operation mode
    fig4 = plt.figure(figsize=(12, 6))
    ax4 = fig4.add_subplot(111)
    mode_mapping = {'00': 0, '01': 1, '10': 2, '11': 3}
    mode_numeric = [mode_mapping[mode] for mode in model.history['mode']]
    
    ax4.step(time, mode_numeric, '-')
    ax4.set_yticks([0, 1, 2, 3])
    ax4.set_yticklabels(['00', '01', '10', '11'])
    ax4.set_ylabel('Mode (λ1λ2)')
    ax4.set_xlabel('Time [s]')
    ax4.set_title('System Operation Mode')
    ax4.grid(True)
    plt.tight_layout()
    
    if save_path:
        mode_save_path = os.path.join(save_path, 'operation_mode.png')
        os.makedirs(os.path.dirname(mode_save_path), exist_ok=True)
        plt.savefig(mode_save_path, dpi=300, bbox_inches='tight')
        print(f"Operation mode plot saved to {mode_save_path}")
    
    plt.show()


def animate_vehicles(model, save_path=None, fps=10, colors=None):
    """
    Create an animation of the vehicles.
    
    Parameters:
    -----------
    model: ThreeCarFollowingModel
        The model containing simulation results
    save_path: str, optional
        If provided, the animation will be saved to this path
    fps: int, optional
        Frames per second for the saved animation (default: 10)
    colors: dict, optional
        Dictionary with color specifications for different elements
        If None, default colors will be used
    
    Returns:
    --------
    ani: FuncAnimation
        The animation object
    """
    # Using default Matplotlib colors
    
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Set axis limits with some margin
    min_pos = min(min(model.history['y2']), min(model.history['y1']), min(model.history['x0']))
    max_pos = max(max(model.history['y2']), max(model.history['y1']), max(model.history['x0']))
    margin = (max_pos - min_pos) * 0.1
    
    ax.set_xlim(min_pos - margin, max_pos + margin)
    ax.set_ylim(-2, 3)
    ax.set_yticks([])
    ax.set_xlabel('Position [m]')
    ax.set_title('Three-Car Following Model Animation')
    
    # Create vehicle objects (rectangles)
    car_length = 15  # Increased size for better visibility
    car_height = 5
    
    # Using default colors for cars
    lead_car = plt.Rectangle((model.history['x0'][0], 0), car_length, car_height, fc='C0', ec='black')
    f1_car = plt.Rectangle((model.history['y1'][0], 0), car_length, car_height, fc='C1', ec='black')
    f2_car = plt.Rectangle((model.history['y2'][0], 0), car_length, car_height, fc='C2', ec='black')
    
    # Add cars to the plot
    ax.add_patch(lead_car)
    ax.add_patch(f1_car)
    ax.add_patch(f2_car)
    
    # Add legend
    lead_rect = plt.Rectangle((0, 0), 1, 1, fc='C0', ec='black', label='Lead Car (L)')
    f1_rect = plt.Rectangle((0, 0), 1, 1, fc='C1', ec='black', label='Following Car 1 (F1)')
    f2_rect = plt.Rectangle((0, 0), 1, 1, fc='C2', ec='black', label='Following Car 2 (F2)')
    ax.legend(handles=[lead_rect, f1_rect, f2_rect], loc='upper right')
    
    # Add time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    # Add mode text
    mode_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    
    def init():
        """Initialize animation."""
        lead_car.set_xy((model.history['x0'][0], 0))
        f1_car.set_xy((model.history['y1'][0], 0))
        f2_car.set_xy((model.history['y2'][0], 0))
        time_text.set_text('')
        mode_text.set_text('')
        return lead_car, f1_car, f2_car, time_text, mode_text
    
    def animate(i):
        """Animate function."""
        lead_car.set_xy((model.history['x0'][i], 0))
        f1_car.set_xy((model.history['y1'][i], 0))
        f2_car.set_xy((model.history['y2'][i], 0))
        time_text.set_text(f'Time: {model.history["time"][i]:.1f} s')
        mode_text.set_text(f'Mode (λ1λ2): {model.history["mode"][i]}')
        return lead_car, f1_car, f2_car, time_text, mode_text
    
    ani = FuncAnimation(fig, animate, frames=len(model.history['time']), 
                      init_func=init, interval=100, blit=True)
    
    plt.tight_layout()
    
    # Save the animation if a save path is provided
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Determine the file format from the extension
            file_ext = os.path.splitext(save_path)[1].lower()
            
            if file_ext == '.mp4':
                try:
                    # Check if ffmpeg is available
                    import matplotlib.animation as animation
                    if 'ffmpeg' in animation.writers.list():
                        print("Using ffmpeg writer for MP4...")
                        writer = animation.FFMpegWriter(fps=fps)
                        ani.save(save_path, writer=writer, dpi=300)
                    else:
                        print("FFmpeg writer not available. Saving as GIF instead.")
                        gif_path = os.path.splitext(save_path)[0] + '.gif'
                        ani.save(gif_path, writer='pillow', fps=fps)
                except Exception as e:
                    print(f"Error saving MP4: {e}")
                    print("Trying to save as GIF instead...")
                    gif_path = os.path.splitext(save_path)[0] + '.gif'
                    ani.save(gif_path, writer='pillow', fps=fps)
            elif file_ext == '.gif':
                print("Saving as GIF...")
                ani.save(save_path, writer='pillow', fps=fps)
            else:
                print(f"Unsupported file extension: {file_ext}. Using GIF instead.")
                gif_path = os.path.splitext(save_path)[0] + '.gif'
                ani.save(gif_path, writer='pillow', fps=fps)
                
            print(f"Animation saved successfully.")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Saving individual frames instead...")
            
            # Fallback: save frames as PNG files
            frames_dir = os.path.join(os.path.dirname(save_path), "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Save key frames (approximately 30 frames total)
            step = max(1, len(model.history['time']) // 30)
            for i in range(0, len(model.history['time']), step):
                # Create a new figure for each frame
                frame_fig, frame_ax = plt.subplots(figsize=(12, 3))
                
                # Set up the axes
                frame_ax.set_xlim(ax.get_xlim())
                frame_ax.set_ylim(ax.get_ylim())
                frame_ax.set_yticks([])
                frame_ax.set_xlabel('Position [m]')
                frame_ax.set_title(f'Three-Car Following Model at t={model.history["time"][i]:.1f}s')
                
                # Draw cars
                lead_car_pos = plt.Rectangle((model.history['x0'][i], 0), car_length, car_height, 
                                          fc='C0', ec='black')
                f1_car_pos = plt.Rectangle((model.history['y1'][i], 0), car_length, car_height, 
                                         fc='C1', ec='black')
                f2_car_pos = plt.Rectangle((model.history['y2'][i], 0), car_length, car_height, 
                                         fc='C2', ec='black')
                
                frame_ax.add_patch(lead_car_pos)
                frame_ax.add_patch(f1_car_pos)
                frame_ax.add_patch(f2_car_pos)
                
                # Add legend
                frame_lead_rect = plt.Rectangle((0, 0), 1, 1, fc='C0', ec='black', label='Lead Car (L)')
                frame_f1_rect = plt.Rectangle((0, 0), 1, 1, fc='C1', ec='black', label='Following Car 1 (F1)')
                frame_f2_rect = plt.Rectangle((0, 0), 1, 1, fc='C2', ec='black', label='Following Car 2 (F2)')
                frame_ax.legend(handles=[frame_lead_rect, frame_f1_rect, frame_f2_rect], loc='upper right')
                
                # Add time and mode text
                frame_ax.text(0.02, 0.95, f'Time: {model.history["time"][i]:.1f} s', transform=frame_ax.transAxes)
                frame_ax.text(0.02, 0.90, f'Mode (λ1λ2): {model.history["mode"][i]}', transform=frame_ax.transAxes)
                
                # Save frame
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                frame_fig.savefig(frame_path, dpi=300, bbox_inches='tight')
                plt.close(frame_fig)
            
            print(f"Frames saved to {frames_dir}")
    
    plt.show()
    
    return ani