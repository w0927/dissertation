import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os

def plot_results(model, save_path=None, colors=None):
    """
    Plot the simulation results with customized colors.
    
    Parameters:
    -----------
    model: ThreeCarFollowingModel
        The model containing simulation results
    save_path: str, optional
        If provided, the figure will be saved to this path
    colors: dict, optional
        Dictionary with color specifications for different elements:
        {
            'lead_car': 'color for lead car',
            'following_car1': 'color for following car 1',
            'following_car2': 'color for following car 2',
            'threshold_line': 'color for threshold line',
            'background': 'color for plot background',
            'grid': 'color for grid lines'
        }
    """
    # Default colors
    default_colors = {
        'lead_car': 'red',
        'following_car1': 'green',
        'following_car2': 'blue',
        'threshold_line': 'red',
        'background': 'white',
        'grid': 'lightgray'
    }
    
    # Use custom colors if provided, otherwise use defaults
    if colors is None:
        colors = default_colors
    else:
        # Merge with defaults for any missing colors
        for key, value in default_colors.items():
            if key not in colors:
                colors[key] = value
    
    # Set up the figure
    fig, axs = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
    fig.patch.set_facecolor(colors['background'])
    
    # Apply background color to all subplots
    for ax in axs:
        ax.set_facecolor(colors['background'])
    
    # Time vector for plotting
    time = model.history['time']
    
    # Plot positions
    axs[0].plot(time, model.history['x0'], color=colors['lead_car'], linestyle='-', label='Lead Car (L)')
    axs[0].plot(time, model.history['y1'], color=colors['following_car1'], linestyle='-', label='Following Car 1 (F1)')
    axs[0].plot(time, model.history['y2'], color=colors['following_car2'], linestyle='-', label='Following Car 2 (F2)')
    axs[0].set_ylabel('Position [m]')
    axs[0].set_title('Vehicle Positions over Time')
    axs[0].legend()
    axs[0].grid(True, color=colors['grid'])
    
    # Plot distances
    axs[1].plot(time, model.history['x1'], color=colors['following_car1'], linestyle='-', label='Distance L-F1')
    axs[1].plot(time, model.history['x2'], color=colors['following_car2'], linestyle='-', label='Distance F1-F2')
    axs[1].axhline(y=model.d, color=colors['threshold_line'], linestyle='--', label=f'Threshold (d={model.d}m)')
    axs[1].set_ylabel('Distance [m]')
    axs[1].set_title('Inter-vehicle Distances')
    axs[1].legend()
    axs[1].grid(True, color=colors['grid'])
    
    # Plot velocities
    axs[2].plot(time, model.history['v0'], color=colors['lead_car'], linestyle='-', label='Lead Car (L) Velocity')
    axs[2].plot(time, model.history['v1'], color=colors['following_car1'], linestyle='-', label='Following Car 1 (F1) Velocity')
    axs[2].plot(time, model.history['v2'], color=colors['following_car2'], linestyle='-', label='Following Car 2 (F2) Velocity')
    axs[2].set_ylabel('Velocity [m/s]')
    axs[2].set_title('Vehicle Velocities over Time')
    axs[2].legend()
    axs[2].grid(True, color=colors['grid'])
    
    # Plot operation mode
    mode_mapping = {'00': 0, '01': 1, '10': 2, '11': 3}
    mode_numeric = [mode_mapping[mode] for mode in model.history['mode']]
    
    axs[3].step(time, mode_numeric, 'k-')
    axs[3].set_yticks([0, 1, 2, 3])
    axs[3].set_yticklabels(['00', '01', '10', '11'])
    axs[3].set_ylabel('Mode (λ1λ2)')
    axs[3].set_xlabel('Time [s]')
    axs[3].set_title('System Operation Mode')
    axs[3].grid(True, color=colors['grid'])
    
    plt.tight_layout()
    
    # Save the figure if a save path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Figure saved to {save_path}")
    
    plt.show()

def animate_vehicles(model, save_path=None, fps=10, colors=None):
    """
    Create an animation of the vehicles with customized colors.
    
    Parameters:
    -----------
    model: ThreeCarFollowingModel
        The model containing simulation results
    save_path: str, optional
        If provided, the animation will be saved to this path
    fps: int, optional
        Frames per second for the saved animation (default: 10)
    colors: dict, optional
        Dictionary with color specifications for different elements:
        {
            'lead_car': 'color for lead car',
            'following_car1': 'color for following car 1',
            'following_car2': 'color for following car 2',
            'car_edge': 'color for car edges',
            'background': 'color for animation background',
            'text': 'color for text elements'
        }
    
    Returns:
    --------
    ani: FuncAnimation
        The animation object
    """
    # Default colors
    default_colors = {
        'lead_car': 'red',
        'following_car1': 'green',
        'following_car2': 'blue',
        'car_edge': 'black',
        'background': 'white',
        'text': 'black'
    }
    
    # Use custom colors if provided, otherwise use defaults
    if colors is None:
        colors = default_colors
    else:
        # Merge with defaults for any missing colors
        for key, value in default_colors.items():
            if key not in colors:
                colors[key] = value
    
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor(colors['background'])
    ax.set_facecolor(colors['background'])
    
    # Set axis limits with some margin
    min_pos = min(min(model.history['y2']), min(model.history['y1']), min(model.history['x0']))
    max_pos = max(max(model.history['y2']), max(model.history['y1']), max(model.history['x0']))
    margin = (max_pos - min_pos) * 0.1
    
    ax.set_xlim(min_pos - margin, max_pos + margin)
    ax.set_ylim(-2, 3)
    ax.set_yticks([])
    ax.set_xlabel('Position [m]', color=colors['text'])
    ax.set_title('Three-Car Following Model Animation', color=colors['text'])
    ax.tick_params(axis='x', colors=colors['text'])
    
    # Create vehicle objects (rectangles)
    car_length = 15  # Increased size for better visibility
    car_height = 5
    
    lead_car = plt.Rectangle((model.history['x0'][0], 0), car_length, car_height, 
                           fc=colors['lead_car'], ec=colors['car_edge'])
    f1_car = plt.Rectangle((model.history['y1'][0], 0), car_length, car_height, 
                          fc=colors['following_car1'], ec=colors['car_edge'])
    f2_car = plt.Rectangle((model.history['y2'][0], 0), car_length, car_height, 
                          fc=colors['following_car2'], ec=colors['car_edge'])
    
    # Add cars to the plot
    ax.add_patch(lead_car)
    ax.add_patch(f1_car)
    ax.add_patch(f2_car)
    
    # Add legend
    lead_rect = plt.Rectangle((0, 0), 1, 1, fc=colors['lead_car'], ec=colors['car_edge'], label='Lead Car (L)')
    f1_rect = plt.Rectangle((0, 0), 1, 1, fc=colors['following_car1'], ec=colors['car_edge'], label='Following Car 1 (F1)')
    f2_rect = plt.Rectangle((0, 0), 1, 1, fc=colors['following_car2'], ec=colors['car_edge'], label='Following Car 2 (F2)')
    ax.legend(handles=[lead_rect, f1_rect, f2_rect], loc='upper right')
    
    # Add time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color=colors['text'])
    
    # Add mode text
    mode_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, color=colors['text'])
    
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
                frame_fig.patch.set_facecolor(colors['background'])
                frame_ax.set_facecolor(colors['background'])
                
                # Set up the axes
                frame_ax.set_xlim(ax.get_xlim())
                frame_ax.set_ylim(ax.get_ylim())
                frame_ax.set_yticks([])
                frame_ax.set_xlabel('Position [m]', color=colors['text'])
                frame_ax.set_title(f'Three-Car Following Model at t={model.history["time"][i]:.1f}s', color=colors['text'])
                frame_ax.tick_params(axis='x', colors=colors['text'])
                
                # Draw cars
                lead_car_pos = plt.Rectangle((model.history['x0'][i], 0), car_length, car_height, 
                                           fc=colors['lead_car'], ec=colors['car_edge'])
                f1_car_pos = plt.Rectangle((model.history['y1'][i], 0), car_length, car_height, 
                                          fc=colors['following_car1'], ec=colors['car_edge'])
                f2_car_pos = plt.Rectangle((model.history['y2'][i], 0), car_length, car_height, 
                                          fc=colors['following_car2'], ec=colors['car_edge'])
                
                frame_ax.add_patch(lead_car_pos)
                frame_ax.add_patch(f1_car_pos)
                frame_ax.add_patch(f2_car_pos)
                
                # Add legend
                frame_lead_rect = plt.Rectangle((0, 0), 1, 1, fc=colors['lead_car'], ec=colors['car_edge'], label='Lead Car (L)')
                frame_f1_rect = plt.Rectangle((0, 0), 1, 1, fc=colors['following_car1'], ec=colors['car_edge'], label='Following Car 1 (F1)')
                frame_f2_rect = plt.Rectangle((0, 0), 1, 1, fc=colors['following_car2'], ec=colors['car_edge'], label='Following Car 2 (F2)')
                frame_ax.legend(handles=[frame_lead_rect, frame_f1_rect, frame_f2_rect], loc='upper right')
                
                # Add time and mode text
                frame_ax.text(0.02, 0.95, f'Time: {model.history["time"][i]:.1f} s', transform=frame_ax.transAxes, color=colors['text'])
                frame_ax.text(0.02, 0.90, f'Mode (λ1λ2): {model.history["mode"][i]}', transform=frame_ax.transAxes, color=colors['text'])
                
                # Save frame
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                frame_fig.savefig(frame_path, dpi=300, bbox_inches='tight', facecolor=frame_fig.get_facecolor())
                plt.close(frame_fig)
            
            print(f"Frames saved to {frames_dir}")
    
    plt.show()
    
    return ani