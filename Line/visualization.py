import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def plot_results(model):
    """
    Plot the simulation results.
    """
    # Set up the figure
    fig, axs = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
    
    # Time vector for plotting
    time = model.history['time']
    
    # Plot positions
    axs[0].plot(time, model.history['x0'], 'r-', label='Lead Car (L)')
    axs[0].plot(time, model.history['y1'], 'g-', label='Following Car 1 (F1)')
    axs[0].plot(time, model.history['y2'], 'b-', label='Following Car 2 (F2)')
    axs[0].set_ylabel('Position [m]')
    axs[0].set_title('Vehicle Positions over Time')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot distances
    axs[1].plot(time, model.history['x1'], 'g-', label='Distance L-F1')
    axs[1].plot(time, model.history['x2'], 'b-', label='Distance F1-F2')
    axs[1].axhline(y=model.d, color='r', linestyle='--', label=f'Threshold (d={model.d}m)')
    axs[1].set_ylabel('Distance [m]')
    axs[1].set_title('Inter-vehicle Distances')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot velocities
    axs[2].plot(time, model.history['v0'], 'r-', label='Lead Car (L) Velocity')
    axs[2].plot(time, model.history['v1'], 'g-', label='Following Car 1 (F1) Velocity')
    axs[2].plot(time, model.history['v2'], 'b-', label='Following Car 2 (F2) Velocity')
    axs[2].set_ylabel('Velocity [m/s]')
    axs[2].set_title('Vehicle Velocities over Time')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot operation mode
    mode_mapping = {'00': 0, '01': 1, '10': 2, '11': 3}
    mode_numeric = [mode_mapping[mode] for mode in model.history['mode']]
    
    axs[3].step(time, mode_numeric, 'k-')
    axs[3].set_yticks([0, 1, 2, 3])
    axs[3].set_yticklabels(['00', '01', '10', '11'])
    axs[3].set_ylabel('Mode (λ1λ2)')
    axs[3].set_xlabel('Time [s]')
    axs[3].set_title('System Operation Mode')
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.show()

def animate_vehicles(model):
    """
    Create an animation of the vehicles.
    """
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
    
    lead_car = plt.Rectangle((model.history['x0'][0], 0), car_length, car_height, fc='r', ec='k')
    f1_car = plt.Rectangle((model.history['y1'][0], 0), car_length, car_height, fc='g', ec='k')
    f2_car = plt.Rectangle((model.history['y2'][0], 0), car_length, car_height, fc='b', ec='k')
    
    # Add cars to the plot
    ax.add_patch(lead_car)
    ax.add_patch(f1_car)
    ax.add_patch(f2_car)
    
    # Add legend
    lead_rect = plt.Rectangle((0, 0), 1, 1, fc='r', ec='k', label='Lead Car (L)')
    f1_rect = plt.Rectangle((0, 0), 1, 1, fc='g', ec='k', label='Following Car 1 (F1)')
    f2_rect = plt.Rectangle((0, 0), 1, 1, fc='b', ec='k', label='Following Car 2 (F2)')
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
    plt.show()
    
    return ani