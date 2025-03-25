import numpy as np

def analyze_stability(model):
    """
    Analyze the stability of the car-following behavior.
    """
    # Calculate statistics for the steady-state portion (last 2/3 of simulation)
    steady_idx = len(model.history['time']) // 3
    
    # Calculate variance and oscillation metrics
    v0_std = np.std(model.history['v0'][steady_idx:])
    v1_std = np.std(model.history['v1'][steady_idx:])
    v2_std = np.std(model.history['v2'][steady_idx:])
    
    x1_std = np.std(model.history['x1'][steady_idx:])
    x2_std = np.std(model.history['x2'][steady_idx:])
    
    # Calculate the proportion of time spent in each mode
    modes = model.history['mode'][steady_idx:]
    mode_counts = {
        '00': modes.count('00'),
        '01': modes.count('01'),
        '10': modes.count('10'),
        '11': modes.count('11')
    }
    
    total_modes = len(modes)
    proportions = {mode: count/total_modes for mode, count in mode_counts.items()}
    
    # Print analysis
    print("\nStability Analysis:")
    print("-------------------")
    print("\nVelocity standard deviations:")
    print(f"  Lead Car (L): {v0_std:.3f} m/s")
    print(f"  Following Car 1 (F1): {v1_std:.3f} m/s")
    print(f"  Following Car 2 (F2): {v2_std:.3f} m/s")
    
    print("\nDistance standard deviations:")
    print(f"  L-F1 (x1): {x1_std:.3f} m")
    print(f"  F1-F2 (x2): {x2_std:.3f} m")
    
    print("\nProportion of time in each mode (λ1λ2):")
    for mode, prop in proportions.items():
        print(f"  Mode {mode}: {prop*100:.1f}%")
    
    # Determine overall stability
    velocity_stable = (v0_std < 2.0 and v1_std < 2.0 and v2_std < 2.0)
    distance_stable = (x1_std < 5.0 and x2_std < 5.0)
    
    print("\nStability assessment:")
    if velocity_stable and distance_stable:
        print("  The system exhibits STABLE following behavior.")
    elif velocity_stable:
        print("  The system exhibits STABLE velocity behavior but UNSTABLE distance behavior.")
    elif distance_stable:
        print("  The system exhibits UNSTABLE velocity behavior but STABLE distance behavior.")
    else:
        print("  The system exhibits UNSTABLE following behavior.")
    
    return {
        'v0_std': v0_std,
        'v1_std': v1_std,
        'v2_std': v2_std,
        'x1_std': x1_std,
        'x2_std': x2_std,
        'mode_proportions': proportions
    }