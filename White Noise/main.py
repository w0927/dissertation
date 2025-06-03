from model import CircularCarFollowingModel
from visualization import CircularTrackVisualizer
from analysis import CircularTrackAnalyzer

import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from model import CircularCarFollowingModel
    from visualization import CircularTrackVisualizer
    from analysis import CircularTrackAnalyzer
    print("所有模块导入成功")
except ImportError as e:
    print(f"导入错误: {e}")
    print("请检查以下几点:")
    print("1. 确保所有.py文件都在同一个目录下")
    print("2. 检查model.py中是否有CircularCarFollowingModel类")
    print("3. 检查文件名拼写是否正确")
    sys.exit(1)

def create_custom_spacing_positions(L_position, L_to_F1_distance, F1_to_F2_distance, track_length=6000):
    """
    根据指定的车间距创建初始位置
    """
    F1_position = (L_position - L_to_F1_distance + track_length) % track_length
    F2_position = (F1_position - F1_to_F2_distance + track_length) % track_length
    
    print(f"🎯 自定义间距设置:")
    print(f"   L车位置: {L_position}m")
    print(f"   F1车位置: {F1_position}m (距离L车: {L_to_F1_distance}m)")
    print(f"   F2车位置: {F2_position}m (距离F1车: {F1_to_F2_distance}m)")
    
    return [L_position, F1_position, F2_position]

def main():
    """
    主程序：延迟引入噪声的车辆跟随模型演示
    """
    print("🚗" + "="*70)
    print("   延迟引入噪声的车辆跟随模型测试")
    print("="*70 + "🚗")
    
    try:
        # 设置自定义间距
        custom_positions = create_custom_spacing_positions(
            L_position=3000,      # 轨道中间位置
            L_to_F1_distance=50,  # 初始间距大于目标值，观察收敛过程
            F1_to_F2_distance=50,
            track_length=6000
        )
        
        # 创建模型实例 - 启用延迟噪声功能
        model = CircularCarFollowingModel(
            # 基础设置
            track_length=6000.0,   # 较长轨道，动画更慢
            d=40.0,                # 目标距离
            initial_positions=custom_positions,
            initial_velocities=[60.0, 60.0, 60.0],  # 统一60 m/s初始速度
            t_max=400.0,           # 总仿真时间400秒
            
            # 稳定性参数（满足老师要求的条件）
            a11=0.5, a0=0.3,      # F1车参数
            b1=1.0, b0=1.5,       # F2车参数  
            c1=0.3, c0=0.5,       # L车参数
            
            # 关键：启用延迟噪声功能
            enable_L_noise=True,        # 启用噪声
            noise_std=5.0,             # 噪声标准差
            noise_start_time=150.0,    # 150秒后引入噪声
            noise_seed=42              # 固定种子确保可重现
        )
        
        print(f"\n✅ 模型创建成功")
        print(f"   噪声启用: {model.enable_L_noise}")
        print(f"   噪声强度: {model.noise_std} m/s")
        print(f"   噪声开始时间: {model.noise_start_time} 秒")
        
        # 验证稳定性条件
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
        print("\n正在运行车辆跟随仿真...")
        model.run_simulation()
        
        # 检查噪声数据是否生成
        print(f"\n🔍 噪声数据检查:")
        print(f"   L_noise长度: {len(model.history.get('L_noise', []))}")
        print(f"   L_speed_without_noise长度: {len(model.history.get('L_speed_without_noise', []))}")
        
        if len(model.history.get('L_noise', [])) > 0:
            non_zero_noise = [n for n in model.history['L_noise'] if n != 0]
            print(f"   非零噪声点数: {len(non_zero_noise)}")
            if non_zero_noise:
                print(f"   噪声范围: [{min(non_zero_noise):.2f}, {max(non_zero_noise):.2f}]")
        
        # 创建可视化工具
        visualizer = CircularTrackVisualizer(model)
        
        # 创建分析工具
        analyzer = CircularTrackAnalyzer(model)
        
        # 执行分析
        print("\n正在进行系统稳定性分析...")
        stability_analysis = analyzer.analyze_stability()
        
        # 保存可视化结果 - 这里会显示5个子图包括噪声对比
        print("\n正在绘制仿真结果...")
        visualizer.plot_results(save=True)
        
        # 创建并保存动画
        print("\n正在生成车辆运动动画...")
        visualizer.animate_vehicles(save=True)
        
        print("\n🎉 仿真完成！所有结果已保存")
        print("\n📋 你应该看到:")
        print("✅ 5个子图（包括噪声图）")
        print("✅ 速度图中的红色实线vs红色虚线对比")
        print("✅ 橙色虚线标记150秒噪声开始时间点")
        print("✅ 第5个子图显示噪声波形")
        
    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback
        traceback.print_exc()

def test_different_noise_levels():
    """
    测试不同噪声强度的影响
    """
    print("🧪 测试不同噪声强度")
    print("="*50)
    
    noise_levels = [0.0, 2.0, 5.0, 8.0]
    
    for i, noise_std in enumerate(noise_levels):
        print(f"\n--- 测试 {i+1}: 噪声强度 {noise_std} m/s ---")
        
        enable_noise = noise_std > 0
        custom_positions = create_custom_spacing_positions(3000, 50, 50, 6000)
        
        model = CircularCarFollowingModel(
            track_length=6000.0,
            d=40.0,
            initial_positions=custom_positions,
            initial_velocities=[60.0, 60.0, 60.0],
            t_max=300.0,  # 缩短时间用于对比测试
            
            a11=0.5, a0=0.3,
            b1=1.0, b0=1.5,
            c1=0.3, c0=0.5,
            
            enable_L_noise=enable_noise,
            noise_std=noise_std,
            noise_start_time=100.0,  # 100秒后引入噪声
            noise_seed=42
        )
        
        model.run_simulation()
        
        # 只为最后一个场景显示图表
        if i == len(noise_levels) - 1:
            visualizer = CircularTrackVisualizer(model)
            visualizer.plot_results()
        
        print(f"✅ 测试 {i+1} 完成")

if __name__ == "__main__":
    # 选择运行模式
    print("选择运行模式:")
    print("1. 延迟噪声演示（推荐）")
    print("2. 不同噪声强度对比测试")
    
    choice = input("请输入选择 (1 或 2，默认1): ").strip()
    
    if choice == "2":
        test_different_noise_levels()
    else:
        main()