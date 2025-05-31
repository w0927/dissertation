from model import CircularCarFollowingModel
from visualization import CircularTrackVisualizer
from analysis import CircularTrackAnalyzer

# 修正后的main.py文件

# 简洁的main.py文件

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

def main():
    """
    主程序：演示三车跟随模型的完整工作流程
    """
    try:
        # 创建模型实例
        model = CircularCarFollowingModel(
            initial_velocities=[60.0, 60.0, 60.0],  # 初始速度
            initial_positions=[380.0, 300.0, 220.0],  # 初始位置
            d=(30.0, 50.0)  # 期望距离范围
        )
        
        # 运行仿真
        print("正在运行车辆跟随仿真...")
        model.run_simulation()
        
        # 创建可视化工具
        visualizer = CircularTrackVisualizer(model)
        
        # 创建分析工具
        analyzer = CircularTrackAnalyzer(model)
        
        # 执行分析
        print("\n正在进行系统稳定性分析...")
        stability_analysis = analyzer.analyze_stability()
        
        # 保存可视化结果
        print("\n正在绘制仿真结果...")
        visualizer.plot_results(save=True)
        
        # 创建并保存动画
        print("\n正在生成车辆运动动画...")
        visualizer.animate_vehicles(save=True)
        
        print("\n仿真完成！所有结果已保存")
        
    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()