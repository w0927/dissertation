import numpy as np

class CircularTrackAnalyzer:
    def __init__(self, model):
        """
        初始化分析工具
        
        Args:
        - model: 车辆跟随模型实例
        """
        self.model = model
    
    def analyze_stability(self):
        """
        分析车辆跟随系统的稳定性
        
        Returns:
        - 包含稳定性分析结果的字典
        """
        # 取最后2/3的数据作为稳态分析区间
        steady_start = len(self.model.history['time']) // 3
        
        # 提取稳态数据
        velocities = {
            'lead_car': np.array(self.model.history['v0'][steady_start:]),
            'follower1': np.array(self.model.history['v1'][steady_start:]),
            'follower2': np.array(self.model.history['v2'][steady_start:])
        }
        
        # 计算位置和距离
        positions = {
            'lead_car': np.array(self.model.history['x0'][steady_start:]),
            'follower1': np.array(self.model.history['y1'][steady_start:]),
            'follower2': np.array(self.model.history['y2'][steady_start:])
        }
        
        distances = {
            'L_F1': np.array(self.model.history['x1'][steady_start:]),
            'F1_F2': np.array(self.model.history['x2'][steady_start:])
        }
        
        # 速度稳定性分析
        velocity_stats = {
            'mean': {name: v.mean() for name, v in velocities.items()},
            'std': {name: v.std() for name, v in velocities.items()},
            'coefficient_of_variation': {name: v.std() / v.mean() * 100 for name, v in velocities.items()}
        }
        
        # 距离稳定性分析
        distance_stats = {
            'mean': {name: d.mean() for name, d in distances.items()},
            'std': {name: d.std() for name, d in distances.items()},
            'deviation_from_target': {
                'L_F1': abs(distances['L_F1'].mean() - self.model.d),
                'F1_F2': abs(distances['F1_F2'].mean() - self.model.d)
            }
        }
        
        # 模式分析
        mode_counts = {}
        total_steps = len(self.model.history['mode'][steady_start:])
        for mode in ['00', '01', '10', '11']:
            mode_counts[mode] = self.model.history['mode'][steady_start:].count(mode) / total_steps * 100
        
        # 稳定性综合评估
        is_velocity_stable = all(std < 2.0 for std in velocity_stats['std'].values())
        is_distance_stable = all(std < 5.0 for std in distance_stats['std'].values())
        
        # 组装分析结果
        analysis_result = {
            'velocity_stability': {
                'stats': velocity_stats,
                'is_stable': is_velocity_stable
            },
            'distance_stability': {
                'stats': distance_stats,
                'is_stable': is_distance_stable
            },
            'mode_distribution': mode_counts,
            'overall_stability': is_velocity_stable and is_distance_stable
        }
        
        # 打印分析结果
        self._print_analysis_summary(analysis_result)
        
        return analysis_result
    
    def _print_analysis_summary(self, analysis_result):
        """
        打印分析结果摘要
        
        Args:
        - analysis_result: 分析结果字典
        """
        print("\nVehicle following system stability analysis report")
        print("=" * 40)
        
        # 速度稳定性
        print("\nVelovity stability:")
        for name, stats in analysis_result['velocity_stability']['stats']['mean'].items():
            print(f"{name.replace('_', ' ').title()}:")
            print(f"  Average velocity: {stats:.2f} m/s")
            print(f"  Velocity standard deviation: {analysis_result['velocity_stability']['stats']['std'][name]:.2f} m/s")
            print(f"  Variable ccoefficient: {analysis_result['velocity_stability']['stats']['coefficient_of_variation'][name]:.2f}%")
        
        # 距离稳定性
        print("\nDistanec stability:")
        for name, stats in analysis_result['distance_stability']['stats']['mean'].items():
            print(f"{name} Distance:")
            print(f"  Mean distance: {stats:.2f} m")
            print(f"  Distance standard deviation: {analysis_result['distance_stability']['stats']['std'][name]:.2f} m")
            print(f"  Distance deiation from terget: {analysis_result['distance_stability']['stats']['deviation_from_target'][name]:.2f} m")
        
        # 模式分布
        print("\nSystem pattern distribution:")
        for mode, percentage in analysis_result['mode_distribution'].items():
            print(f"Mode {mode}: {percentage:.2f}%")
        
        # 总体稳定性
        print("\nGlobal stability assessment:")
        if analysis_result['overall_stability']:
            print("The system presents highly stable following behavior.")
        else:
            print("The following behavior of the system is unstable")
            if not analysis_result['velocity_stability']['is_stable']:
                print("  - Great speed fluctuation")
            if not analysis_result['distance_stability']['is_stable']:
                print("  - Distance control is unstable")
    
    def compare_scenarios(self, scenarios):
        """
        比较不同场景下的系统表现
        
        Args:
        - scenarios: 包含不同模型配置的列表
        
        Returns:
        - 各场景分析结果的比较
        """
        scenario_results = {}
        
        for i, scenario_model in enumerate(scenarios, 1):
            # 运行模拟
            scenario_model.run_simulation()
            
            # 创建分析器并分析
            analyzer = CircularTrackAnalyzer(scenario_model)
            scenario_results[f'Scenario {i}'] = analyzer.analyze_stability()
        
        # 打印比较结果
        self._print_scenario_comparison(scenario_results)
        
        return scenario_results
    
    def _print_scenario_comparison(self, scenario_results):
        """
        打印场景比较结果
        
        Args:
        - scenario_results: 不同场景的分析结果
        """
        print("\nScene comparative analysis")
        print("=" * 30)
        
        # 表头
        headers = ["Scenario", "Velocity Stable", "Distance Stable", "Overall Stable"]
        print(f"{headers[0]:<10} {headers[1]:<15} {headers[2]:<15} {headers[3]}")
        print("-" * 50)
        
        # 逐个场景输出
        for scenario, result in scenario_results.items():
            print(f"{scenario:<10} "
                  f"{str(result['velocity_stability']['is_stable']):<15} "
                  f"{str(result['distance_stability']['is_stable']):<15} "
                  f"{str(result['overall_stability'])}")