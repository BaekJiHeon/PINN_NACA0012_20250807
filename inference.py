"""
NACA 0012 Flutter PINN - 추론 스크립트
Physics-Informed Neural Networks for 2-DOF Flutter Analysis

학습된 모델을 이용한 추론:
- 새로운 조건에서 유동장 예측
- 플러터 응답 분석
- 실시간 시뮬레이션
- 파라미터 스터디
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import argparse
import os
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

# 프로젝트 모듈들
from config import PhysicalParameters, DomainParameters, PINNConfig
from pinn_model import PINNModel, load_model
from boundary_conditions import NACAGeometry, BoundaryConditionManager
from structure_dynamics import TwoDOFStructure, ResponseAnalysis
from utils import get_device, setup_logging, Timer, FlowFieldVisualizer
from data_io import DataProcessor

class PINNInference:
    """PINN 추론 클래스"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.device = get_device()
        
        # 모델 로드
        self.model = self._load_trained_model(model_path, config_path)
        self.model.eval()
        
        # 기본 설정 (모델에서 로드 또는 기본값 사용)
        self._setup_default_configs()
        
        # 유틸리티 초기화
        self.geometry = NACAGeometry()
        self.visualizer = None  # 나중에 초기화
        self.response_analyzer = ResponseAnalysis()
        
        print(f"✅ 추론 시스템 초기화 완료 (디바이스: {self.device})")
    
    def _load_trained_model(self, model_path: str, config_path: Optional[str] = None) -> PINNModel:
        """학습된 모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 설정 정보 추출
        if 'model_config' in checkpoint and checkpoint['model_config'] is not None:
            config = checkpoint['model_config']
        else:
            # 기본 설정 사용
            config = PINNConfig()
            print("⚠️ 모델 설정을 찾을 수 없어 기본값을 사용합니다.")
        
        # 모델 생성 및 가중치 로드
        model = PINNModel(config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 메타데이터 출력
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            print(f"📊 모델 정보:")
            print(f"   - 학습 손실: {checkpoint.get('loss', 'N/A')}")
            print(f"   - 에포크: {checkpoint.get('epoch', 'N/A')}")
            print(f"   - 타임스탬프: {checkpoint.get('timestamp', 'N/A')}")
        
        return model
    
    def _setup_default_configs(self):
        """기본 설정 초기화"""
        from config import DEFAULT_PHYS_PARAMS, DEFAULT_DOMAIN_PARAMS
        
        self.phys_params = DEFAULT_PHYS_PARAMS
        self.domain_params = DEFAULT_DOMAIN_PARAMS
        
        # 도메인 경계
        self.domain_bounds = self.domain_params.get_nondim_bounds(self.phys_params.C_phys)
        
        # 시각화 도구 초기화
        self.visualizer = FlowFieldVisualizer(self.domain_bounds)
    
    def predict_flow_field(self, time_points: np.ndarray,
                          spatial_resolution: Tuple[int, int] = (100, 80)) -> Dict[str, np.ndarray]:
        """유동장 예측"""
        
        print(f"🔮 유동장 예측 중... (해상도: {spatial_resolution})")
        
        results = {
            'time': time_points,
            'x': None, 'y': None,
            'u': [], 'v': [], 'p': [],
            'vorticity': [], 'q_criterion': []
        }
        
        # 공간 그리드 생성
        nx, ny = spatial_resolution
        x = np.linspace(self.domain_bounds['x_min'], self.domain_bounds['x_max'], nx)
        y = np.linspace(self.domain_bounds['y_min'], self.domain_bounds['y_max'], ny)
        X, Y = np.meshgrid(x, y)
        results['x'], results['y'] = X, Y
        
        with torch.no_grad():
            for t in time_points:
                # 입력 그리드 생성
                T = np.full_like(X, t)
                grid_points = torch.tensor(
                    np.stack([T.flatten(), X.flatten(), Y.flatten()], axis=1),
                    dtype=torch.float32, device=self.device
                )
                
                # 모델 예측
                output = self.model(grid_points)
                flow_field = self.model.predict_flow_field(grid_points)
                
                # 결과 저장
                u = flow_field['u'].cpu().numpy().reshape(ny, nx)
                v = flow_field['v'].cpu().numpy().reshape(ny, nx)
                p = flow_field['p'].cpu().numpy().reshape(ny, nx)
                
                results['u'].append(u)
                results['v'].append(v)
                results['p'].append(p)
                
                # 와도 계산
                vorticity = self.model.compute_vorticity(grid_points)
                results['vorticity'].append(vorticity.cpu().numpy().reshape(ny, nx))
                
                # Q-criterion 계산
                q_crit = self.model.compute_q_criterion(grid_points)
                results['q_criterion'].append(q_crit.cpu().numpy().reshape(ny, nx))
        
        # 리스트를 numpy 배열로 변환
        for key in ['u', 'v', 'p', 'vorticity', 'q_criterion']:
            results[key] = np.array(results[key])
        
        print("✅ 유동장 예측 완료")
        return results
    
    def simulate_flutter_response(self, time_span: Tuple[float, float],
                                n_time_points: int = 500,
                                initial_conditions: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """플러터 응답 시뮬레이션"""
        
        print(f"🌊 플러터 응답 시뮬레이션 ({time_span[0]:.1f}s ~ {time_span[1]:.1f}s)")
        
        # 기본 초기조건
        if initial_conditions is None:
            initial_conditions = {
                'h0': 0.01,      # 초기 heave 변위
                'theta0': 0.05,  # 초기 pitch 각도
                'h_dot0': 0.0,   # 초기 heave 속도
                'theta_dot0': 0.0  # 초기 pitch 속도
            }
        
        # 구조 시스템 생성
        structure = TwoDOFStructure(self.phys_params)
        
        # 시뮬레이션 실행
        response = structure.solve_free_vibration(
            time_span, initial_conditions, n_time_points
        )
        
        # 주파수 분석
        for signal_name in ['h', 'theta']:
            fft_result = self.response_analyzer.fft_analysis(
                response['time'], response[signal_name]
            )
            dominant_freq = self.response_analyzer.identify_dominant_frequency(
                response['time'], response[signal_name]
            )
            
            print(f"   {signal_name} 지배 주파수: {dominant_freq['frequency']:.2f} Hz")
        
        print("✅ 플러터 시뮬레이션 완료")
        return response
    
    def parameter_study(self, parameter_name: str, 
                       parameter_values: np.ndarray,
                       base_conditions: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """파라미터 스터디"""
        
        print(f"📊 파라미터 스터디: {parameter_name}")
        
        if base_conditions is None:
            base_conditions = {
                'time_span': (0, 10),
                'initial_h': 0.01,
                'initial_theta': 0.05
            }
        
        results = {
            'parameter_values': parameter_values,
            'max_h_amplitude': [],
            'max_theta_amplitude': [],
            'h_frequency': [],
            'theta_frequency': [],
            'h_damping_ratio': [],
            'theta_damping_ratio': []
        }
        
        for param_value in parameter_values:
            print(f"   {parameter_name} = {param_value}")
            
            # 파라미터 업데이트
            modified_params = self._modify_parameter(parameter_name, param_value)
            
            # 구조 시스템 생성
            structure = TwoDOFStructure(modified_params)
            
            # 응답 시뮬레이션
            initial_conditions = {
                'h0': base_conditions['initial_h'],
                'theta0': base_conditions['initial_theta'],
                'h_dot0': 0.0,
                'theta_dot0': 0.0
            }
            
            response = structure.solve_free_vibration(
                base_conditions['time_span'], initial_conditions
            )
            
            # 결과 분석
            results['max_h_amplitude'].append(np.max(np.abs(response['h'])))
            results['max_theta_amplitude'].append(np.max(np.abs(response['theta'])))
            
            # 주파수 분석
            h_freq_info = self.response_analyzer.identify_dominant_frequency(
                response['time'], response['h']
            )
            theta_freq_info = self.response_analyzer.identify_dominant_frequency(
                response['time'], response['theta']
            )
            
            results['h_frequency'].append(h_freq_info['frequency'])
            results['theta_frequency'].append(theta_freq_info['frequency'])
            
            # 댐핑비 추정 (로그 감쇠법)
            h_damping = self._estimate_damping_ratio(response['time'], response['h'])
            theta_damping = self._estimate_damping_ratio(response['time'], response['theta'])
            
            results['h_damping_ratio'].append(h_damping)
            results['theta_damping_ratio'].append(theta_damping)
        
        # 배열로 변환
        for key in results:
            if key != 'parameter_values':
                results[key] = np.array(results[key])
        
        print("✅ 파라미터 스터디 완료")
        return results
    
    def _modify_parameter(self, param_name: str, param_value: float) -> PhysicalParameters:
        """파라미터 수정"""
        modified_params = PhysicalParameters()
        
        # 기존 값들 복사
        for attr in ['m', 'I_alpha', 'c_h', 'k_h', 'c_alpha', 'k_alpha', 'x_ref', 'C_phys']:
            setattr(modified_params, attr, getattr(self.phys_params, attr))
        
        # 지정된 파라미터 수정
        if hasattr(modified_params, param_name):
            setattr(modified_params, param_name, param_value)
        else:
            raise ValueError(f"알 수 없는 파라미터: {param_name}")
        
        return modified_params
    
    def _estimate_damping_ratio(self, time: np.ndarray, signal: np.ndarray) -> float:
        """로그 감쇠법으로 댐핑비 추정"""
        try:
            # 피크 찾기
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(signal, height=np.max(signal) * 0.1)
            
            if len(peaks) < 2:
                return 0.0  # 피크가 충분하지 않음
            
            # 연속된 피크들의 진폭비
            peak_amplitudes = signal[peaks]
            
            # 로그 감쇠율
            delta = np.mean(np.log(peak_amplitudes[:-1] / peak_amplitudes[1:]))
            
            # 댐핑비 계산
            zeta = delta / np.sqrt((2 * np.pi)**2 + delta**2)
            
            return max(0.0, min(1.0, zeta))  # 0-1 범위로 제한
            
        except:
            return 0.0
    
    def create_flow_animation(self, flow_results: Dict[str, np.ndarray],
                            structural_response: Optional[Dict[str, np.ndarray]] = None,
                            save_path: str = "flow_animation.mp4",
                            fps: int = 10,
                            show_airfoil: bool = True) -> None:
        """유동장 애니메이션 생성"""
        
        print(f"🎬 애니메이션 생성 중... ({save_path})")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('NACA 0012 Flutter Flow Field Animation')
        
        def animate(frame):
            for ax in axes.flat:
                ax.clear()
            
            t = flow_results['time'][frame]
            x, y = flow_results['x'], flow_results['y']
            u = flow_results['u'][frame]
            v = flow_results['v'][frame]
            p = flow_results['p'][frame]
            vorticity = flow_results['vorticity'][frame]
            
            # 압력장
            im1 = axes[0, 0].contourf(x, y, p, levels=20, cmap='RdBu_r')
            axes[0, 0].set_title(f'Pressure Field (t={t:.2f}s)')
            axes[0, 0].set_xlabel('x/C')
            axes[0, 0].set_ylabel('y/C')
            
            # 속도 크기
            vel_mag = np.sqrt(u**2 + v**2)
            im2 = axes[0, 1].contourf(x, y, vel_mag, levels=20, cmap='viridis')
            axes[0, 1].set_title('Velocity Magnitude')
            axes[0, 1].set_xlabel('x/C')
            axes[0, 1].set_ylabel('y/C')
            
            # 속도 벡터
            stride = max(1, len(x) // 15)
            axes[1, 0].quiver(x[::stride, ::stride], y[::stride, ::stride],
                             u[::stride, ::stride], v[::stride, ::stride],
                             alpha=0.7)
            axes[1, 0].set_title('Velocity Vectors')
            axes[1, 0].set_xlabel('x/C')
            axes[1, 0].set_ylabel('y/C')
            
            # 와도 (소용돌이 강도)
            im4 = axes[1, 1].contourf(x, y, vorticity, levels=20, cmap='RdBu')
            axes[1, 1].set_title('Vorticity')
            axes[1, 1].set_xlabel('x/C')
            axes[1, 1].set_ylabel('y/C')
            
            # 에어포일 추가
            if show_airfoil:
                h, theta = 0.0, 0.0  # 기본값
                
                if structural_response and frame < len(structural_response['time']):
                    h = structural_response['h'][frame]
                    theta = structural_response['theta'][frame]
                
                # 변환된 에어포일 좌표
                airfoil_x, airfoil_y = self.geometry.transform_airfoil(
                    h, theta, self.phys_params.x_ref
                )
                
                for ax in axes.flat:
                    ax.plot(airfoil_x, airfoil_y, 'k-', linewidth=2)
                    ax.fill(airfoil_x, airfoil_y, color='white', alpha=0.8)
            
            # 축 설정
            for ax in axes.flat:
                ax.set_xlim(self.domain_bounds['x_min'], self.domain_bounds['x_max'])
                ax.set_ylim(self.domain_bounds['y_min'], self.domain_bounds['y_max'])
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
        
        # 애니메이션 생성
        anim = animation.FuncAnimation(
            fig, animate, frames=len(flow_results['time']),
            interval=1000//fps, blit=False
        )
        
        # 저장
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(save_path, writer=writer)
            print(f"✅ 애니메이션 저장 완료: {save_path}")
        except Exception as e:
            print(f"⚠️ 애니메이션 저장 실패: {e}")
        
        plt.close()
    
    def plot_parameter_study_results(self, study_results: Dict[str, np.ndarray],
                                   parameter_name: str,
                                   save_path: Optional[str] = None) -> None:
        """파라미터 스터디 결과 플롯"""
        
        param_values = study_results['parameter_values']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Parameter Study: {parameter_name}')
        
        # 최대 진폭
        axes[0, 0].plot(param_values, study_results['max_h_amplitude'], 'b-o', label='Heave')
        axes[0, 0].plot(param_values, study_results['max_theta_amplitude'], 'r-s', label='Pitch')
        axes[0, 0].set_xlabel(parameter_name)
        axes[0, 0].set_ylabel('Max Amplitude')
        axes[0, 0].set_title('Maximum Amplitude vs Parameter')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 주파수
        axes[0, 1].plot(param_values, study_results['h_frequency'], 'b-o', label='Heave')
        axes[0, 1].plot(param_values, study_results['theta_frequency'], 'r-s', label='Pitch')
        axes[0, 1].set_xlabel(parameter_name)
        axes[0, 1].set_ylabel('Frequency (Hz)')
        axes[0, 1].set_title('Dominant Frequency vs Parameter')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 댐핑비
        axes[1, 0].plot(param_values, study_results['h_damping_ratio'], 'b-o', label='Heave')
        axes[1, 0].plot(param_values, study_results['theta_damping_ratio'], 'r-s', label='Pitch')
        axes[1, 0].set_xlabel(parameter_name)
        axes[1, 0].set_ylabel('Damping Ratio')
        axes[1, 0].set_title('Damping Ratio vs Parameter')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 안정성 다이어그램
        axes[1, 1].plot(param_values, study_results['h_damping_ratio'], 'b-o', label='Heave')
        axes[1, 1].plot(param_values, study_results['theta_damping_ratio'], 'r-s', label='Pitch')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Flutter Boundary')
        axes[1, 1].set_xlabel(parameter_name)
        axes[1, 1].set_ylabel('Damping Ratio')
        axes[1, 1].set_title('Stability Diagram')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 파라미터 스터디 결과 저장: {save_path}")
        else:
            plt.show()

def main():
    """메인 추론 함수"""
    parser = argparse.ArgumentParser(description="NACA 0012 Flutter PINN 추론")
    
    parser.add_argument("--model", type=str, required=True,
                       help="학습된 모델 파일 경로")
    parser.add_argument("--mode", choices=["flow", "flutter", "param_study", "animation"],
                       default="flow", help="추론 모드")
    parser.add_argument("--time_start", type=float, default=0.0,
                       help="시작 시간")
    parser.add_argument("--time_end", type=float, default=5.0,
                       help="종료 시간")
    parser.add_argument("--n_time_points", type=int, default=50,
                       help="시간 점 수")
    parser.add_argument("--resolution", type=int, nargs=2, default=[100, 80],
                       help="공간 해상도 [nx, ny]")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                       help="출력 디렉터리")
    parser.add_argument("--param_name", type=str, default="k_h",
                       help="파라미터 스터디용 파라미터 이름")
    parser.add_argument("--param_range", type=float, nargs=3, 
                       default=[400, 600, 21],  # [min, max, num_points]
                       help="파라미터 범위 [min, max, num_points]")
    
    args = parser.parse_args()
    
    # 출력 디렉터리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 로깅 설정
    setup_logging("INFO", os.path.join(args.output_dir, "inference.log"))
    
    # 추론 시스템 초기화
    inference = PINNInference(args.model)
    
    try:
        if args.mode == "flow":
            # 유동장 예측
            time_points = np.linspace(args.time_start, args.time_end, args.n_time_points)
            
            with Timer("유동장 예측"):
                flow_results = inference.predict_flow_field(
                    time_points, tuple(args.resolution)
                )
            
            # 특정 시간에서의 유동장 시각화
            mid_idx = len(time_points) // 2
            inference.visualizer.plot_flow_field(
                flow_results['x'], flow_results['y'],
                flow_results['u'][mid_idx], flow_results['v'][mid_idx], 
                flow_results['p'][mid_idx],
                save_path=os.path.join(args.output_dir, "flow_field_prediction.png")
            )
            
        elif args.mode == "flutter":
            # 플러터 응답 시뮬레이션
            with Timer("플러터 시뮬레이션"):
                flutter_response = inference.simulate_flutter_response(
                    (args.time_start, args.time_end), args.n_time_points
                )
            
            # 응답 분석 및 시각화
            inference.response_analyzer.plot_response_analysis(
                flutter_response['time'], flutter_response['h'], flutter_response['theta'],
                np.zeros_like(flutter_response['time']), np.zeros_like(flutter_response['time']),
                save_path=os.path.join(args.output_dir, "flutter_response.png")
            )
            
        elif args.mode == "param_study":
            # 파라미터 스터디
            param_values = np.linspace(args.param_range[0], args.param_range[1], 
                                     int(args.param_range[2]))
            
            with Timer(f"파라미터 스터디 ({args.param_name})"):
                study_results = inference.parameter_study(args.param_name, param_values)
            
            # 결과 시각화
            inference.plot_parameter_study_results(
                study_results, args.param_name,
                save_path=os.path.join(args.output_dir, f"param_study_{args.param_name}.png")
            )
            
        elif args.mode == "animation":
            # 애니메이션 생성
            time_points = np.linspace(args.time_start, args.time_end, args.n_time_points)
            
            with Timer("유동장 예측 (애니메이션용)"):
                flow_results = inference.predict_flow_field(
                    time_points, tuple(args.resolution)
                )
            
            with Timer("플러터 응답 (애니메이션용)"):
                flutter_response = inference.simulate_flutter_response(
                    (args.time_start, args.time_end), args.n_time_points
                )
            
            with Timer("애니메이션 생성"):
                inference.create_flow_animation(
                    flow_results, flutter_response,
                    save_path=os.path.join(args.output_dir, "flutter_animation.mp4"),
                    fps=10
                )
        
        print(f"🎉 추론 완료! 결과는 {args.output_dir}에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 추론 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()