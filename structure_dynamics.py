"""
NACA 0012 Flutter PINN - 구조 동역학 모듈
Physics-Informed Neural Networks for 2-DOF Flutter Analysis

2-DOF 플러터 시스템:
- Heave-Pitch 결합 운동
- 공력 하중 계산
- 구조 파라미터 식별
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.integrate import solve_ivp
from config import PhysicalParameters
import matplotlib.pyplot as plt

class TwoDOFStructure:
    """2-DOF 구조 시스템 (Heave-Pitch)"""
    
    def __init__(self, phys_params: PhysicalParameters):
        # 구조 파라미터
        self.m = phys_params.m              # 질량 (kg)
        self.I_alpha = phys_params.I_alpha  # 관성모멘트 (kg·m²)
        self.c_h = phys_params.c_h          # heave 댐핑 (N·s/m)
        self.k_h = phys_params.k_h          # heave 강성 (N/m)
        self.c_alpha = phys_params.c_alpha  # pitch 댐핑 (N·m·s)
        self.k_alpha = phys_params.k_alpha  # pitch 강성 (N·m/rad)
        self.x_ref = phys_params.x_ref      # 기준점 (chord 기준)
        
        # 비차원 고유 주파수
        self.omega_h = np.sqrt(self.k_h / self.m)
        self.omega_alpha = np.sqrt(self.k_alpha / self.I_alpha)
        
        # 댐핑비
        self.zeta_h = self.c_h / (2 * np.sqrt(self.k_h * self.m))
        self.zeta_alpha = self.c_alpha / (2 * np.sqrt(self.k_alpha * self.I_alpha))
        
        print(f"구조 고유 주파수: ω_h = {self.omega_h:.2f} rad/s, ω_α = {self.omega_alpha:.2f} rad/s")
        print(f"댐핑비: ζ_h = {self.zeta_h:.4f}, ζ_α = {self.zeta_alpha:.4f}")
    
    def get_mass_matrix(self) -> np.ndarray:
        """질량 행렬"""
        return np.array([
            [self.m, 0],
            [0, self.I_alpha]
        ])
    
    def get_damping_matrix(self) -> np.ndarray:
        """댐핑 행렬"""
        return np.array([
            [self.c_h, 0],
            [0, self.c_alpha]
        ])
    
    def get_stiffness_matrix(self) -> np.ndarray:
        """강성 행렬"""
        return np.array([
            [self.k_h, 0],
            [0, self.k_alpha]
        ])
    
    def equations_of_motion(self, t: float, y: np.ndarray, 
                          aero_force_func: Optional[callable] = None) -> np.ndarray:
        """
        운동 방정식 (1차 ODE 시스템)
        
        state vector: y = [h, θ, ḣ, θ̇]
        derivative: ẏ = [ḣ, θ̇, ḧ, θ̈]
        """
        h, theta, h_dot, theta_dot = y
        
        # 공력 하중 계산
        if aero_force_func is not None:
            lift, moment = aero_force_func(t, h, theta, h_dot, theta_dot)
        else:
            lift, moment = 0.0, 0.0
        
        # 구조 방정식
        h_ddot = (-self.c_h * h_dot - self.k_h * h - lift) / self.m
        theta_ddot = (-self.c_alpha * theta_dot - self.k_alpha * theta + moment) / self.I_alpha
        
        return np.array([h_dot, theta_dot, h_ddot, theta_ddot])
    
    def solve_free_vibration(self, t_span: Tuple[float, float], 
                           initial_conditions: Dict[str, float],
                           n_points: int = 1000) -> Dict[str, np.ndarray]:
        """자유 진동 해석"""
        
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        y0 = [initial_conditions['h0'], initial_conditions['theta0'],
              initial_conditions['h_dot0'], initial_conditions['theta_dot0']]
        
        sol = solve_ivp(
            lambda t, y: self.equations_of_motion(t, y),
            t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8
        )
        
        return {
            'time': sol.t,
            'h': sol.y[0],
            'theta': sol.y[1], 
            'h_vel': sol.y[2],
            'theta_vel': sol.y[3]
        }

class AerodynamicLoads:
    """공력 하중 계산"""
    
    def __init__(self, phys_params: PhysicalParameters):
        self.C_phys = phys_params.C_phys
        self.rho = phys_params.rho
        self.U_inf = phys_params.U_inf
        
        # 공력 계수 (단순 모델)
        self.C_L_alpha = 2 * np.pi  # 양력 기울기
        self.C_M_alpha = -0.25      # 모멘트 기울기
        
    def compute_unsteady_loads(self, model_output: torch.Tensor,
                             surface_points: torch.Tensor,
                             normals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        비정상 공력 하중 계산 (압력 적분)
        
        Args:
            model_output: 표면에서의 유동장 [N, 3] (u, v, p)
            surface_points: 표면점 좌표 [N, 3] (t, x, y)
            normals: 표면 법선 벡터 [N, 2]
        """
        pressure = model_output[:, 2:3]  # 압력
        
        # 표면 적분을 위한 요소 길이
        x_surf = surface_points[:, 1]
        y_surf = surface_points[:, 2]
        
        # 인접 점 간 거리
        dx = torch.diff(x_surf, prepend=x_surf[-1:])
        dy = torch.diff(y_surf, prepend=y_surf[-1:])
        ds = torch.sqrt(dx**2 + dy**2)
        
        # 압력에 의한 힘
        force_x = pressure.squeeze() * normals[:, 0] * ds
        force_y = pressure.squeeze() * normals[:, 1] * ds
        
        # 총 힘 (양력은 y 방향)
        lift = -torch.sum(force_y)  # 상향이 양수
        
        # 기준점에 대한 모멘트
        x_rel = x_surf - 0.19  # x_ref = 0.19C
        moment = torch.sum(force_y * x_rel * ds - force_x * y_surf * ds)
        
        return {
            'lift': lift,
            'moment': moment,
            'pressure_distribution': pressure
        }
    
    def quasi_steady_model(self, h: float, theta: float,
                          h_vel: float, theta_vel: float) -> Tuple[float, float]:
        """준정상 공력 모델"""
        
        # 유효 받음각
        alpha_eff = theta + h_vel / self.U_inf
        
        # 양력 및 모멘트 계수
        C_L = self.C_L_alpha * alpha_eff
        C_M = self.C_M_alpha * alpha_eff
        
        # 동압
        q = 0.5 * self.rho * self.U_inf**2
        
        # 힘과 모멘트
        lift = q * self.C_phys * C_L
        moment = q * self.C_phys**2 * C_M
        
        return lift, moment

class ParameterIdentification:
    """구조 파라미터 식별"""
    
    def __init__(self, reference_params: PhysicalParameters):
        self.ref_params = reference_params
        
        # 학습 가능한 파라미터
        self.log_c_h = nn.Parameter(torch.log(torch.tensor(reference_params.c_h)))
        self.log_k_h = nn.Parameter(torch.log(torch.tensor(reference_params.k_h)))
        self.log_c_alpha = nn.Parameter(torch.log(torch.tensor(reference_params.c_alpha)))
        self.log_k_alpha = nn.Parameter(torch.log(torch.tensor(reference_params.k_alpha)))
        
    @property
    def c_h(self) -> torch.Tensor:
        return torch.exp(self.log_c_h)
    
    @property
    def k_h(self) -> torch.Tensor:
        return torch.exp(self.log_k_h)
    
    @property
    def c_alpha(self) -> torch.Tensor:
        return torch.exp(self.log_c_alpha)
    
    @property
    def k_alpha(self) -> torch.Tensor:
        return torch.exp(self.log_k_alpha)
    
    def get_current_params(self) -> Dict[str, float]:
        """현재 파라미터 값 반환"""
        return {
            'c_h': self.c_h.item(),
            'k_h': self.k_h.item(),
            'c_alpha': self.c_alpha.item(),
            'k_alpha': self.k_alpha.item()
        }
    
    def compute_parameter_loss(self, predicted_response: Dict[str, torch.Tensor],
                             measured_response: Dict[str, torch.Tensor],
                             weight_displacement: float = 1.0,
                             weight_velocity: float = 1.0) -> torch.Tensor:
        """파라미터 식별을 위한 손실 계산"""
        
        loss = torch.tensor(0.0)
        
        # 변위 오차
        if 'h' in predicted_response and 'h' in measured_response:
            loss += weight_displacement * torch.mean(
                (predicted_response['h'] - measured_response['h'])**2
            )
        
        if 'theta' in predicted_response and 'theta' in measured_response:
            loss += weight_displacement * torch.mean(
                (predicted_response['theta'] - measured_response['theta'])**2
            )
        
        # 속도 오차
        if 'h_vel' in predicted_response and 'h_vel' in measured_response:
            loss += weight_velocity * torch.mean(
                (predicted_response['h_vel'] - measured_response['h_vel'])**2
            )
        
        if 'theta_vel' in predicted_response and 'theta_vel' in measured_response:
            loss += weight_velocity * torch.mean(
                (predicted_response['theta_vel'] - measured_response['theta_vel'])**2
            )
        
        return loss

class FlutterAnalysis:
    """플러터 해석"""
    
    def __init__(self, structure: TwoDOFStructure):
        self.structure = structure
        
    def eigenvalue_analysis(self, U_inf_range: np.ndarray) -> Dict[str, np.ndarray]:
        """고유값 해석 (플러터 속도 예측)"""
        
        frequencies = []
        damping_ratios = []
        
        # 질량, 댐핑, 강성 행렬
        M = self.structure.get_mass_matrix()
        C = self.structure.get_damping_matrix()
        K = self.structure.get_stiffness_matrix()
        
        for U in U_inf_range:
            # 공력 행렬 (단순 모델)
            # 실제로는 비정상 공력 모델이 필요
            A_aero = np.array([
                [0, 2*np.pi*U],
                [0, -0.25*U**2]
            ])
            
            # 특성방정식: det(M*s² + C*s + (K - A_aero)) = 0
            # 1차 시스템으로 변환: [ẋ] = [A][x]
            # x = [h, θ, ḣ, θ̇]
            A = np.block([
                [np.zeros((2, 2)), np.eye(2)],
                [-np.linalg.inv(M) @ (K - A_aero), -np.linalg.inv(M) @ C]
            ])
            
            # 고유값 계산
            eigenvals = np.linalg.eigvals(A)
            
            # 주파수와 댐핑비 추출
            freq = np.abs(eigenvals.imag) / (2 * np.pi)
            damp = -eigenvals.real / np.abs(eigenvals)
            
            frequencies.append(freq)
            damping_ratios.append(damp)
        
        return {
            'U_inf': U_inf_range,
            'frequencies': np.array(frequencies),
            'damping_ratios': np.array(damping_ratios)
        }
    
    def find_flutter_speed(self, U_inf_range: np.ndarray) -> Dict[str, float]:
        """플러터 속도 탐지"""
        
        results = self.eigenvalue_analysis(U_inf_range)
        damping_ratios = results['damping_ratios']
        
        # 댐핑비가 0이 되는 지점 찾기
        flutter_indices = []
        for mode in range(damping_ratios.shape[1]):
            damp_curve = damping_ratios[:, mode]
            
            # 부호 변화 지점 찾기
            sign_changes = np.where(np.diff(np.sign(damp_curve)))[0]
            
            for idx in sign_changes:
                if damp_curve[idx] < 0 and damp_curve[idx+1] > 0:
                    # 선형 보간으로 정확한 플러터 속도 계산
                    U1, U2 = U_inf_range[idx], U_inf_range[idx+1]
                    d1, d2 = damp_curve[idx], damp_curve[idx+1]
                    U_flutter = U1 - d1 * (U2 - U1) / (d2 - d1)
                    
                    flutter_indices.append({
                        'mode': mode,
                        'U_flutter': U_flutter,
                        'frequency': results['frequencies'][idx, mode]
                    })
        
        return flutter_indices
    
    def plot_v_g_diagram(self, U_inf_range: np.ndarray, 
                        save_path: Optional[str] = None):
        """V-g 다이어그램 그리기"""
        
        results = self.eigenvalue_analysis(U_inf_range)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 주파수 그래프
        for mode in range(results['frequencies'].shape[1]):
            ax1.plot(U_inf_range, results['frequencies'][:, mode], 
                    label=f'Mode {mode+1}')
        
        ax1.set_xlabel('풍속 U (m/s)')
        ax1.set_ylabel('주파수 f (Hz)')
        ax1.set_title('주파수 vs 풍속')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 댐핑비 그래프
        for mode in range(results['damping_ratios'].shape[1]):
            ax2.plot(U_inf_range, results['damping_ratios'][:, mode], 
                    label=f'Mode {mode+1}')
        
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Flutter Boundary')
        ax2.set_xlabel('풍속 U (m/s)')
        ax2.set_ylabel('댐핑비 ζ')
        ax2.set_title('댐핑비 vs 풍속')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

class ResponseAnalysis:
    """응답 분석"""
    
    def __init__(self):
        pass
    
    def compute_rms(self, time_series: np.ndarray) -> float:
        """RMS 값 계산"""
        return np.sqrt(np.mean(time_series**2))
    
    def fft_analysis(self, time: np.ndarray, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """FFT 주파수 분석"""
        
        dt = time[1] - time[0]
        N = len(signal)
        
        # FFT 계산
        fft_signal = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(N, dt)
        
        # 양의 주파수만
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        amplitude = np.abs(fft_signal[positive_freq_idx]) * 2 / N
        phase = np.angle(fft_signal[positive_freq_idx])
        
        return {
            'frequencies': frequencies,
            'amplitude': amplitude,
            'phase': phase,
            'power_spectrum': amplitude**2
        }
    
    def identify_dominant_frequency(self, time: np.ndarray, 
                                  signal: np.ndarray) -> Dict[str, float]:
        """지배 주파수 식별"""
        
        fft_result = self.fft_analysis(time, signal)
        
        # 최대 진폭의 주파수 찾기
        max_idx = np.argmax(fft_result['amplitude'])
        dominant_freq = fft_result['frequencies'][max_idx]
        dominant_amp = fft_result['amplitude'][max_idx]
        
        return {
            'frequency': dominant_freq,
            'amplitude': dominant_amp,
            'period': 1.0 / dominant_freq if dominant_freq > 0 else np.inf
        }
    
    def plot_response_analysis(self, time: np.ndarray, 
                             h: np.ndarray, theta: np.ndarray,
                             lift: np.ndarray, moment: np.ndarray,
                             save_path: Optional[str] = None):
        """응답 분석 플롯"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 시간 이력
        axes[0, 0].plot(time, h, 'b-', label='Heave')
        axes[0, 0].set_xlabel('시간 (s)')
        axes[0, 0].set_ylabel('Heave h/C')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('Heave 응답')
        
        axes[0, 1].plot(time, theta * 180/np.pi, 'r-', label='Pitch')
        axes[0, 1].set_xlabel('시간 (s)')
        axes[0, 1].set_ylabel('Pitch θ (deg)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Pitch 응답')
        
        axes[0, 2].plot(time, lift, 'g-', label='Lift')
        axes[0, 2].plot(time, moment, 'm-', label='Moment')
        axes[0, 2].set_xlabel('시간 (s)')
        axes[0, 2].set_ylabel('공력 하중')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_title('공력 하중')
        
        # FFT 분석
        h_fft = self.fft_analysis(time, h)
        theta_fft = self.fft_analysis(time, theta)
        lift_fft = self.fft_analysis(time, lift)
        
        axes[1, 0].semilogy(h_fft['frequencies'], h_fft['amplitude'], 'b-')
        axes[1, 0].set_xlabel('주파수 (Hz)')
        axes[1, 0].set_ylabel('Heave 진폭')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Heave FFT')
        
        axes[1, 1].semilogy(theta_fft['frequencies'], theta_fft['amplitude'], 'r-')
        axes[1, 1].set_xlabel('주파수 (Hz)')
        axes[1, 1].set_ylabel('Pitch 진폭')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('Pitch FFT')
        
        axes[1, 2].semilogy(lift_fft['frequencies'], lift_fft['amplitude'], 'g-')
        axes[1, 2].set_xlabel('주파수 (Hz)')
        axes[1, 2].set_ylabel('Lift 진폭')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_title('Lift FFT')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# 편의 함수들
def create_structure_system(phys_params: PhysicalParameters) -> TwoDOFStructure:
    """구조 시스템 생성"""
    return TwoDOFStructure(phys_params)

def perform_flutter_analysis(structure: TwoDOFStructure, 
                           U_range: np.ndarray) -> Dict:
    """플러터 해석 수행"""
    analyzer = FlutterAnalysis(structure)
    flutter_points = analyzer.find_flutter_speed(U_range)
    
    print("플러터 해석 결과:")
    for fp in flutter_points:
        print(f"Mode {fp['mode']+1}: U_flutter = {fp['U_flutter']:.2f} m/s, "
              f"f_flutter = {fp['frequency']:.2f} Hz")
    
    return {
        'flutter_points': flutter_points,
        'analyzer': analyzer
    }

if __name__ == "__main__":
    # 테스트 실행
    from config import DEFAULT_PHYS_PARAMS
    
    # 구조 시스템 생성
    structure = create_structure_system(DEFAULT_PHYS_PARAMS)
    
    # 자유 진동 시뮬레이션
    initial_conditions = {
        'h0': 0.01, 'theta0': 0.05,
        'h_dot0': 0.0, 'theta_dot0': 0.0
    }
    
    response = structure.solve_free_vibration(
        (0, 10), initial_conditions
    )
    
    # 응답 분석
    analyzer = ResponseAnalysis()
    analyzer.plot_response_analysis(
        response['time'], response['h'], response['theta'],
        np.zeros_like(response['time']), np.zeros_like(response['time'])
    )
    
    # 플러터 해석
    U_range = np.linspace(1, 50, 100)
    flutter_results = perform_flutter_analysis(structure, U_range)