"""
NACA 0012 Flutter PINN 프로젝트 설정
Physics-Informed Neural Networks for 2-DOF Flutter Analysis
"""

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class PhysicalParameters:
    """물리적 파라미터 클래스"""
    # 실제 chord length (m)
    C_phys: float = 0.156
    
    # 구조 파라미터 (UDF 값)
    m: float = 2.5              # 질량 (kg)
    I_alpha: float = 0.00135    # 관성모멘트 (kg·m²)
    c_h: float = 0.494974       # heave 댐핑 (N·s/m)
    k_h: float = 500.0          # heave 강성 (N/m)
    c_alpha: float = 0.000321   # pitch 댐핑 (N·m·s)
    k_alpha: float = 0.3        # pitch 강성 (N·m/rad)
    x_ref: float = 0.19         # 기준점 위치 (C_phys 기준)
    
    # 유동 파라미터
    U_inf: float = 1.0          # 자유류 속도 (비차원)
    rho: float = 1.0           # 밀도 (비차원)
    
    def __post_init__(self):
        """후처리: 기준점을 물리 단위로 변환"""
        self.x_ref_phys = self.x_ref * self.C_phys

@dataclass 
class DomainParameters:
    """도메인 파라미터 클래스"""
    # 물리적 도메인 경계 (m)
    x_min_phys: float = -0.45
    x_max_phys: float = 0.45
    y_min_phys: float = -0.35
    y_max_phys: float = 0.35
    
    # 원점 (m)
    x0_phys: float = 0.0
    y0_phys: float = 0.0
    
    # 시간 도메인
    t_start: float = 0.0
    t_end: float = 10.0
    
    def get_nondim_bounds(self, C_phys: float) -> Dict[str, float]:
        """비차원 경계 반환"""
        return {
            'x_min': (self.x_min_phys - self.x0_phys) / C_phys,
            'x_max': (self.x_max_phys - self.x0_phys) / C_phys,
            'y_min': (self.y_min_phys - self.y0_phys) / C_phys,
            'y_max': (self.y_max_phys - self.y0_phys) / C_phys
        }

@dataclass
class PINNConfig:
    """PINN 모델 설정"""
    # 네트워크 아키텍처
    num_layers: int = 10
    num_neurons: int = 128
    activation: str = "tanh"  # "tanh", "siren", "fourier"
    
    # 입력/출력 차원
    input_dim: int = 3   # (t, x, y)
    output_dim: int = 3  # (u, v, p)
    
    # Fourier/SIREN 설정
    fourier_modes: int = 64
    siren_w0: float = 1.0
    
    # 초기화
    init_method: str = "xavier_normal"

@dataclass
class TrainingConfig:
    """학습 설정"""
    # 학습률 및 옵티마이저
    adam_lr: float = 1e-3
    adam_epochs: int = 5000
    lbfgs_epochs: int = 2000
    
    # 배치 크기
    batch_size: int = 2048
    
    # 손실 가중치 (FSI 완전 비활성화로 순수 유동 PINN)
    lambda_data: float = 1.0
    lambda_pde: float = 1.0
    lambda_bc: float = 0.5      # 경계조건 적당히 감소
    lambda_fsi: float = 0.0     # FSI 완전 비활성화
    
    # 샘플링 비율
    boundary_layer_ratio: float = 0.25
    wake_core_ratio: float = 0.35
    domain_ratio: float = 0.40
    
    # 경계층 및 후류 정의
    boundary_layer_thickness: float = 0.008  # 물리 단위 (m)
    wake_x_start: float = -0.1               # 물리 단위 (m)
    wake_x_end: float = 0.25                 # 물리 단위 (m)
    wake_y_thickness: float = 0.08           # 물리 단위 (m)
    
    # AMP 및 다중 GPU
    use_amp: bool = True
    use_multi_gpu: bool = True

@dataclass
class FileConfig:
    """파일 경로 설정"""
    cfd_csv_path: str = "Wind_turnel_DATA/mesh center postion"
    mesh_csv_path: str = "Wind_turnel_DATA/Node postion" 
    damping_csv_path: str = "Wind_turnel_DATA/Damping_data.csv"
    
    output_dir: str = "results"
    model_save_path: str = "results/model.pt"
    
    # 출력 파일명
    loss_curve_file: str = "results/loss_curve.png"
    animation_file: str = "results/swirling_strength.mp4"
    fft_analysis_file: str = "results/cl_cm_fft.png"

def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="NACA 0012 Flutter PINN")
    
    # 물리 파라미터
    parser.add_argument("--Re", type=float, default=1000.0, 
                       help="Reynolds number")
    parser.add_argument("--coord_sys", choices=["lab", "body"], default="lab",
                       help="Coordinate system (lab or body)")
    
    # 도메인
    parser.add_argument("--x_min", type=float, default=-0.45,
                       help="Domain x minimum (m)")
    parser.add_argument("--x_max", type=float, default=0.45,
                       help="Domain x maximum (m)")
    parser.add_argument("--y_min", type=float, default=-0.35,
                       help="Domain y minimum (m)")
    parser.add_argument("--y_max", type=float, default=0.35,
                       help="Domain y maximum (m)")
    
    # 학습 파라미터
    parser.add_argument("--adam_epochs", type=int, default=5000,
                       help="Adam optimizer epochs")
    parser.add_argument("--lbfgs_epochs", type=int, default=2000,
                       help="LBFGS optimizer epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2048,
                       help="Batch size")
    
    # 네트워크 구조
    parser.add_argument("--num_layers", type=int, default=10,
                       help="Number of hidden layers")
    parser.add_argument("--num_neurons", type=int, default=128,
                       help="Neurons per layer")
    parser.add_argument("--activation", choices=["tanh", "siren", "fourier"],
                       default="tanh", help="Activation function")
    
    # 파일 경로
    parser.add_argument("--cfd_csv", type=str, default="Wind_turnel_DATA/mesh center postion",
                       help="CFD data CSV path")
    parser.add_argument("--mesh_csv", type=str, default="Wind_turnel_DATA/Node postion",
                       help="Mesh data CSV path")
    parser.add_argument("--damping_csv", type=str, default="Wind_turnel_DATA/Damping_data.csv",
                       help="Damping data CSV path")
    
    # 고급 옵션
    parser.add_argument("--inverse_id", action="store_true",
                       help="Learn damping/stiffness parameters")
    parser.add_argument("--turbulence", action="store_true",
                       help="Include turbulence model")
    
    return parser.parse_args()

def create_config_from_args(args) -> Tuple[PhysicalParameters, DomainParameters, 
                                         PINNConfig, TrainingConfig, FileConfig]:
    """명령행 인수로부터 설정 객체들 생성"""
    
    # 물리 파라미터
    phys_params = PhysicalParameters()
    phys_params.nu = 1.0 / args.Re  # 동점성계수
    
    # 도메인 파라미터
    domain_params = DomainParameters(
        x_min_phys=args.x_min,
        x_max_phys=args.x_max,
        y_min_phys=args.y_min,
        y_max_phys=args.y_max
    )
    
    # PINN 설정
    pinn_config = PINNConfig(
        num_layers=args.num_layers,
        num_neurons=args.num_neurons,
        activation=args.activation
    )
    
    # 학습 설정
    training_config = TrainingConfig(
        adam_lr=args.lr,
        adam_epochs=args.adam_epochs,
        lbfgs_epochs=args.lbfgs_epochs,
        batch_size=args.batch_size
    )
    
    # 파일 설정
    file_config = FileConfig(
        cfd_csv_path=args.cfd_csv,
        mesh_csv_path=args.mesh_csv,
        damping_csv_path=args.damping_csv
    )
    
    return phys_params, domain_params, pinn_config, training_config, file_config

# 기본 설정 인스턴스
DEFAULT_PHYS_PARAMS = PhysicalParameters()
DEFAULT_DOMAIN_PARAMS = DomainParameters()
DEFAULT_PINN_CONFIG = PINNConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_FILE_CONFIG = FileConfig()