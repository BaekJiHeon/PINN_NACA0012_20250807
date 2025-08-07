"""
NACA 0012 Flutter PINN - 신경망 모델 아키텍처
Physics-Informed Neural Networks for 2-DOF Flutter Analysis
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional, List, Dict
from config import PINNConfig

class FourierFeatures(nn.Module):
    """Fourier Features 임베딩 레이어"""
    
    def __init__(self, input_dim: int, num_modes: int = 64, sigma: float = 10.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_modes = num_modes
        
        # 가우시안 랜덤 매트릭스 B (고정)
        B = torch.randn(input_dim, num_modes) * sigma
        self.register_buffer('B', B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 [batch_size, input_dim]
        Returns:
            Fourier features [batch_size, 2*num_modes]
        """
        x_proj = 2 * math.pi * torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SIRENLayer(nn.Module):
    """SIREN (Sinusoidal Representation Networks) 레이어"""
    
    def __init__(self, input_dim: int, output_dim: int, w0: float = 1.0, 
                 is_first: bool = False, is_final: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w0 = w0
        self.is_first = is_first
        self.is_final = is_final
        
        self.linear = nn.Linear(input_dim, output_dim)
        self._initialize_weights()
        
    def _initialize_weights(self):
        """SIREN 초기화 방법"""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.input_dim, 1 / self.input_dim)
            else:
                bound = math.sqrt(6 / self.input_dim) / self.w0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_final:
            return self.linear(x)  # 마지막 레이어는 선형
        else:
            return torch.sin(self.w0 * self.linear(x))

class PINNModel(nn.Module):
    """NACA 0012 Flutter용 Physics-Informed Neural Network"""
    
    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim   # (t, x, y)
        self.output_dim = config.output_dim # (u, v, p)
        
        # 활성화 함수에 따른 네트워크 구성
        if config.activation == "fourier":
            self._build_fourier_network()
        elif config.activation == "siren":
            self._build_siren_network()
        else:
            self._build_standard_network()
            
        # 출력 스케일링을 위한 매개변수
        self.register_parameter('output_scale', 
                               nn.Parameter(torch.ones(self.output_dim)))
        
    def _build_fourier_network(self):
        """Fourier Features 기반 네트워크 구성"""
        self.fourier_features = FourierFeatures(
            self.input_dim, self.config.fourier_modes
        )
        
        feature_dim = 2 * self.config.fourier_modes
        layers = [nn.Linear(feature_dim, self.config.num_neurons)]
        
        for _ in range(self.config.num_layers - 1):
            layers.extend([
                nn.Tanh(),
                nn.Linear(self.config.num_neurons, self.config.num_neurons)
            ])
        
        layers.extend([
            nn.Tanh(),
            nn.Linear(self.config.num_neurons, self.output_dim)
        ])
        
        self.network = nn.Sequential(*layers)
        
    def _build_siren_network(self):
        """SIREN 네트워크 구성"""
        layers = []
        
        # 첫 번째 레이어
        layers.append(SIRENLayer(
            self.input_dim, self.config.num_neurons,
            w0=self.config.siren_w0, is_first=True
        ))
        
        # 중간 레이어들
        for _ in range(self.config.num_layers - 1):
            layers.append(SIRENLayer(
                self.config.num_neurons, self.config.num_neurons,
                w0=self.config.siren_w0
            ))
        
        # 출력 레이어
        layers.append(SIRENLayer(
            self.config.num_neurons, self.output_dim,
            w0=self.config.siren_w0, is_final=True
        ))
        
        self.network = nn.Sequential(*layers)
        
    def _build_standard_network(self):
        """표준 활성화 함수 네트워크 구성"""
        layers = [nn.Linear(self.input_dim, self.config.num_neurons)]
        
        # 활성화 함수 선택
        if self.config.activation == "tanh":
            activation = nn.Tanh()
        elif self.config.activation == "relu":
            activation = nn.ReLU()
        elif self.config.activation == "gelu":
            activation = nn.GELU()
        elif self.config.activation == "swish":
            activation = nn.SiLU()
        else:
            activation = nn.Tanh()  # 기본값
        
        for _ in range(self.config.num_layers):
            layers.extend([
                activation,
                nn.Linear(self.config.num_neurons, self.config.num_neurons)
            ])
        
        layers.extend([
            activation,
            nn.Linear(self.config.num_neurons, self.output_dim)
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Xavier 초기화
        self._xavier_init()
    
    def _xavier_init(self):
        """Xavier 정규 초기화"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        Args:
            x: 입력 텐서 [batch_size, 3] (t, x, y)
        Returns:
            output: 출력 텐서 [batch_size, 3] (u, v, p)
        """
        if self.config.activation == "fourier":
            features = self.fourier_features(x)
            output = self.network(features)
        else:
            output = self.network(x)
        
        # 출력 스케일링 적용
        output = output * self.output_scale
        
        return output
    
    def predict_flow_field(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """유동장 예측 및 구성 요소별 분해"""
        output = self.forward(x)
        
        return {
            'u': output[:, 0:1],  # x 방향 속도
            'v': output[:, 1:2],  # y 방향 속도  
            'p': output[:, 2:3]   # 압력
        }
    
    def compute_gradients(self, x: torch.Tensor, 
                         output: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """자동 미분을 이용한 기울기 계산"""
        if output is None:
            output = self.forward(x)
        
        u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]
        
        # 1차 편미분 계산
        gradients = {}
        
        # u의 편미분
        u_t = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 0:1]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 1:2]
        u_y = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 2:3]
        
        # v의 편미분
        v_t = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 0:1]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 1:2]
        v_y = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 2:3]
        
        # p의 편미분
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:, 1:2]
        p_y = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:, 2:3]
        
        gradients.update({
            'u_t': u_t, 'u_x': u_x, 'u_y': u_y,
            'v_t': v_t, 'v_x': v_x, 'v_y': v_y,
            'p_x': p_x, 'p_y': p_y
        })
        
        # 2차 편미분 (라플라시안)
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 1:2]
        u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 2:3]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0][:, 1:2]
        v_yy = torch.autograd.grad(v_y.sum(), x, create_graph=True)[0][:, 2:3]
        
        gradients.update({
            'u_xx': u_xx, 'u_yy': u_yy,
            'v_xx': v_xx, 'v_yy': v_yy
        })
        
        return gradients
    
    def compute_vorticity(self, x: torch.Tensor) -> torch.Tensor:
        """와도 계산 (ω = ∂v/∂x - ∂u/∂y)"""
        output = self.forward(x)
        u, v = output[:, 0:1], output[:, 1:2]
        
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 1:2]
        u_y = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 2:3]
        
        vorticity = v_x - u_y
        return vorticity
    
    def compute_q_criterion(self, x: torch.Tensor) -> torch.Tensor:
        """Q-criterion 계산 (와류 식별)"""
        output = self.forward(x)
        u, v = output[:, 0:1], output[:, 1:2]
        
        # 속도 기울기 텐서
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 1:2]
        u_y = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 2:3]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 1:2]
        v_y = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 2:3]
        
        # 변형률 텐서와 회전 텐서
        S11, S22 = u_x, v_y
        S12 = 0.5 * (u_y + v_x)
        W12 = 0.5 * (v_x - u_y)
        
        # Q = 0.5 * (||Ω||² - ||S||²)
        omega_norm_sq = 2 * W12**2
        strain_norm_sq = S11**2 + S22**2 + 2 * S12**2
        
        Q = 0.5 * (omega_norm_sq - strain_norm_sq)
        return Q
    
    def get_model_summary(self) -> Dict:
        """모델 요약 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': f"{self.config.activation.upper()} PINN",
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layers': self.config.num_layers,
            'neurons_per_layer': self.config.num_neurons,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'activation': self.config.activation
        }

class EnsemblePINN(nn.Module):
    """앙상블 PINN (불확실성 정량화용)"""
    
    def __init__(self, configs: List[PINNConfig], num_models: int = 3):
        super().__init__()
        self.num_models = num_models
        
        # 여러 PINN 모델 생성
        self.models = nn.ModuleList([
            PINNModel(config) for config in configs[:num_models]
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        앙상블 예측
        Returns:
            mean: 평균 예측값
            std: 표준편차 (불확실성)
        """
        predictions = torch.stack([model(x) for model in self.models], dim=0)
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """불확실성을 포함한 예측"""
        mean, std = self.forward(x)
        
        return {
            'u_mean': mean[:, 0:1], 'u_std': std[:, 0:1],
            'v_mean': mean[:, 1:2], 'v_std': std[:, 1:2],
            'p_mean': mean[:, 2:3], 'p_std': std[:, 2:3]
        }

def create_pinn_model(config: PINNConfig, ensemble: bool = False) -> nn.Module:
    """PINN 모델 생성 함수"""
    if ensemble:
        # 약간 다른 설정으로 앙상블 모델 생성
        configs = []
        for i in range(3):
            ensemble_config = PINNConfig(
                num_layers=config.num_layers + i,
                num_neurons=config.num_neurons + 16*i,
                activation=config.activation,
                fourier_modes=config.fourier_modes + 8*i,
                siren_w0=config.siren_w0 * (1 + 0.1*i)
            )
            configs.append(ensemble_config)
        return EnsemblePINN(configs)
    else:
        return PINNModel(config)

# 모델 저장/로딩 유틸리티
def save_model(model: nn.Module, filepath: str, metadata: Optional[Dict] = None):
    """모델 저장"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': model.config if hasattr(model, 'config') else None,
        'metadata': metadata or {}
    }
    torch.save(save_dict, filepath)
    print(f"✅ 모델 저장 완료: {filepath}")

def load_model(filepath: str, config: Optional[PINNConfig] = None) -> nn.Module:
    """모델 로딩"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    if config is None:
        config = checkpoint.get('model_config')
        if config is None:
            raise ValueError("Config가 체크포인트에 없습니다. config 매개변수를 제공해주세요.")
    
    model = PINNModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ 모델 로딩 완료: {filepath}")
    return model