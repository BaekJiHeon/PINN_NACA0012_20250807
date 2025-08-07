"""
NACA 0012 Flutter PINN - 샘플링 전략 모듈
Physics-Informed Neural Networks for 2-DOF Flutter Analysis

고급 샘플링 전략:
- Latin Hypercube Sampling (LHS)
- 적응적 샘플링
- 경계층/후류 집중 샘플링
- 에어포일 표면 샘플링
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Callable
from pyDOE2 import lhs
from sklearn.cluster import KMeans
from config import PhysicalParameters, DomainParameters, TrainingConfig
from boundary_conditions import NACAGeometry
import matplotlib.pyplot as plt

class LatinHypercubeSampler:
    """Latin Hypercube 샘플링"""
    
    def __init__(self, domain_bounds: Dict[str, float], 
                 time_bounds: Tuple[float, float]):
        self.domain_bounds = domain_bounds
        self.time_bounds = time_bounds
        
    def sample(self, n_samples: int, seed: Optional[int] = None) -> torch.Tensor:
        """
        3D Latin Hypercube 샘플링 (t, x, y)
        
        Args:
            n_samples: 샘플 수
            seed: 난수 시드
            
        Returns:
            samples: [n_samples, 3] (t, x, y)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # LHS 샘플링 (0-1 범위)
        lhs_samples = lhs(3, samples=n_samples, criterion='maximin')
        
        # 실제 도메인으로 스케일링
        t_min, t_max = self.time_bounds
        samples = np.zeros_like(lhs_samples)
        
        samples[:, 0] = lhs_samples[:, 0] * (t_max - t_min) + t_min  # t
        samples[:, 1] = lhs_samples[:, 1] * (self.domain_bounds['x_max'] - 
                                           self.domain_bounds['x_min']) + self.domain_bounds['x_min']  # x
        samples[:, 2] = lhs_samples[:, 2] * (self.domain_bounds['y_max'] - 
                                           self.domain_bounds['y_min']) + self.domain_bounds['y_min']  # y
        
        return torch.tensor(samples, dtype=torch.float32)

class AdaptiveSampler:
    """적응적 샘플링 (잔차 기반)"""
    
    def __init__(self, initial_samples: torch.Tensor):
        self.samples = initial_samples.clone()
        self.residuals = None
        self.history = []
        
    def update_residuals(self, residuals: torch.Tensor):
        """잔차 업데이트"""
        self.residuals = residuals.detach().clone()
        self.history.append(residuals.mean().item())
        
    def adaptive_refinement(self, model: torch.nn.Module, 
                          n_new_samples: int,
                          refinement_ratio: float = 0.8) -> torch.Tensor:
        """
        잔차 기반 적응적 세분화
        
        Args:
            model: PINN 모델
            n_new_samples: 추가할 샘플 수
            refinement_ratio: 상위 비율 (높은 잔차 영역)
        """
        if self.residuals is None:
            raise ValueError("잔차가 업데이트되지 않았습니다.")
        
        # 높은 잔차를 가진 점들 식별
        n_high_residual = int(len(self.residuals) * refinement_ratio)
        high_residual_idx = torch.topk(self.residuals, n_high_residual).indices
        
        # 높은 잔차 영역 주변에서 새 샘플 생성
        high_residual_points = self.samples[high_residual_idx]
        
        new_samples = []
        for _ in range(n_new_samples):
            # 랜덤하게 높은 잔차 점 선택
            base_point = high_residual_points[np.random.randint(len(high_residual_points))]
            
            # 주변에 가우시안 노이즈 추가
            noise_std = 0.05  # 도메인 크기의 5%
            noise = torch.randn(3) * noise_std
            new_point = base_point + noise
            
            # 도메인 경계 내로 클리핑
            new_point = self._clip_to_domain(new_point)
            new_samples.append(new_point)
        
        new_samples = torch.stack(new_samples)
        
        # 기존 샘플에 추가
        self.samples = torch.cat([self.samples, new_samples], dim=0)
        
        return new_samples
    
    def _clip_to_domain(self, point: torch.Tensor) -> torch.Tensor:
        """도메인 경계 내로 클리핑"""
        # 구현 필요: 도메인 경계 정보 필요
        return point
    
    def get_refinement_history(self) -> List[float]:
        """세분화 이력 반환"""
        return self.history

class StratifiedSampler:
    """계층화 샘플링 (경계층, 후류, 일반 영역)"""
    
    def __init__(self, config: TrainingConfig, 
                 domain_bounds: Dict[str, float]):
        self.config = config
        self.domain_bounds = domain_bounds
        
        # 물리적 단위로 정의된 특수 영역
        self.boundary_layer_thickness = config.boundary_layer_thickness
        self.wake_x_start = config.wake_x_start
        self.wake_x_end = config.wake_x_end
        self.wake_y_thickness = config.wake_y_thickness
        
    def identify_region(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """점들의 영역 분류"""
        x = points[:, 1]
        y = points[:, 2]
        
        # 경계층 영역 (에어포일 주변)
        boundary_layer_mask = torch.abs(y) < (self.boundary_layer_thickness / 0.156)  # 비차원화
        
        # 후류 영역 
        wake_mask = ((x >= self.wake_x_start / 0.156) & 
                    (x <= self.wake_x_end / 0.156) & 
                    (torch.abs(y) < self.wake_y_thickness / 0.156))
        
        # 일반 영역
        general_mask = ~(boundary_layer_mask | wake_mask)
        
        return {
            'boundary_layer': boundary_layer_mask,
            'wake': wake_mask,
            'general': general_mask
        }
    
    def stratified_sampling(self, n_total: int, 
                          time_bounds: Tuple[float, float]) -> Dict[str, torch.Tensor]:
        """계층화 샘플링 수행"""
        
        # 각 영역별 샘플 수
        n_boundary = int(n_total * self.config.boundary_layer_ratio)
        n_wake = int(n_total * self.config.wake_core_ratio)
        n_general = n_total - n_boundary - n_wake
        
        samples = {}
        
        # 1. 경계층 영역 샘플링
        samples['boundary_layer'] = self._sample_boundary_layer(n_boundary, time_bounds)
        
        # 2. 후류 영역 샘플링
        samples['wake'] = self._sample_wake_region(n_wake, time_bounds)
        
        # 3. 일반 영역 샘플링
        samples['general'] = self._sample_general_domain(n_general, time_bounds)
        
        # 전체 샘플 결합
        all_samples = torch.cat([
            samples['boundary_layer'],
            samples['wake'], 
            samples['general']
        ], dim=0)
        
        samples['all'] = all_samples
        
        return samples
    
    def _sample_boundary_layer(self, n_samples: int, 
                             time_bounds: Tuple[float, float]) -> torch.Tensor:
        """경계층 영역 샘플링"""
        t_min, t_max = time_bounds
        
        # 시간 좌표
        t = torch.rand(n_samples) * (t_max - t_min) + t_min
        
        # x 좌표 (에어포일 주변)
        x = torch.rand(n_samples) * (1.2 - (-0.2)) + (-0.2)  # 에어포일 앞뒤
        
        # y 좌표 (경계층 두께 내)
        bl_thickness_nd = self.boundary_layer_thickness / 0.156  # 비차원
        y = (torch.rand(n_samples) - 0.5) * 2 * bl_thickness_nd
        
        return torch.stack([t, x, y], dim=1)
    
    def _sample_wake_region(self, n_samples: int,
                           time_bounds: Tuple[float, float]) -> torch.Tensor:
        """후류 영역 샘플링"""
        t_min, t_max = time_bounds
        
        # 시간 좌표
        t = torch.rand(n_samples) * (t_max - t_min) + t_min
        
        # x 좌표 (후류 영역)
        x_start_nd = self.wake_x_start / 0.156
        x_end_nd = self.wake_x_end / 0.156
        x = torch.rand(n_samples) * (x_end_nd - x_start_nd) + x_start_nd
        
        # y 좌표 (후류 두께 내)
        wake_thickness_nd = self.wake_y_thickness / 0.156
        y = (torch.rand(n_samples) - 0.5) * 2 * wake_thickness_nd
        
        return torch.stack([t, x, y], dim=1)
    
    def _sample_general_domain(self, n_samples: int,
                             time_bounds: Tuple[float, float]) -> torch.Tensor:
        """일반 도메인 샘플링"""
        lhs_sampler = LatinHypercubeSampler(self.domain_bounds, time_bounds)
        return lhs_sampler.sample(n_samples)

class AirfoilSurfaceSampler:
    """에어포일 표면 샘플링"""
    
    def __init__(self, geometry: NACAGeometry):
        self.geometry = geometry
        
    def sample_surface_points(self, n_points: int, 
                            t: float, h: float, theta: float,
                            x_ref: float = 0.19) -> torch.Tensor:
        """특정 시간에서 표면점 샘플링"""
        
        # 변환된 에어포일 좌표
        x_surf, y_surf = self.geometry.transform_airfoil(h, theta, x_ref)
        
        # 균등 간격 또는 곡률 기반 샘플링
        if n_points >= len(x_surf):
            # 모든 점 사용
            indices = np.arange(len(x_surf))
        else:
            # 곡률 기반 샘플링 (앞전/뒷전 집중)
            indices = self._curvature_based_sampling(x_surf, y_surf, n_points)
        
        # 인덱스 범위 확인
        indices = indices[indices < len(x_surf)]
        if len(indices) == 0:
            indices = np.array([0])  # 최소 하나의 점은 유지
            
        x_sampled = x_surf[indices]
        y_sampled = y_surf[indices]
        t_sampled = np.full_like(x_sampled, t)
        
        return torch.tensor(
            np.column_stack([t_sampled, x_sampled, y_sampled]),
            dtype=torch.float32
        )
    
    def _curvature_based_sampling(self, x: np.ndarray, y: np.ndarray, 
                                n_points: int) -> np.ndarray:
        """곡률 기반 샘플링"""
        
        # n_points가 전체 점 개수보다 크면 모든 점 사용
        if n_points >= len(x):
            return np.arange(len(x))
            
        # 곡률 계산 (단순화된 방법)
        dx = np.gradient(x)
        dy = np.gradient(y)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5
        
        # 곡률이 높은 지점에 더 많은 점 배치
        curvature_weights = curvature / curvature.sum()
        
        # 가중 샘플링 (replace=False 안전하게)
        indices = np.random.choice(
            len(x), size=min(n_points, len(x)), 
            p=curvature_weights, replace=False
        )
        
        return np.sort(indices)
    
    def sample_time_series(self, time_points: np.ndarray,
                          h_series: np.ndarray, theta_series: np.ndarray,
                          n_surface_points: int = 50) -> torch.Tensor:
        """시계열 표면점 샘플링"""
        
        all_surface_points = []
        
        for i, (t, h, theta) in enumerate(zip(time_points, h_series, theta_series)):
            surface_points = self.sample_surface_points(
                n_surface_points, t, h, theta
            )
            all_surface_points.append(surface_points)
        
        return torch.cat(all_surface_points, dim=0)

class MultiScaleSampler:
    """다중 스케일 샘플링"""
    
    def __init__(self, domain_bounds: Dict[str, float]):
        self.domain_bounds = domain_bounds
        self.scales = [1.0, 0.5, 0.25, 0.1]  # 다양한 스케일
        
    def multiscale_sampling(self, n_samples_per_scale: int,
                          time_bounds: Tuple[float, float]) -> Dict[str, torch.Tensor]:
        """다중 스케일 샘플링"""
        
        scale_samples = {}
        
        for scale in self.scales:
            # 스케일에 따른 도메인 축소
            scaled_bounds = self._scale_domain(scale)
            
            # 해당 스케일에서 샘플링
            lhs_sampler = LatinHypercubeSampler(scaled_bounds, time_bounds)
            samples = lhs_sampler.sample(n_samples_per_scale)
            
            scale_samples[f'scale_{scale}'] = samples
        
        # 모든 스케일 결합
        all_samples = torch.cat(list(scale_samples.values()), dim=0)
        scale_samples['all'] = all_samples
        
        return scale_samples
    
    def _scale_domain(self, scale: float) -> Dict[str, float]:
        """도메인 스케일링"""
        center_x = (self.domain_bounds['x_max'] + self.domain_bounds['x_min']) / 2
        center_y = (self.domain_bounds['y_max'] + self.domain_bounds['y_min']) / 2
        
        width_x = (self.domain_bounds['x_max'] - self.domain_bounds['x_min']) * scale
        width_y = (self.domain_bounds['y_max'] - self.domain_bounds['y_min']) * scale
        
        return {
            'x_min': center_x - width_x / 2,
            'x_max': center_x + width_x / 2,
            'y_min': center_y - width_y / 2,
            'y_max': center_y + width_y / 2
        }

class CompositeSampler:
    """통합 샘플러"""
    
    def __init__(self, phys_params: PhysicalParameters,
                 domain_params: DomainParameters,
                 training_config: TrainingConfig):
        
        self.phys_params = phys_params
        self.domain_params = domain_params
        self.training_config = training_config
        
        # 도메인 경계
        self.domain_bounds = domain_params.get_nondim_bounds(phys_params.C_phys)
        self.time_bounds = (domain_params.t_start, domain_params.t_end)
        
        # 개별 샘플러들
        self.lhs_sampler = LatinHypercubeSampler(self.domain_bounds, self.time_bounds)
        self.stratified_sampler = StratifiedSampler(training_config, self.domain_bounds)
        self.multiscale_sampler = MultiScaleSampler(self.domain_bounds)
        
        # 에어포일 관련
        self.geometry = NACAGeometry()
        self.surface_sampler = AirfoilSurfaceSampler(self.geometry)
        
        # 적응적 샘플러 (나중에 초기화)
        self.adaptive_sampler = None
        
    def generate_training_samples(self, n_collocation: int,
                                n_boundary: int,
                                n_surface: int,
                                strategy: str = "stratified") -> Dict[str, torch.Tensor]:
        """학습용 샘플 생성"""
        
        samples = {}
        
        # 1. 콜로케이션 점 (PDE 잔차용)
        if strategy == "stratified":
            collocation_samples = self.stratified_sampler.stratified_sampling(
                n_collocation, self.time_bounds
            )
            samples['collocation'] = collocation_samples['all']
            samples['collocation_by_region'] = {
                'boundary_layer': collocation_samples['boundary_layer'],
                'wake': collocation_samples['wake'],
                'general': collocation_samples['general']
            }
        elif strategy == "lhs":
            samples['collocation'] = self.lhs_sampler.sample(n_collocation)
        elif strategy == "multiscale":
            multiscale_samples = self.multiscale_sampler.multiscale_sampling(
                n_collocation // len(self.multiscale_sampler.scales), self.time_bounds
            )
            samples['collocation'] = multiscale_samples['all']
        
        # 2. 경계점 (경계조건용)
        samples['boundary'] = {}
        boundary_types = ['inlet', 'outlet', 'top', 'bottom']
        
        for bc_type in boundary_types:
            # 각 경계별 시간 샘플링
            t_boundary = torch.rand(n_boundary) * (self.time_bounds[1] - self.time_bounds[0]) + self.time_bounds[0]
            
            if bc_type == 'inlet':
                x_bc = torch.full((n_boundary,), self.domain_bounds['x_min'])
                y_bc = torch.linspace(self.domain_bounds['y_min'], 
                                    self.domain_bounds['y_max'], n_boundary)
            elif bc_type == 'outlet':
                x_bc = torch.full((n_boundary,), self.domain_bounds['x_max'])
                y_bc = torch.linspace(self.domain_bounds['y_min'], 
                                    self.domain_bounds['y_max'], n_boundary)
            elif bc_type == 'top':
                x_bc = torch.linspace(self.domain_bounds['x_min'], 
                                    self.domain_bounds['x_max'], n_boundary)
                y_bc = torch.full((n_boundary,), self.domain_bounds['y_max'])
            elif bc_type == 'bottom':
                x_bc = torch.linspace(self.domain_bounds['x_min'], 
                                    self.domain_bounds['x_max'], n_boundary)
                y_bc = torch.full((n_boundary,), self.domain_bounds['y_min'])
            
            samples['boundary'][bc_type] = torch.stack([t_boundary, x_bc, y_bc], dim=1)
        
        # 3. 표면점 (에어포일 경계조건용) - 동적 생성 필요
        # 여기서는 정적 표면점만 생성 (실제로는 시간에 따라 변함)
        t_surface = torch.rand(n_surface) * (self.time_bounds[1] - self.time_bounds[0]) + self.time_bounds[0]
        static_surface = self.surface_sampler.sample_surface_points(
            n_surface, 0.0, 0.0, 0.0  # 정적 상태
        )
        static_surface[:, 0] = t_surface  # 시간 좌표 업데이트
        samples['surface'] = static_surface
        
        return samples
    
    def adaptive_resample(self, model: torch.nn.Module,
                        current_samples: torch.Tensor,
                        n_new_samples: int) -> torch.Tensor:
        """적응적 재샘플링"""
        
        if self.adaptive_sampler is None:
            self.adaptive_sampler = AdaptiveSampler(current_samples)
        
        # 모델로 잔차 계산 (구현 필요)
        with torch.no_grad():
            # 여기서는 단순히 랜덤 잔차로 대체
            residuals = torch.rand(len(current_samples))
            
        self.adaptive_sampler.update_residuals(residuals)
        new_samples = self.adaptive_sampler.adaptive_refinement(
            model, n_new_samples
        )
        
        return new_samples
    
    def visualize_samples(self, samples: Dict[str, torch.Tensor],
                        time_slice: float = 0.0,
                        save_path: Optional[str] = None):
        """샘플 분포 시각화"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 시간 슬라이스에서의 점들 추출
        def extract_time_slice(points: torch.Tensor, t: float, tol: float = 0.1):
            mask = torch.abs(points[:, 0] - t) < tol
            return points[mask]
        
        # 콜로케이션 점
        if 'collocation' in samples:
            coll_points = extract_time_slice(samples['collocation'], time_slice)
            axes[0, 0].scatter(coll_points[:, 1], coll_points[:, 2], 
                             s=1, alpha=0.6, c='blue', label='Collocation')
            axes[0, 0].set_title('Collocation Points')
        
        # 경계점
        if 'boundary' in samples:
            colors = {'inlet': 'red', 'outlet': 'green', 'top': 'orange', 'bottom': 'purple'}
            for bc_type, bc_points in samples['boundary'].items():
                bc_slice = extract_time_slice(bc_points, time_slice)
                if len(bc_slice) > 0:
                    axes[0, 1].scatter(bc_slice[:, 1], bc_slice[:, 2], 
                                     s=10, c=colors.get(bc_type, 'black'), 
                                     label=bc_type, alpha=0.7)
            axes[0, 1].set_title('Boundary Points')
            axes[0, 1].legend()
        
        # 표면점
        if 'surface' in samples:
            surf_points = extract_time_slice(samples['surface'], time_slice)
            axes[1, 0].scatter(surf_points[:, 1], surf_points[:, 2], 
                             s=5, c='black', label='Airfoil Surface')
            axes[1, 0].set_title('Surface Points')
            axes[1, 0].legend()
        
        # 모든 점 종합
        all_points = []
        labels = []
        colors = []
        
        if 'collocation' in samples:
            coll_slice = extract_time_slice(samples['collocation'], time_slice)
            all_points.append(coll_slice)
            labels.extend(['Collocation'] * len(coll_slice))
            colors.extend(['blue'] * len(coll_slice))
        
        if 'surface' in samples:
            surf_slice = extract_time_slice(samples['surface'], time_slice)
            all_points.append(surf_slice)
            labels.extend(['Surface'] * len(surf_slice))
            colors.extend(['red'] * len(surf_slice))
        
        if all_points:
            all_pts = torch.cat(all_points, dim=0)
            axes[1, 1].scatter(all_pts[:, 1], all_pts[:, 2], 
                             s=1, c=colors, alpha=0.6)
            axes[1, 1].set_title('All Sampling Points')
        
        # 설정 적용
        for ax in axes.flat:
            ax.set_xlim(self.domain_bounds['x_min'], self.domain_bounds['x_max'])
            ax.set_ylim(self.domain_bounds['y_min'], self.domain_bounds['y_max'])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x/C')
            ax.set_ylabel('y/C')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# 편의 함수들
def create_composite_sampler(phys_params: PhysicalParameters,
                           domain_params: DomainParameters,
                           training_config: TrainingConfig) -> CompositeSampler:
    """통합 샘플러 생성"""
    return CompositeSampler(phys_params, domain_params, training_config)

def test_sampling_strategies():
    """샘플링 전략 테스트"""
    from config import DEFAULT_PHYS_PARAMS, DEFAULT_DOMAIN_PARAMS, DEFAULT_TRAINING_CONFIG
    
    sampler = create_composite_sampler(
        DEFAULT_PHYS_PARAMS, DEFAULT_DOMAIN_PARAMS, DEFAULT_TRAINING_CONFIG
    )
    
    # 다양한 샘플링 전략 테스트
    strategies = ['stratified', 'lhs', 'multiscale']
    
    for strategy in strategies:
        print(f"\n=== {strategy.upper()} 샘플링 테스트 ===")
        
        samples = sampler.generate_training_samples(
            n_collocation=1000,
            n_boundary=100,
            n_surface=200,
            strategy=strategy
        )
        
        print(f"콜로케이션 점: {len(samples['collocation'])}")
        print(f"경계점: {sum(len(bc) for bc in samples['boundary'].values())}")
        print(f"표면점: {len(samples['surface'])}")
        
        # 시각화
        sampler.visualize_samples(samples, save_path=f"results/sampling_{strategy}.png")

if __name__ == "__main__":
    test_sampling_strategies()