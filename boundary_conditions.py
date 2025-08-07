"""
NACA 0012 Flutter PINN - 경계조건 모듈
Physics-Informed Neural Networks for 2-DOF Flutter Analysis

상세한 경계조건 관리:
- NACA 0012 에어포일 형상
- 동적 표면점 추적
- 다양한 경계조건 타입
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Callable
from config import PhysicalParameters, DomainParameters
import matplotlib.pyplot as plt

class NACAGeometry:
    """NACA 0012 에어포일 형상 클래스"""
    
    def __init__(self, n_points: int = 200):
        self.n_points = n_points
        self.thickness_ratio = 0.12  # NACA 0012
        
        # 에어포일 좌표 생성
        self.x_airfoil, self.y_airfoil = self._generate_naca0012()
        
        # 법선 벡터 계산
        self.normals = self._compute_normals()
        
    def _generate_naca0012(self) -> Tuple[np.ndarray, np.ndarray]:
        """NACA 0012 에어포일 좌표 생성"""
        
        # 코사인 분포 (앞전 근처 점 집중)
        beta = np.linspace(0, np.pi, self.n_points // 2)
        x_c = 0.5 * (1 - np.cos(beta))
        
        # NACA 0012 두께 분포
        t = self.thickness_ratio
        y_t = 5 * t * (
            0.2969 * np.sqrt(x_c) - 
            0.1260 * x_c - 
            0.3516 * x_c**2 + 
            0.2843 * x_c**3 - 
            0.1015 * x_c**4
        )
        
        # 뒷전 닫기 (수정된 공식)
        y_t[-1] = 0.0
        
        # 상면과 하면
        x_upper = x_c
        y_upper = y_t
        x_lower = x_c[::-1]
        y_lower = -y_t[::-1]
        
        # 전체 윤곽선 (시계방향)
        x_airfoil = np.concatenate([x_upper, x_lower[1:]])  # 앞전 중복 제거
        y_airfoil = np.concatenate([y_upper, y_lower[1:]])
        
        return x_airfoil, y_airfoil
    
    def _compute_normals(self) -> np.ndarray:
        """표면 법선 벡터 계산"""
        n_pts = len(self.x_airfoil)
        normals = np.zeros((n_pts, 2))
        
        for i in range(n_pts):
            # 인접 점들
            i_prev = (i - 1) % n_pts
            i_next = (i + 1) % n_pts
            
            # 접선 벡터
            dx = self.x_airfoil[i_next] - self.x_airfoil[i_prev]
            dy = self.y_airfoil[i_next] - self.y_airfoil[i_prev]
            
            # 법선 벡터 (내부를 향하도록)
            norm = np.sqrt(dx**2 + dy**2)
            normals[i, 0] = dy / norm   # n_x
            normals[i, 1] = -dx / norm  # n_y
            
        return normals
    
    def transform_airfoil(self, h: float, theta: float, 
                         x_ref: float = 0.19) -> Tuple[np.ndarray, np.ndarray]:
        """
        에어포일 좌표 변환 (heave + pitch)
        
        Args:
            h: heave 변위
            theta: pitch 각도 (라디안)
            x_ref: 회전 중심 (chord 기준)
        """
        
        # 회전 중심 이동
        x_centered = self.x_airfoil - x_ref
        y_centered = self.y_airfoil
        
        # 회전 변환
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        x_rotated = x_centered * cos_theta - y_centered * sin_theta
        y_rotated = x_centered * sin_theta + y_centered * cos_theta
        
        # 회전 중심 복원 + heave 변위
        x_transformed = x_rotated + x_ref
        y_transformed = y_rotated + h
        
        return x_transformed, y_transformed
    
    def get_surface_points(self, n_sample: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """표면점 샘플링"""
        indices = np.linspace(0, len(self.x_airfoil)-1, n_sample, dtype=int)
        return self.x_airfoil[indices], self.y_airfoil[indices]
    
    def visualize(self, save_path: Optional[str] = None):
        """에어포일 형상 시각화"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.x_airfoil, self.y_airfoil, 'b-', linewidth=2, label='NACA 0012')
        ax.fill(self.x_airfoil, self.y_airfoil, alpha=0.3, color='lightblue')
        
        # 법선 벡터 표시 (일부만)
        step = max(1, len(self.x_airfoil) // 20)
        for i in range(0, len(self.x_airfoil), step):
            x, y = self.x_airfoil[i], self.y_airfoil[i]
            nx, ny = self.normals[i]
            ax.arrow(x, y, 0.05*nx, 0.05*ny, head_width=0.01, 
                    head_length=0.01, fc='red', ec='red')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.3, 0.3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('NACA 0012 Airfoil with Normal Vectors')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

class BoundaryRegions:
    """도메인 경계 영역 관리"""
    
    def __init__(self, domain_bounds: Dict[str, float]):
        self.bounds = domain_bounds
        
    def identify_boundary_type(self, x: torch.Tensor, 
                             tolerance: float = 1e-6) -> Dict[str, torch.Tensor]:
        """
        점들의 경계 타입 식별
        
        Args:
            x: 좌표 [N, 3] (t, x, y)
            tolerance: 경계 허용 오차
            
        Returns:
            Dict of boundary masks
        """
        x_coord = x[:, 1]
        y_coord = x[:, 2]
        
        masks = {
            'inlet': torch.abs(x_coord - self.bounds['x_min']) < tolerance,
            'outlet': torch.abs(x_coord - self.bounds['x_max']) < tolerance,
            'top': torch.abs(y_coord - self.bounds['y_max']) < tolerance,
            'bottom': torch.abs(y_coord - self.bounds['y_min']) < tolerance,
            'interior': torch.ones_like(x_coord, dtype=torch.bool)
        }
        
        # 내부점은 경계가 아닌 점들
        boundary_mask = (masks['inlet'] | masks['outlet'] | 
                        masks['top'] | masks['bottom'])
        masks['interior'] = ~boundary_mask
        
        return masks
    
    def sample_boundary_points(self, n_points: int, 
                             boundary_type: str) -> torch.Tensor:
        """특정 경계에서 점 샘플링"""
        
        if boundary_type == 'inlet':
            x = torch.full((n_points, 1), self.bounds['x_min'])
            y = torch.linspace(self.bounds['y_min'], self.bounds['y_max'], n_points).unsqueeze(1)
            
        elif boundary_type == 'outlet':
            x = torch.full((n_points, 1), self.bounds['x_max'])
            y = torch.linspace(self.bounds['y_min'], self.bounds['y_max'], n_points).unsqueeze(1)
            
        elif boundary_type == 'top':
            x = torch.linspace(self.bounds['x_min'], self.bounds['x_max'], n_points).unsqueeze(1)
            y = torch.full((n_points, 1), self.bounds['y_max'])
            
        elif boundary_type == 'bottom':
            x = torch.linspace(self.bounds['x_min'], self.bounds['x_max'], n_points).unsqueeze(1)
            y = torch.full((n_points, 1), self.bounds['y_min'])
            
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
        
        # 시간 좌표는 0으로 설정 (별도 처리)
        t = torch.zeros_like(x)
        
        return torch.cat([t, x, y], dim=1)

class DynamicAirfoilBC:
    """동적 에어포일 경계조건"""
    
    def __init__(self, geometry: NACAGeometry, phys_params: PhysicalParameters):
        self.geometry = geometry
        self.x_ref = phys_params.x_ref
        self.coord_sys = "lab"  # 기본값
        
    def set_coordinate_system(self, coord_sys: str):
        """좌표계 설정"""
        self.coord_sys = coord_sys
        
    def get_surface_points_at_time(self, t: float, h: float, theta: float,
                                  n_points: int = 100) -> torch.Tensor:
        """특정 시간에서의 에어포일 표면점"""
        
        # 변환된 에어포일 좌표
        x_surf, y_surf = self.geometry.transform_airfoil(h, theta, self.x_ref)
        
        # 샘플링
        indices = np.linspace(0, len(x_surf)-1, n_points, dtype=int)
        x_sampled = x_surf[indices]
        y_sampled = y_surf[indices]
        
        # 시간 좌표 추가
        t_coord = np.full_like(x_sampled, t)
        
        # 텐서 변환
        surface_points = torch.tensor(
            np.column_stack([t_coord, x_sampled, y_sampled]),
            dtype=torch.float32
        )
        
        return surface_points
    
    def compute_wall_velocity(self, surface_points: torch.Tensor, 
                            h_vel: float, theta_vel: float, theta: float) -> Dict[str, torch.Tensor]:
        """벽면 속도 계산"""
        
        if self.coord_sys == "lab":
            # lab 좌표계: 실제 벽면 운동
            x_rel = surface_points[:, 1] - self.x_ref
            y_rel = surface_points[:, 2]  # heave 변위 이미 포함
            
            u_wall = (-h_vel * torch.sin(torch.tensor(theta)) - 
                     y_rel * theta_vel)
            v_wall = (h_vel * torch.cos(torch.tensor(theta)) + 
                     x_rel * theta_vel)
            
        else:  # body 좌표계
            # body 좌표계: 정지 에어포일
            u_wall = torch.zeros(len(surface_points))
            v_wall = torch.zeros(len(surface_points))
        
        return {
            'u_wall': u_wall.unsqueeze(1),
            'v_wall': v_wall.unsqueeze(1)
        }
    
    def get_normal_vectors(self, surface_points: torch.Tensor, 
                          theta: float) -> torch.Tensor:
        """표면 법선 벡터 (회전 고려)"""
        
        n_points = len(surface_points)
        normals = np.zeros((n_points, 2))
        
        # 회전된 법선 벡터
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 에어포일 표면점에 가장 가까운 원래 점 찾기
        x_surf = surface_points[:, 1].numpy()
        y_surf = surface_points[:, 2].numpy()
        
        for i in range(n_points):
            # 가장 가까운 원래 점의 인덱스
            distances = np.sqrt(
                (self.geometry.x_airfoil - x_surf[i])**2 + 
                (self.geometry.y_airfoil - y_surf[i])**2
            )
            idx = np.argmin(distances)
            
            # 원래 법선 벡터
            nx_orig = self.geometry.normals[idx, 0]
            ny_orig = self.geometry.normals[idx, 1]
            
            # 회전 적용
            normals[i, 0] = nx_orig * cos_theta - ny_orig * sin_theta
            normals[i, 1] = nx_orig * sin_theta + ny_orig * cos_theta
            
        return torch.tensor(normals, dtype=torch.float32)

class BoundaryConditionManager:
    """통합 경계조건 관리자"""
    
    def __init__(self, phys_params: PhysicalParameters, 
                 domain_params: DomainParameters,
                 coord_sys: str = "lab"):
        
        self.phys_params = phys_params
        self.domain_params = domain_params
        self.coord_sys = coord_sys
        
        # 도메인 경계
        self.domain_bounds = domain_params.get_nondim_bounds(phys_params.C_phys)
        self.boundary_regions = BoundaryRegions(self.domain_bounds)
        
        # 에어포일 형상
        self.geometry = NACAGeometry()
        self.airfoil_bc = DynamicAirfoilBC(self.geometry, phys_params)
        self.airfoil_bc.set_coordinate_system(coord_sys)
        
    def get_all_boundary_points(self, t: float, h: float, theta: float,
                              n_boundary: int = 50, 
                              n_surface: int = 100) -> Dict[str, torch.Tensor]:
        """모든 경계점 생성"""
        
        boundary_points = {}
        
        # 도메인 경계
        for bc_type in ['inlet', 'outlet', 'top', 'bottom']:
            points = self.boundary_regions.sample_boundary_points(n_boundary, bc_type)
            points[:, 0] = t  # 시간 좌표 설정
            boundary_points[bc_type] = points
        
        # 에어포일 표면
        surface_points = self.airfoil_bc.get_surface_points_at_time(
            t, h, theta, n_surface
        )
        boundary_points['airfoil'] = surface_points
        
        return boundary_points
    
    def apply_boundary_conditions(self, points: torch.Tensor, 
                                predictions: torch.Tensor,
                                bc_type: str, 
                                structure_state: Optional[Dict] = None) -> torch.Tensor:
        """경계조건 적용 및 잔차 계산"""
        
        if bc_type == 'inlet':
            # 입구: u = 1, v = 0
            u_target = torch.ones_like(predictions[:, 0:1])
            v_target = torch.zeros_like(predictions[:, 1:2])
            residual = torch.cat([
                predictions[:, 0:1] - u_target,
                predictions[:, 1:2] - v_target
            ], dim=1)
            
        elif bc_type == 'outlet':
            # 출구: ∂p/∂n = 0 (압력 기울기는 별도 계산 필요)
            residual = torch.zeros_like(predictions[:, 0:2])
            
        elif bc_type in ['top', 'bottom']:
            # 상/하 경계: 슬립 조건 (속도 기울기는 별도 계산 필요)
            residual = torch.zeros_like(predictions[:, 0:2])
            
        elif bc_type == 'airfoil':
            # 에어포일 표면: no-slip 또는 벽면 운동
            if structure_state is not None:
                wall_vel = self.airfoil_bc.compute_wall_velocity(
                    points, 
                    structure_state['h_vel'], 
                    structure_state['theta_vel'],
                    structure_state['theta']
                )
                residual = torch.cat([
                    predictions[:, 0:1] - wall_vel['u_wall'],
                    predictions[:, 1:2] - wall_vel['v_wall']
                ], dim=1)
            else:
                # 정지 벽면
                residual = predictions[:, 0:2]
        
        else:
            raise ValueError(f"Unknown boundary type: {bc_type}")
        
        return residual
    
    def visualize_boundaries(self, t: float = 0.0, h: float = 0.0, theta: float = 0.0,
                           save_path: Optional[str] = None):
        """경계 시각화"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 도메인 경계
        x_min, x_max = self.domain_bounds['x_min'], self.domain_bounds['x_max']
        y_min, y_max = self.domain_bounds['y_min'], self.domain_bounds['y_max']
        
        # 도메인 테두리
        ax.plot([x_min, x_max, x_max, x_min, x_min],
               [y_min, y_min, y_max, y_max, y_min], 'k-', linewidth=2, label='Domain')
        
        # 경계 타입별 색상
        colors = {'inlet': 'blue', 'outlet': 'red', 'top': 'green', 'bottom': 'orange'}
        
        boundary_points = self.get_all_boundary_points(t, h, theta)
        
        for bc_type, points in boundary_points.items():
            if bc_type == 'airfoil':
                ax.plot(points[:, 1], points[:, 2], 'k-', linewidth=3, label='Airfoil')
                ax.fill(points[:, 1], points[:, 2], alpha=0.5, color='gray')
            else:
                color = colors.get(bc_type, 'black')
                ax.scatter(points[:, 1], points[:, 2], c=color, s=20, 
                          label=f'{bc_type.capitalize()} BC', alpha=0.7)
        
        ax.set_xlim(x_min-0.1, x_max+0.1)
        ax.set_ylim(y_min-0.1, y_max+0.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Boundary Conditions (t={t:.2f}, h={h:.3f}, θ={theta:.3f})')
        ax.set_xlabel('x/C')
        ax.set_ylabel('y/C')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# 편의 함수들
def create_boundary_manager(phys_params: PhysicalParameters,
                          domain_params: DomainParameters,
                          coord_sys: str = "lab") -> BoundaryConditionManager:
    """경계조건 관리자 생성"""
    return BoundaryConditionManager(phys_params, domain_params, coord_sys)

def test_airfoil_motion():
    """에어포일 운동 테스트"""
    from config import DEFAULT_PHYS_PARAMS, DEFAULT_DOMAIN_PARAMS
    
    bc_manager = create_boundary_manager(
        DEFAULT_PHYS_PARAMS, DEFAULT_DOMAIN_PARAMS
    )
    
    # 플러터 운동 시뮬레이션
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def animate(frame):
        ax.clear()
        t = frame * 0.1
        h = 0.05 * np.sin(2 * np.pi * t)
        theta = 0.1 * np.sin(2 * np.pi * t + np.pi/4)
        
        boundary_points = bc_manager.get_all_boundary_points(t, h, theta)
        airfoil_points = boundary_points['airfoil']
        
        ax.plot(airfoil_points[:, 1], airfoil_points[:, 2], 'b-', linewidth=2)
        ax.fill(airfoil_points[:, 1], airfoil_points[:, 2], alpha=0.3)
        
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.4, 0.4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Flutter Motion (t={t:.2f})')
        
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=100)
    plt.show()
    
    return anim

if __name__ == "__main__":
    # 테스트 실행
    test_airfoil_motion()