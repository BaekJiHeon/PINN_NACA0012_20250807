"""
NACA 0012 Flutter PINN - 손실 함수 모듈
Physics-Informed Neural Networks for 2-DOF Flutter Analysis

복합 손실 함수:
- Navier-Stokes PDE 잔차
- 경계조건 위반
- FSI 결합 (구조 ODE)
- 데이터 피팅
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from config import PhysicalParameters, TrainingConfig

class NavierStokesLoss:
    """2D 비압축성 Navier-Stokes 방정식 손실"""
    
    def __init__(self, phys_params: PhysicalParameters, coord_sys: str = "lab"):
        self.nu = phys_params.nu if hasattr(phys_params, 'nu') else 1.0/1000.0  # Re=1000 기본
        self.coord_sys = coord_sys
        
    def compute_residual(self, x: torch.Tensor, model_output: torch.Tensor,
                        gradients: Dict[str, torch.Tensor],
                        mesh_velocity: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        NS 방정식 잔차 계산
        
        Args:
            x: 입력 좌표 [batch, 3] (t, x, y)
            model_output: 모델 출력 [batch, 3] (u, v, p)
            gradients: 자동미분으로 계산된 기울기들
            mesh_velocity: ALE 격자 속도 (body 좌표계용)
        """
        u = model_output[:, 0:1]
        v = model_output[:, 1:2]
        
        # 편미분들 추출
        u_t, u_x, u_y = gradients['u_t'], gradients['u_x'], gradients['u_y']
        v_t, v_x, v_y = gradients['v_t'], gradients['v_x'], gradients['v_y']
        p_x, p_y = gradients['p_x'], gradients['p_y']
        u_xx, u_yy = gradients['u_xx'], gradients['u_yy']
        v_xx, v_yy = gradients['v_xx'], gradients['v_yy']
        
        # ALE 격자 속도 (body 좌표계)
        w_x = torch.zeros_like(u)
        w_y = torch.zeros_like(v)
        
        if self.coord_sys == "body" and mesh_velocity is not None:
            w_x = mesh_velocity.get('w_x', w_x)
            w_y = mesh_velocity.get('w_y', w_y)
        
        # 운동량 방정식 잔차
        # ∂u/∂t + (u-w_x)∂u/∂x + (v-w_y)∂u/∂y = -∂p/∂x + ν∇²u
        momentum_x = (u_t + (u - w_x) * u_x + (v - w_y) * u_y + 
                     p_x - self.nu * (u_xx + u_yy))
        
        # ∂v/∂t + (u-w_x)∂v/∂x + (v-w_y)∂v/∂y = -∂p/∂y + ν∇²v  
        momentum_y = (v_t + (u - w_x) * v_x + (v - w_y) * v_y + 
                     p_y - self.nu * (v_xx + v_yy))
        
        # 연속성 방정식 잔차
        # ∂u/∂x + ∂v/∂y = 0
        continuity = u_x + v_y
        
        return {
            'momentum_x': momentum_x,
            'momentum_y': momentum_y,
            'continuity': continuity
        }

class BoundaryConditionLoss:
    """경계조건 손실 함수"""
    
    def __init__(self, domain_bounds: Dict[str, float]):
        self.domain_bounds = domain_bounds
        
    def inlet_bc(self, x: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        """입구 경계조건: u = 1, v = 0"""
        u_inlet = model_output[:, 0:1] - 1.0
        v_inlet = model_output[:, 1:2] - 0.0
        return torch.cat([u_inlet, v_inlet], dim=1)
    
    def outlet_bc(self, x: torch.Tensor, gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """출구 경계조건: ∂p/∂x = 0"""
        return gradients['p_x']
    
    def wall_bc(self, x: torch.Tensor, model_output: torch.Tensor,
               wall_velocity: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """벽면 경계조건 (no-slip 또는 이동 벽면)"""
        u_wall_target = torch.zeros_like(model_output[:, 0:1])
        v_wall_target = torch.zeros_like(model_output[:, 1:2])
        
        if wall_velocity is not None:
            u_wall_target = wall_velocity.get('u_wall', u_wall_target)
            v_wall_target = wall_velocity.get('v_wall', v_wall_target)
        
        u_wall_residual = model_output[:, 0:1] - u_wall_target
        v_wall_residual = model_output[:, 1:2] - v_wall_target
        
        return torch.cat([u_wall_residual, v_wall_residual], dim=1)
    
    def slip_bc(self, x: torch.Tensor, gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """슬립 경계조건: ∂u/∂y = ∂v/∂y = 0 (상/하 경계)"""
        u_slip = gradients['u_y']
        v_slip = gradients['v_y']
        return torch.cat([u_slip, v_slip], dim=1)

class StructuralDynamicsLoss:
    """2-DOF 구조 동역학 손실"""
    
    def __init__(self, phys_params: PhysicalParameters):
        self.m = phys_params.m
        self.I_alpha = phys_params.I_alpha
        self.c_h = phys_params.c_h
        self.k_h = phys_params.k_h
        self.c_alpha = phys_params.c_alpha
        self.k_alpha = phys_params.k_alpha
        
        # 학습 가능한 파라미터로 설정 (inverse problem)
        self.learnable_c_h = nn.Parameter(torch.tensor(self.c_h))
        self.learnable_k_h = nn.Parameter(torch.tensor(self.k_h))
        self.learnable_c_alpha = nn.Parameter(torch.tensor(self.c_alpha))
        self.learnable_k_alpha = nn.Parameter(torch.tensor(self.k_alpha))
        
    def compute_residual(self, t: torch.Tensor, h: torch.Tensor, theta: torch.Tensor,
                        h_vel: torch.Tensor, theta_vel: torch.Tensor,
                        lift: torch.Tensor, moment: torch.Tensor,
                        learn_params: bool = False) -> Dict[str, torch.Tensor]:
        """
        구조 ODE 잔차 계산
        
        2-DOF 방정식:
        m ḧ + c_h ḣ + k_h h = -Lift
        Iα θ̈ + c_α θ̇ + k_α θ = Moment
        """
        
        # h와 theta의 시간 미분 계산 (데이터 기반 - 수치 미분으로 대체)
        # 고정된 데이터에서는 gradient 계산 대신 잔차만 평가
        try:
            h_acc_grad = torch.autograd.grad(h_vel.sum(), t, create_graph=True, allow_unused=True)[0]
            theta_acc_grad = torch.autograd.grad(theta_vel.sum(), t, create_graph=True, allow_unused=True)[0]
            
            # None 체크
            h_acc = h_acc_grad if h_acc_grad is not None else torch.zeros_like(h_vel)
            theta_acc = theta_acc_grad if theta_acc_grad is not None else torch.zeros_like(theta_vel)
        except:
            # gradient 계산 실패시 0으로 설정 (데이터 기반 학습)
            h_acc = torch.zeros_like(h_vel)
            theta_acc = torch.zeros_like(theta_vel)
        
        # 사용할 파라미터 선택
        if learn_params:
            c_h_used = self.learnable_c_h
            k_h_used = self.learnable_k_h
            c_alpha_used = self.learnable_c_alpha
            k_alpha_used = self.learnable_k_alpha
        else:
            c_h_used = self.c_h
            k_h_used = self.k_h
            c_alpha_used = self.c_alpha
            k_alpha_used = self.k_alpha
        
        # Heave 방정식 잔차
        heave_residual = (self.m * h_acc + c_h_used * h_vel + 
                         k_h_used * h + lift)
        
        # Pitch 방정식 잔차  
        pitch_residual = (self.I_alpha * theta_acc + c_alpha_used * theta_vel + 
                         k_alpha_used * theta - moment)
        
        return {
            'heave_ode': heave_residual,
            'pitch_ode': pitch_residual
        }

class FSICouplingLoss:
    """유동-구조 상호작용 결합 손실"""
    
    def __init__(self, phys_params: PhysicalParameters):
        self.x_ref = phys_params.x_ref  # 기준점 위치
        self.C_phys = phys_params.C_phys
        
    def compute_airfoil_surface_velocity(self, x_surface: torch.Tensor, 
                                       h: torch.Tensor, theta: torch.Tensor,
                                       h_vel: torch.Tensor, theta_vel: torch.Tensor,
                                       coord_sys: str = "lab") -> Dict[str, torch.Tensor]:
        """
        에어포일 표면 속도 계산
        
        lab 좌표계:
        u_wall = -ḣ sin(θ) - (y-y_ref) θ̇
        v_wall = ḣ cos(θ) + (x-x_ref) θ̇
        """
        
        if coord_sys == "lab":
            x_rel = x_surface[:, 1:2] - self.x_ref  # x - x_ref  
            y_rel = x_surface[:, 2:3]               # y - y_ref (=0)
            
            u_wall = (-h_vel * torch.sin(theta) - y_rel * theta_vel)
            v_wall = (h_vel * torch.cos(theta) + x_rel * theta_vel)
        else:  # body 좌표계
            u_wall = torch.zeros_like(x_surface[:, 1:2])
            v_wall = torch.zeros_like(x_surface[:, 2:3])
        
        return {'u_wall': u_wall, 'v_wall': v_wall}
    
    def compute_mesh_velocity(self, x: torch.Tensor,
                            h_vel: torch.Tensor, theta_vel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ALE 격자 속도 계산 (body 좌표계용)"""
        # 단순한 강체 운동으로 가정
        w_x = -h_vel * torch.sin(torch.zeros_like(h_vel))  # 선형화된 small angle
        w_y = h_vel + x[:, 1:2] * theta_vel
        
        return {'w_x': w_x, 'w_y': w_y}

class DataFittingLoss:
    """CFD 데이터 피팅 손실"""
    
    def __init__(self):
        self.mse_loss = nn.MSELoss()
        
    def compute_loss(self, predicted: torch.Tensor, 
                    target: torch.Tensor, 
                    weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """가중 MSE 손실 계산"""
        residual = predicted - target
        
        if weights is not None:
            residual = residual * weights
            
        return torch.mean(residual**2)

class AdaptiveWeighting:
    """적응적 가중치 조정 (GradNorm 기반)"""
    
    def __init__(self, num_losses: int, alpha: float = 0.16):
        self.num_losses = num_losses
        self.alpha = alpha
        self.initial_losses = None
        self.loss_weights = nn.Parameter(torch.ones(num_losses))
        
    def update_weights(self, losses: List[torch.Tensor], 
                      shared_params: List[torch.Tensor]) -> torch.Tensor:
        """GradNorm을 이용한 가중치 업데이트"""
        
        if self.initial_losses is None:
            self.initial_losses = [loss.detach() for loss in losses]
            
        # 각 손실에 대한 기울기 계산
        grads = []
        for i, loss in enumerate(losses):
            grad = torch.autograd.grad(
                self.loss_weights[i] * loss, shared_params,
                retain_graph=True, create_graph=True
            )
            grad_norm = torch.sqrt(sum([g.pow(2).sum() for g in grad]))
            grads.append(grad_norm)
        
        grads = torch.stack(grads)
        
        # 상대적 손실 변화율
        loss_ratios = torch.stack([
            losses[i] / self.initial_losses[i] for i in range(len(losses))
        ])
        
        # 평균 기울기
        avg_grad = grads.mean()
        
        # 목표 기울기
        relative_rate = loss_ratios / loss_ratios.mean()
        target_grads = avg_grad * (relative_rate ** self.alpha)
        
        # 가중치 업데이트를 위한 손실
        grad_loss = torch.sum(torch.abs(grads - target_grads))
        
        return grad_loss

class CompositeLoss:
    """복합 손실 함수 클래스"""
    
    def __init__(self, phys_params: PhysicalParameters, 
                 training_config: TrainingConfig,
                 domain_bounds: Dict[str, float],
                 coord_sys: str = "lab"):
        
        self.coord_sys = coord_sys
        
        # 개별 손실 함수들
        self.ns_loss = NavierStokesLoss(phys_params, coord_sys)
        self.bc_loss = BoundaryConditionLoss(domain_bounds)
        self.struct_loss = StructuralDynamicsLoss(phys_params)
        self.fsi_loss = FSICouplingLoss(phys_params)
        self.data_loss = DataFittingLoss()
        
        # 손실 가중치
        self.lambda_data = training_config.lambda_data
        self.lambda_pde = training_config.lambda_pde
        self.lambda_bc = training_config.lambda_bc
        self.lambda_fsi = training_config.lambda_fsi
        
        # 적응적 가중치 (선택사항)
        self.adaptive_weighting = AdaptiveWeighting(num_losses=4)
        self.use_adaptive = False
        
    def compute_total_loss(self, 
                          model: torch.nn.Module,
                          model_output: torch.Tensor,
                          x_collocation: torch.Tensor,
                          x_boundary: Dict[str, torch.Tensor],
                          x_surface: torch.Tensor,
                          structure_data: Dict[str, torch.Tensor],
                          cfd_data: Optional[Dict[str, torch.Tensor]] = None,
                          gradients: Dict[str, torch.Tensor] = None,
                          learn_struct_params: bool = False) -> Dict[str, torch.Tensor]:
        """
        전체 복합 손실 계산
        
        Returns:
            Dict containing individual losses and total loss
        """
        
        losses = {}
        
        # 1. PDE 잔차 손실 (Navier-Stokes)
        if gradients is not None:
            mesh_vel = None
            if self.coord_sys == "body":
                mesh_vel = self.fsi_loss.compute_mesh_velocity(
                    x_collocation, 
                    structure_data['h_vel'], 
                    structure_data['theta_vel']
                )
            
            ns_residuals = self.ns_loss.compute_residual(
                x_collocation, model_output, gradients, mesh_vel
            )
            
            pde_loss = (torch.mean(ns_residuals['momentum_x']**2) + 
                       torch.mean(ns_residuals['momentum_y']**2) + 
                       torch.mean(ns_residuals['continuity']**2))
            losses['pde'] = pde_loss
        
        # 2. 경계조건 손실
        bc_total = torch.tensor(0.0, device=model_output.device)
        
        # 경계조건은 main.py에서 이미 계산된 bc_residuals 사용
        # 여기서는 주어진 bc_residuals를 손실로 변환
        if 'bc_residuals' in cfd_data:
            for bc_type, residual in cfd_data['bc_residuals'].items():
                if residual is not None and len(residual) > 0:
                    bc_total += torch.mean(residual**2)
        
        # 에어포일 표면 경계조건도 bc_residuals에서 처리됨
        # (이미 위에서 처리)
        
        losses['bc'] = bc_total
        
        # 3. 구조 동역학 손실 (FSI)
        struct_residuals = self.struct_loss.compute_residual(
            structure_data['time'], 
            structure_data['h'], structure_data['theta'],
            structure_data['h_vel'], structure_data['theta_vel'],
            structure_data['lift'], structure_data['moment'],
            learn_struct_params
        )
        
        fsi_loss = (torch.mean(struct_residuals['heave_ode']**2) + 
                   torch.mean(struct_residuals['pitch_ode']**2))
        losses['fsi'] = fsi_loss
        
        # 4. 데이터 피팅 손실
        data_loss_total = torch.tensor(0.0, device=model_output.device)
        if cfd_data is not None:
            for field in ['u', 'v', 'p']:
                if field in cfd_data:
                    data_loss_total += self.data_loss.compute_loss(
                        model_output[:, ['u', 'v', 'p'].index(field):['u', 'v', 'p'].index(field)+1],
                        cfd_data[field]
                    )
        losses['data'] = data_loss_total
        
        # 5. 총 손실 계산
        if self.use_adaptive:
            # 적응적 가중치 사용
            loss_list = [losses['data'], losses['pde'], losses['bc'], losses['fsi']]
            # TODO: shared_params 구현 필요
            # grad_loss = self.adaptive_weighting.update_weights(loss_list, shared_params)
            total_loss = sum(loss_list)  # 임시
        else:
            # 고정 가중치 사용
            total_loss = (self.lambda_data * losses['data'] + 
                         self.lambda_pde * losses['pde'] + 
                         self.lambda_bc * losses['bc'] + 
                         self.lambda_fsi * losses['fsi'])
        
        losses['total'] = total_loss
        
        return losses
    
    def get_loss_weights(self) -> Dict[str, float]:
        """현재 손실 가중치 반환"""
        return {
            'lambda_data': self.lambda_data,
            'lambda_pde': self.lambda_pde, 
            'lambda_bc': self.lambda_bc,
            'lambda_fsi': self.lambda_fsi
        }
    
    def update_loss_weights(self, new_weights: Dict[str, float]):
        """손실 가중치 업데이트"""
        for key, value in new_weights.items():
            if hasattr(self, key):
                setattr(self, key, value)