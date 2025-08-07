#!/usr/bin/env python3
"""
NACA 0012 Flutter PINN - 퀄리티 대폭 개선 버전
현재 문제점들을 모두 해결한 고품질 PINN
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 모듈
from config import *

class AdvancedPINNModel(nn.Module):
    """고급 PINN 모델 - 품질 최적화"""
    
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=256, num_layers=8):
        super().__init__()
        
        # 향상된 아키텍처
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 입력 정규화
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Fourier Feature Embedding (고품질화)
        self.fourier_dim = 64
        B = torch.randn(input_dim, self.fourier_dim) * 5.0
        self.register_buffer('B', B)
        
        # 메인 네트워크 (더 깊고 넓게)
        layers = []
        current_dim = input_dim + 2 * self.fourier_dim
        
        # 첫 번째 레이어
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()  # Swish activation (더 부드러움)
        ])
        
        # 중간 레이어들 (잔차 연결 포함)
        for i in range(num_layers - 2):
            layers.extend([
                ResidualBlock(hidden_dim),
                nn.Dropout(0.1)  # 정규화
            ])
        
        # 출력 레이어
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        ])
        
        self.network = nn.Sequential(*layers)
        
        # 출력 스케일링
        self.output_scale = nn.Parameter(torch.ones(output_dim))
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def fourier_features(self, x):
        """향상된 Fourier Features"""
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    def forward(self, x):
        # 입력 정규화
        x_norm = self.input_norm(x)
        
        # Fourier features 추가
        fourier_feat = self.fourier_features(x_norm)
        x_enhanced = torch.cat([x_norm, fourier_feat], dim=-1)
        
        # 메인 네트워크
        output = self.network(x_enhanced)
        
        # 출력 스케일링
        return output * self.output_scale


class ResidualBlock(nn.Module):
    """잔차 블록"""
    
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.SiLU()
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class QualityPINNTrainer:
    """고품질 PINN 트레이너"""
    
    def __init__(self):
        """초기화"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 디바이스: {self.device}")
        
        # 고품질 설정
        self.Re = 1000.0
        self.domain = {
            'x_min': -2.0, 'x_max': 4.0,  # 더 긴 도메인 (후류 포함)
            'y_min': -1.5, 'y_max': 1.5
        }
        
        # 모델 생성 (고사양)
        self.model = AdvancedPINNModel(
            input_dim=3, output_dim=3, 
            hidden_dim=512, num_layers=12  # 대폭 확장
        ).to(self.device)
        
        print(f"🧠 모델 파라미터: {sum(p.numel() for p in self.model.parameters()):,}개")
        
        # 최적화기 (고급)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=3e-4, weight_decay=1e-5
        )
        
        # 학습률 스케줄러
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5000, eta_min=1e-6
        )
        
        # 손실 추적
        self.loss_history = {'epoch': [], 'total': [], 'physics': [], 'boundary': []}
        
        # NACA 0012 형상
        self.airfoil_points = self._generate_naca0012(n_points=200)
        
        print("✅ 고품질 PINN 트레이너 초기화 완료")
    
    def _generate_naca0012(self, n_points=100):
        """NACA 0012 에어포일 생성"""
        x = np.linspace(0, 1, n_points//2)
        
        # NACA 0012 두께 분포
        t = 0.12  # 12% 두께
        y_upper = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
        y_lower = -y_upper
        
        # 상면과 하면 결합
        x_airfoil = np.concatenate([x, x[::-1]])
        y_airfoil = np.concatenate([y_upper, y_lower[::-1]])
        
        return torch.tensor(np.column_stack([x_airfoil, y_airfoil]), dtype=torch.float32)
    
    def sample_training_points(self, n_interior=8000, n_boundary=1000, n_airfoil=400):
        """고품질 샘플링"""
        
        # 1. 내부점 (에어포일 외부, 적응적 샘플링)
        interior_points = []
        
        # 기본 랜덤 샘플링
        base_points = self._sample_interior_basic(n_interior // 2)
        interior_points.append(base_points)
        
        # 에어포일 근처 집중 샘플링
        near_airfoil = self._sample_near_airfoil(n_interior // 4)
        interior_points.append(near_airfoil)
        
        # 후류 영역 집중 샘플링
        wake_points = self._sample_wake_region(n_interior // 4)
        interior_points.append(wake_points)
        
        interior = torch.cat(interior_points, dim=0)
        
        # 2. 도메인 경계점
        boundary = self._sample_boundary_points(n_boundary)
        
        # 3. 에어포일 경계점
        airfoil_boundary = self._sample_airfoil_boundary(n_airfoil)
        
        return {
            'interior': interior.to(self.device),
            'boundary': boundary.to(self.device),
            'airfoil': airfoil_boundary.to(self.device)
        }
    
    def _sample_interior_basic(self, n_points):
        """기본 내부점 샘플링"""
        points = []
        attempts = 0
        
        while len(points) < n_points and attempts < n_points * 3:
            x = torch.rand(n_points * 2) * (self.domain['x_max'] - self.domain['x_min']) + self.domain['x_min']
            y = torch.rand(n_points * 2) * (self.domain['y_max'] - self.domain['y_min']) + self.domain['y_min']
            t = torch.zeros_like(x)
            
            # 에어포일 외부 체크
            mask = self._is_outside_airfoil(x, y)
            valid_points = torch.stack([t[mask], x[mask], y[mask]], dim=1)
            
            if len(valid_points) > 0:
                take = min(len(valid_points), n_points - len(points))
                points.append(valid_points[:take])
            
            attempts += n_points * 2
        
        return torch.cat(points, dim=0)[:n_points] if points else torch.empty(0, 3)
    
    def _sample_near_airfoil(self, n_points):
        """에어포일 근처 집중 샘플링"""
        points = []
        attempts = 0
        
        while len(points) < n_points and attempts < n_points * 3:
            # 에어포일 주변 좁은 영역
            x = torch.rand(n_points * 2) * 2.0 - 0.5  # [-0.5, 1.5]
            y = torch.rand(n_points * 2) * 1.0 - 0.5  # [-0.5, 0.5]
            t = torch.zeros_like(x)
            
            mask = self._is_outside_airfoil(x, y)
            valid_points = torch.stack([t[mask], x[mask], y[mask]], dim=1)
            
            if len(valid_points) > 0:
                take = min(len(valid_points), n_points - len(points))
                points.append(valid_points[:take])
            
            attempts += n_points * 2
        
        return torch.cat(points, dim=0)[:n_points] if points else torch.empty(0, 3)
    
    def _sample_wake_region(self, n_points):
        """후류 영역 집중 샘플링"""
        # 후류 영역: x > 1.0, |y| < 0.8
        x = torch.rand(n_points) * 2.5 + 1.0  # [1.0, 3.5]
        y = torch.rand(n_points) * 1.6 - 0.8  # [-0.8, 0.8]
        t = torch.zeros_like(x)
        
        return torch.stack([t, x, y], dim=1)
    
    def _sample_boundary_points(self, n_points):
        """경계점 샘플링"""
        boundary_points = []
        n_per_side = n_points // 4
        
        # 입구 (x = x_min)
        y_inlet = torch.linspace(self.domain['y_min'], self.domain['y_max'], n_per_side)
        x_inlet = torch.full_like(y_inlet, self.domain['x_min'])
        t_inlet = torch.zeros_like(x_inlet)
        boundary_points.append(torch.stack([t_inlet, x_inlet, y_inlet], dim=1))
        
        # 출구 (x = x_max)
        y_outlet = torch.linspace(self.domain['y_min'], self.domain['y_max'], n_per_side)
        x_outlet = torch.full_like(y_outlet, self.domain['x_max'])
        t_outlet = torch.zeros_like(x_outlet)
        boundary_points.append(torch.stack([t_outlet, x_outlet, y_outlet], dim=1))
        
        # 상하 경계
        x_top = torch.linspace(self.domain['x_min'], self.domain['x_max'], n_per_side)
        y_top = torch.full_like(x_top, self.domain['y_max'])
        t_top = torch.zeros_like(x_top)
        boundary_points.append(torch.stack([t_top, x_top, y_top], dim=1))
        
        x_bottom = torch.linspace(self.domain['x_min'], self.domain['x_max'], n_per_side)
        y_bottom = torch.full_like(x_bottom, self.domain['y_min'])
        t_bottom = torch.zeros_like(x_bottom)
        boundary_points.append(torch.stack([t_bottom, x_bottom, y_bottom], dim=1))
        
        return torch.cat(boundary_points, dim=0)
    
    def _sample_airfoil_boundary(self, n_points):
        """에어포일 경계점 샘플링"""
        # 균등 분포로 에어포일 포인트 선택
        indices = torch.linspace(0, len(self.airfoil_points)-1, n_points).long()
        selected_points = self.airfoil_points[indices]
        
        t = torch.zeros(n_points, 1)
        return torch.cat([t, selected_points], dim=1)
    
    def _is_outside_airfoil(self, x, y):
        """에어포일 외부 판정 (간단화된 버전)"""
        # 에어포일 영역 근사: 0 <= x <= 1, |y| <= 0.06 * (대략적 두께)
        inside_x = (x >= 0) & (x <= 1)
        inside_y = torch.abs(y) <= 0.08  # 약간 여유를 둠
        inside_airfoil = inside_x & inside_y
        
        return ~inside_airfoil
    
    def physics_loss(self, points):
        """향상된 물리 손실"""
        points.requires_grad_(True)
        
        # 모델 예측
        output = self.model(points)
        u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]
        
        # 1차 미분 (배치별로 안전하게)
        grad_outputs = torch.ones_like(u)
        
        u_grads = torch.autograd.grad(u, points, grad_outputs, create_graph=True)[0]
        v_grads = torch.autograd.grad(v, points, grad_outputs, create_graph=True)[0]
        p_grads = torch.autograd.grad(p, points, grad_outputs, create_graph=True)[0]
        
        u_t, u_x, u_y = u_grads[:, 0:1], u_grads[:, 1:2], u_grads[:, 2:3]
        v_t, v_x, v_y = v_grads[:, 0:1], v_grads[:, 1:2], v_grads[:, 2:3]
        p_x, p_y = p_grads[:, 1:2], p_grads[:, 2:3]
        
        # 2차 미분
        u_xx = torch.autograd.grad(u_x, points, grad_outputs, create_graph=True)[0][:, 1:2]
        u_yy = torch.autograd.grad(u_y, points, grad_outputs, create_graph=True)[0][:, 2:3]
        v_xx = torch.autograd.grad(v_x, points, grad_outputs, create_graph=True)[0][:, 1:2]
        v_yy = torch.autograd.grad(v_y, points, grad_outputs, create_graph=True)[0][:, 2:3]
        
        # Navier-Stokes 방정식
        momentum_x = u_t + u * u_x + v * u_y + p_x - (1/self.Re) * (u_xx + u_yy)
        momentum_y = v_t + u * v_x + v * v_y + p_y - (1/self.Re) * (v_xx + v_yy)
        continuity = u_x + v_y
        
        # 가중 평균 (더 안정적)
        return (torch.mean(momentum_x**2) + torch.mean(momentum_y**2) + 
                10.0 * torch.mean(continuity**2))  # 연속성 강화
    
    def boundary_loss(self, samples):
        """경계조건 손실"""
        total_loss = 0.0
        
        # 도메인 경계
        if len(samples['boundary']) > 0:
            boundary_pred = self.model(samples['boundary'])
            
            # 간단한 경계조건 (개선 가능)
            # 입구: u ≈ 1, v ≈ 0
            # 다른 경계: slip 조건 등
            u_boundary = boundary_pred[:, 0:1]
            v_boundary = boundary_pred[:, 1:2]
            
            total_loss += torch.mean((u_boundary - 1.0)**2) * 0.1  # 완화된 조건
            total_loss += torch.mean(v_boundary**2) * 0.1
        
        # 에어포일 경계 (no-slip)
        if len(samples['airfoil']) > 0:
            airfoil_pred = self.model(samples['airfoil'])
            u_airfoil = airfoil_pred[:, 0:1]
            v_airfoil = airfoil_pred[:, 1:2]
            
            total_loss += torch.mean(u_airfoil**2) + torch.mean(v_airfoil**2)
        
        return total_loss
    
    def train(self, epochs=8000):
        """고품질 훈련"""
        print(f"🚀 고품질 PINN 훈련 시작 ({epochs} epochs)")
        print(f"   모델 크기: {sum(p.numel() for p in self.model.parameters()):,} 파라미터")
        print(f"   디바이스: {self.device}")
        
        best_loss = float('inf')
        patience = 500
        patience_counter = 0
        
        progress_bar = tqdm(range(epochs), desc="Training")
        
        for epoch in progress_bar:
            self.model.train()
            self.optimizer.zero_grad()
            
            # 매 에포크마다 새로운 샘플링
            samples = self.sample_training_points()
            
            # 손실 계산
            physics_loss = self.physics_loss(samples['interior'])
            boundary_loss = self.boundary_loss(samples)
            
            total_loss = physics_loss + 5.0 * boundary_loss  # 균형 조정
            
            # 역전파
            total_loss.backward()
            
            # 그래디언트 클리핑 (안정성)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 손실 기록
            self.loss_history['epoch'].append(epoch)
            self.loss_history['total'].append(total_loss.item())
            self.loss_history['physics'].append(physics_loss.item())
            self.loss_history['boundary'].append(boundary_loss.item())
            
            # 진행률 업데이트
            progress_bar.set_postfix({
                'Total': f"{total_loss.item():.2e}",
                'Physics': f"{physics_loss.item():.2e}",
                'Boundary': f"{boundary_loss.item():.2e}",
                'LR': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # 조기 종료 체크
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
                # 최고 모델 저장
                self.save_model(f"quality_boost_best_model.pt", epoch, best_loss)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n⏰ 조기 종료 (에포크 {epoch})")
                break
            
            # 주기적 체크포인트
            if epoch % 1000 == 0 and epoch > 0:
                self.save_model(f"quality_boost_checkpoint_{epoch}.pt", epoch, total_loss.item())
        
        print("✅ 고품질 훈련 완료!")
        return best_loss
    
    def save_model(self, filename, epoch, loss):
        """모델 저장"""
        results_dir = Path(r"C:\Users\Master\Desktop\NACA0012_Flutter_PINN\results")
        results_dir.mkdir(exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'loss_history': self.loss_history
        }
        
        torch.save(save_dict, results_dir / filename)
    
    def visualize_ultra_quality(self):
        """초고품질 시각화"""
        print("🎨 초고품질 시각화 생성 중...")
        
        # 초고해상도 그리드
        x = np.linspace(self.domain['x_min'], self.domain['x_max'], 400)
        y = np.linspace(self.domain['y_min'], self.domain['y_max'], 300)
        X, Y = np.meshgrid(x, y)
        
        # 배치별 예측
        points_list = []
        batch_size = 5000
        
        for i in range(0, len(X.flatten()), batch_size):
            end_idx = min(i + batch_size, len(X.flatten()))
            t_batch = np.zeros(end_idx - i)
            x_batch = X.flatten()[i:end_idx]
            y_batch = Y.flatten()[i:end_idx]
            
            points_batch = torch.tensor(
                np.column_stack([t_batch, x_batch, y_batch]),
                dtype=torch.float32
            ).to(self.device)
            
            with torch.no_grad():
                pred_batch = self.model(points_batch)
            
            points_list.append(pred_batch.cpu())
        
        predictions = torch.cat(points_list, dim=0)
        
        u_pred = predictions[:, 0].numpy().reshape(X.shape)
        v_pred = predictions[:, 1].numpy().reshape(X.shape)
        p_pred = predictions[:, 2].numpy().reshape(X.shape)
        
        # 에어포일 마스킹
        mask = self._create_airfoil_mask(X, Y)
        u_pred = np.where(mask, u_pred, np.nan)
        v_pred = np.where(mask, v_pred, np.nan)
        p_pred = np.where(mask, p_pred, np.nan)
        
        # 시각화
        self._plot_ultra_quality_results(X, Y, u_pred, v_pred, p_pred)
    
    def _create_airfoil_mask(self, X, Y):
        """에어포일 마스크 생성"""
        # 간단한 에어포일 마스크
        mask = ~((X >= 0) & (X <= 1) & (np.abs(Y) <= 0.08))
        return mask
    
    def _plot_ultra_quality_results(self, X, Y, u, v, p):
        """초고품질 플롯"""
        # 속도 크기 및 와도 계산
        velocity_mag = np.sqrt(u**2 + v**2)
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]
        dudy = np.gradient(u, dy, axis=0)
        dvdx = np.gradient(v, dx, axis=1)
        vorticity = dvdx - dudy
        
        # 전체 도메인 시각화
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        
        # 1. 속도 크기
        im1 = axes[0, 0].contourf(X, Y, velocity_mag, levels=150, cmap='viridis')
        axes[0, 0].set_title('Velocity Magnitude |V|', fontsize=18, fontweight='bold')
        cbar1 = fig.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        cbar1.ax.tick_params(labelsize=14)
        
        # 2. U 속도
        im2 = axes[0, 1].contourf(X, Y, u, levels=150, cmap='RdBu_r')
        axes[0, 1].set_title('U-Velocity', fontsize=18, fontweight='bold')
        cbar2 = fig.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        cbar2.ax.tick_params(labelsize=14)
        
        # 3. V 속도
        im3 = axes[0, 2].contourf(X, Y, v, levels=150, cmap='RdBu_r')
        axes[0, 2].set_title('V-Velocity', fontsize=18, fontweight='bold')
        cbar3 = fig.colorbar(im3, ax=axes[0, 2], shrink=0.8)
        cbar3.ax.tick_params(labelsize=14)
        
        # 4. 압력
        im4 = axes[1, 0].contourf(X, Y, p, levels=150, cmap='coolwarm')
        axes[1, 0].set_title('Pressure', fontsize=18, fontweight='bold')
        cbar4 = fig.colorbar(im4, ax=axes[1, 0], shrink=0.8)
        cbar4.ax.tick_params(labelsize=14)
        
        # 5. 유선
        axes[1, 1].streamplot(X, Y, u, v, density=8, color=velocity_mag, 
                             cmap='plasma', linewidth=2, arrowsize=2)
        axes[1, 1].set_title('Streamlines', fontsize=18, fontweight='bold')
        
        # 6. 와도
        im6 = axes[1, 2].contourf(X, Y, vorticity, levels=150, cmap='RdBu_r')
        axes[1, 2].set_title('Vorticity ωz', fontsize=18, fontweight='bold')
        cbar6 = fig.colorbar(im6, ax=axes[1, 2], shrink=0.8)
        cbar6.ax.tick_params(labelsize=14)
        
        # 에어포일 표시
        airfoil_np = self.airfoil_points.numpy()
        for ax in axes.flat:
            ax.plot(airfoil_np[:, 0], airfoil_np[:, 1], 'k-', linewidth=4)
            ax.fill(airfoil_np[:, 0], airfoil_np[:, 1], color='black', alpha=0.9)
            ax.set_xlim(self.domain['x_min'], self.domain['x_max'])
            ax.set_ylim(self.domain['y_min'], self.domain['y_max'])
            ax.set_xlabel('x/C', fontsize=16, fontweight='bold')
            ax.set_ylabel('y/C', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=14)
        
        plt.tight_layout(pad=4.0)
        
        # 저장
        results_dir = Path(r"C:\Users\Master\Desktop\NACA0012_Flutter_PINN\results")
        save_path = results_dir / "quality_boost_ultra_high_resolution.png"
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ 초고품질 시각화 저장: {save_path}")
        
        # 후류 분석
        self._analyze_wake_quality(X, Y, u, v)
    
    def _analyze_wake_quality(self, X, Y, u, v):
        """후류 품질 분석"""
        # 후류 중심선 (y=0 근처)
        center_idx = len(Y) // 2
        wake_start_idx = np.argmin(np.abs(X[center_idx, :] - 1.0))
        
        x_wake = X[center_idx, wake_start_idx:]
        u_wake = u[center_idx, wake_start_idx:]
        
        # 속도 부족 계산
        velocity_deficit = 1.0 - u_wake
        max_deficit = np.nanmax(velocity_deficit)
        
        print(f"\n🌊 후류 품질 분석:")
        print(f"   최대 속도 부족: {max_deficit:.3f}")
        print(f"   후류 복구 거리: {x_wake[np.argmin(np.abs(velocity_deficit - 0.1))] - 1.0:.2f}C")
        
        if max_deficit > 0.2:
            print("✅ 우수한 후류 품질!")
        elif max_deficit > 0.1:
            print("⚠️ 보통 후류 품질")
        else:
            print("❌ 후류 부족")
    
    def plot_loss_history(self):
        """손실 이력 플롯"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.loss_history['epoch']
        
        # 총 손실
        axes[0, 0].semilogy(epochs, self.loss_history['total'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 물리 손실
        axes[0, 1].semilogy(epochs, self.loss_history['physics'], 'r-', linewidth=2)
        axes[0, 1].set_title('Physics Loss', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 경계 손실
        axes[1, 0].semilogy(epochs, self.loss_history['boundary'], 'g-', linewidth=2)
        axes[1, 0].set_title('Boundary Loss', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 학습률
        lr_values = [self.scheduler.get_last_lr()[0] * (0.99 ** epoch) for epoch in epochs]
        axes[1, 1].semilogy(epochs, lr_values, 'm-', linewidth=2)
        axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        for ax in axes.flat:
            ax.set_xlabel('Epoch')
        
        plt.tight_layout()
        
        results_dir = Path(r"C:\Users\Master\Desktop\NACA0012_Flutter_PINN\results")
        save_path = results_dir / "quality_boost_loss_history.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 손실 이력 저장: {save_path}")


def main():
    """메인 함수"""
    print("🚀 NACA 0012 퀄리티 부스트 PINN")
    print("="*60)
    print("🔥 대폭 개선된 아키텍처와 훈련 전략")
    print("🎯 목표: 전문가급 CFD 품질 달성")
    print("="*60)
    
    # 고품질 트레이너 생성
    trainer = QualityPINNTrainer()
    
    # 훈련
    best_loss = trainer.train(epochs=8000)
    
    # 시각화
    trainer.visualize_ultra_quality()
    trainer.plot_loss_history()
    
    print(f"\n🎉 퀄리티 부스트 완료!")
    print(f"   최고 손실: {best_loss:.2e}")
    print(f"   모델 크기: {sum(p.numel() for p in trainer.model.parameters()):,} 파라미터")


if __name__ == "__main__":
    main()