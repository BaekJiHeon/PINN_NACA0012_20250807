#!/usr/bin/env python3
"""
초간단 원형 실린더 유동 PINN
복잡한 NACA 0012 대신 검증된 간단한 문제로 시작
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class SimplePINN(nn.Module):
    """간단한 PINN 모델"""
    
    def __init__(self, hidden_dim=50, num_layers=4):
        super().__init__()
        
        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]  # (t, x, y) input
        
        for _ in range(num_layers-2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        
        layers.append(nn.Linear(hidden_dim, 3))  # (u, v, p) output
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class CircularCylinderPINN:
    """원형 실린더 주변 유동 PINN"""
    
    def __init__(self):
        self.model = SimplePINN()
        self.Re = 40.0  # 낮은 Reynolds 수로 안정한 유동
        self.cylinder_radius = 0.5
        
        # 도메인: [-2, 6] x [-2, 2]
        self.x_min, self.x_max = -2.0, 6.0
        self.y_min, self.y_max = -2.0, 2.0
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        print("✅ 간단한 원형 실린더 PINN 초기화 완료")
        print(f"   Reynolds 수: {self.Re}")
        print(f"   실린더 반지름: {self.cylinder_radius}")
    
    def sample_points(self, n_interior=1000, n_boundary=200, n_cylinder=100):
        """샘플링"""
        
        # 1. 내부점 (실린더 외부만)
        interior_points = []
        while len(interior_points) < n_interior:
            x = torch.rand(n_interior * 2) * (self.x_max - self.x_min) + self.x_min
            y = torch.rand(n_interior * 2) * (self.y_max - self.y_min) + self.y_min
            t = torch.zeros_like(x)  # 정상 유동
            
            # 실린더 외부만 선택
            mask = (x**2 + y**2) > self.cylinder_radius**2
            valid_points = torch.stack([t[mask], x[mask], y[mask]], dim=1)
            
            interior_points.append(valid_points[:n_interior-len(interior_points)])
        
        interior = torch.cat(interior_points, dim=0)[:n_interior]
        
        # 2. 도메인 경계점
        # 입구 (x = x_min)
        y_inlet = torch.linspace(self.y_min, self.y_max, n_boundary//4)
        x_inlet = torch.full_like(y_inlet, self.x_min)
        t_inlet = torch.zeros_like(x_inlet)
        inlet = torch.stack([t_inlet, x_inlet, y_inlet], dim=1)
        
        # 출구 (x = x_max)  
        y_outlet = torch.linspace(self.y_min, self.y_max, n_boundary//4)
        x_outlet = torch.full_like(y_outlet, self.x_max)
        t_outlet = torch.zeros_like(x_outlet)
        outlet = torch.stack([t_outlet, x_outlet, y_outlet], dim=1)
        
        # 상하 경계
        x_top = torch.linspace(self.x_min, self.x_max, n_boundary//4)
        y_top = torch.full_like(x_top, self.y_max)
        t_top = torch.zeros_like(x_top)
        top = torch.stack([t_top, x_top, y_top], dim=1)
        
        x_bottom = torch.linspace(self.x_min, self.x_max, n_boundary//4)
        y_bottom = torch.full_like(x_bottom, self.y_min)
        t_bottom = torch.zeros_like(x_bottom)
        bottom = torch.stack([t_bottom, x_bottom, y_bottom], dim=1)
        
        # 3. 실린더 경계
        theta = torch.linspace(0, 2*np.pi, n_cylinder)
        x_cyl = self.cylinder_radius * torch.cos(theta)
        y_cyl = self.cylinder_radius * torch.sin(theta)
        t_cyl = torch.zeros_like(x_cyl)
        cylinder = torch.stack([t_cyl, x_cyl, y_cyl], dim=1)
        
        return {
            'interior': interior,
            'inlet': inlet,
            'outlet': outlet, 
            'top': top,
            'bottom': bottom,
            'cylinder': cylinder
        }
    
    def physics_loss(self, points):
        """물리 법칙 손실 (Navier-Stokes)"""
        points.requires_grad_(True)
        
        # 모델 예측
        output = self.model(points)
        u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]
        
        # 미분 계산
        u_t = torch.autograd.grad(u.sum(), points, create_graph=True)[0][:, 0:1]
        u_x = torch.autograd.grad(u.sum(), points, create_graph=True)[0][:, 1:2] 
        u_y = torch.autograd.grad(u.sum(), points, create_graph=True)[0][:, 2:3]
        
        v_t = torch.autograd.grad(v.sum(), points, create_graph=True)[0][:, 0:1]
        v_x = torch.autograd.grad(v.sum(), points, create_graph=True)[0][:, 1:2]
        v_y = torch.autograd.grad(v.sum(), points, create_graph=True)[0][:, 2:3]
        
        p_x = torch.autograd.grad(p.sum(), points, create_graph=True)[0][:, 1:2]
        p_y = torch.autograd.grad(p.sum(), points, create_graph=True)[0][:, 2:3]
        
        # 2차 미분
        u_xx = torch.autograd.grad(u_x.sum(), points, create_graph=True)[0][:, 1:2]
        u_yy = torch.autograd.grad(u_y.sum(), points, create_graph=True)[0][:, 2:3]
        v_xx = torch.autograd.grad(v_x.sum(), points, create_graph=True)[0][:, 1:2]
        v_yy = torch.autograd.grad(v_y.sum(), points, create_graph=True)[0][:, 2:3]
        
        # Navier-Stokes 방정식
        momentum_x = u_t + u * u_x + v * u_y + p_x - (1/self.Re) * (u_xx + u_yy)
        momentum_y = v_t + u * v_x + v * v_y + p_y - (1/self.Re) * (v_xx + v_yy)
        continuity = u_x + v_y
        
        return torch.mean(momentum_x**2) + torch.mean(momentum_y**2) + torch.mean(continuity**2)
    
    def boundary_loss(self, samples):
        """경계조건 손실"""
        total_loss = 0.0
        
        # 입구: u=1, v=0
        if len(samples['inlet']) > 0:
            inlet_pred = self.model(samples['inlet'])
            u_inlet = inlet_pred[:, 0:1]
            v_inlet = inlet_pred[:, 1:2]
            total_loss += torch.mean((u_inlet - 1.0)**2) + torch.mean(v_inlet**2)
        
        # 실린더: u=0, v=0 (no-slip)
        if len(samples['cylinder']) > 0:
            cyl_pred = self.model(samples['cylinder'])
            u_cyl = cyl_pred[:, 0:1]
            v_cyl = cyl_pred[:, 1:2]
            total_loss += torch.mean(u_cyl**2) + torch.mean(v_cyl**2)
        
        # 상하 경계: v=0 (slip condition)
        for boundary in ['top', 'bottom']:
            if len(samples[boundary]) > 0:
                boundary_pred = self.model(samples[boundary])
                v_boundary = boundary_pred[:, 1:2]
                total_loss += torch.mean(v_boundary**2)
        
        return total_loss
    
    def train(self, epochs=5000):
        """훈련"""
        print(f"🚀 간단한 실린더 유동 PINN 훈련 시작 ({epochs} epochs)")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # 샘플링
            samples = self.sample_points()
            
            # 손실 계산
            physics_loss = self.physics_loss(samples['interior'])
            boundary_loss = self.boundary_loss(samples)
            
            total_loss = physics_loss + 10.0 * boundary_loss  # 경계조건 강화
            
            # 역전파
            total_loss.backward()
            self.optimizer.step()
            
            # 진행률 출력
            if epoch % 500 == 0:
                print(f"Epoch {epoch:4d}: Total={total_loss.item():.2e}, "
                      f"Physics={physics_loss.item():.2e}, "
                      f"Boundary={boundary_loss.item():.2e}")
        
        print("✅ 훈련 완료!")
    
    def visualize(self):
        """결과 시각화"""
        print("🎨 결과 시각화 중...")
        
        # 고해상도 그리드
        x = np.linspace(self.x_min, self.x_max, 200)
        y = np.linspace(self.y_min, self.y_max, 150)
        X, Y = np.meshgrid(x, y)
        
        # 실린더 외부만 마스킹
        mask = (X**2 + Y**2) > self.cylinder_radius**2
        
        # 예측
        points_grid = torch.tensor(
            np.stack([np.zeros_like(X).flatten(), X.flatten(), Y.flatten()], axis=1),
            dtype=torch.float32
        )
        
        with torch.no_grad():
            pred = self.model(points_grid)
        
        u_pred = pred[:, 0].numpy().reshape(X.shape)
        v_pred = pred[:, 1].numpy().reshape(X.shape)
        p_pred = pred[:, 2].numpy().reshape(X.shape)
        
        # 실린더 내부 마스킹
        u_pred = np.where(mask, u_pred, np.nan)
        v_pred = np.where(mask, v_pred, np.nan)
        p_pred = np.where(mask, p_pred, np.nan)
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 속도 크기
        velocity_mag = np.sqrt(u_pred**2 + v_pred**2)
        im1 = axes[0, 0].contourf(X, Y, velocity_mag, levels=50, cmap='viridis')
        axes[0, 0].set_title('Velocity Magnitude', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 압력
        im2 = axes[0, 1].contourf(X, Y, p_pred, levels=50, cmap='coolwarm')
        axes[0, 1].set_title('Pressure', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 유선
        axes[1, 0].streamplot(X, Y, u_pred, v_pred, density=2, color=velocity_mag, cmap='plasma')
        axes[1, 0].set_title('Streamlines', fontsize=14, fontweight='bold')
        
        # 와도
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dudy = np.gradient(u_pred, dy, axis=0)
        dvdx = np.gradient(v_pred, dx, axis=1)
        vorticity = dvdx - dudy
        
        im4 = axes[1, 1].contourf(X, Y, vorticity, levels=50, cmap='RdBu_r')
        axes[1, 1].set_title('Vorticity', fontsize=14, fontweight='bold')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # 실린더 표시
        theta = np.linspace(0, 2*np.pi, 100)
        x_cyl = self.cylinder_radius * np.cos(theta)
        y_cyl = self.cylinder_radius * np.sin(theta)
        
        for ax in axes.flat:
            ax.fill(x_cyl, y_cyl, color='black', alpha=0.8)
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        results_dir = Path(r"C:\Users\Master\Desktop\NACA0012_Flutter_PINN\results")
        save_path = results_dir / "simple_cylinder_flow.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 시각화 저장: {save_path}")
        
        # 후류 체크
        wake_x = 2.0  # 실린더 뒤쪽
        wake_indices = np.where(np.abs(X[75, :] - wake_x) < 0.1)[0]
        if len(wake_indices) > 0:
            wake_idx = wake_indices[0]
            wake_velocity = u_pred[75, wake_idx]  # 중심선 속도
            velocity_deficit = 1.0 - wake_velocity
            
            print(f"\n🌊 후류 분석:")
            print(f"   위치 x = {wake_x}")
            print(f"   중심선 속도: {wake_velocity:.3f}")
            print(f"   속도 부족: {velocity_deficit:.3f}")
            
            if velocity_deficit > 0.1:
                print("✅ 후류가 잘 형성됨!")
            else:
                print("⚠️ 후류가 약함")

def main():
    """메인 함수"""
    print("🎯 초간단 원형 실린더 유동 PINN")
    print("복잡한 NACA 0012 대신 검증된 간단한 문제부터 시작")
    print("="*60)
    
    # PINN 생성 및 훈련
    pinn = CircularCylinderPINN()
    pinn.train(epochs=3000)  # 빠른 테스트
    pinn.visualize()
    
    print(f"\n💡 이 결과가 만족스럽다면:")
    print("1. NACA 0012로 다시 도전")
    print("2. 더 복잡한 설정 적용")
    print("3. 실제 데이터와 비교")
    
    print(f"\n❌ 이 결과도 별로라면:")
    print("1. PINN 자체에 문제가 있음")
    print("2. 다른 접근법 고려 필요")

if __name__ == "__main__":
    main()