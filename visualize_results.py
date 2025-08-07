#!/usr/bin/env python3
"""
NACA 0012 Flutter PINN 결과 시각화 스크립트
학습된 모델로 유동장, 구조 응답, 애니메이션 등 모든 시각화 생성
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple

# 프로젝트 모듈 import
from config import *
from pinn_model import create_pinn_model
from boundary_conditions import create_boundary_manager
from structure_dynamics import create_structure_system
from data_io import DataProcessor
from utils import *

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class PINNVisualizer:
    """PINN 결과 시각화 클래스"""
    
    def __init__(self):
        """초기화"""
        # 설정 로드
        self.phys_params = PhysicalParameters()
        self.domain_params = DomainParameters()
        self.pinn_config = PINNConfig()
        self.training_config = TrainingConfig()
        self.file_config = FileConfig()
        
        # 결과 디렉터리
        self.results_dir = Path(r"C:\Users\Master\Desktop\NACA0012_Flutter_PINN\results")
        
        # 모델 로드
        self.model = None
        self.load_trained_model()
        
        # 구성 요소 초기화
        self.data_processor = DataProcessor(
            self.phys_params, self.domain_params, self.file_config
        )
        
        self.bc_manager = create_boundary_manager(
            self.phys_params, self.domain_params, "lab"
        )
        
        print("✅ PINN 시각화 도구 초기화 완료")
    
    def load_trained_model(self):
        """학습된 모델 로드"""
        # best_model.pt가 없으면 최신 checkpoint 사용
        model_path = self.results_dir / "best_model.pt"
        if not model_path.exists():
            model_path = self.results_dir / "checkpoint_epoch_4999.pt"
            print(f"📁 최신 체크포인트 사용: {model_path}")
        
        if not model_path.exists():
            print("❌ 모델 파일을 찾을 수 없습니다!")
            return
        
        # 모델 생성
        self.model = create_pinn_model(self.pinn_config)
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ 모델 로드 완료: 에포크 {checkpoint.get('epoch', 'unknown')}")
        print(f"📊 최종 손실: {checkpoint.get('loss', 'unknown'):.6f}")
    
    def create_prediction_grid(self, t: float = 0.0, nx: int = 100, ny: int = 80) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """예측용 그리드 생성"""
        # 비차원 도메인 경계
        bounds = self.domain_params.get_nondim_bounds(self.phys_params.C_phys)
        
        # 그리드 생성
        x = np.linspace(bounds['x_min'], bounds['x_max'], nx)
        y = np.linspace(bounds['y_min'], bounds['y_max'], ny)
        X, Y = np.meshgrid(x, y)
        
        # 시간 좌표 추가
        T = np.full_like(X, t)
        
        # 텐서로 변환
        grid_points = torch.tensor(
            np.stack([T.flatten(), X.flatten(), Y.flatten()], axis=1),
            dtype=torch.float32
        )
        
        return grid_points, X, Y
    
    def predict_flow_field(self, t: float = 0.0) -> Dict[str, np.ndarray]:
        """유동장 예측"""
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다!")
            return {}
        
        print(f"🔮 시간 t={t:.3f}에서 유동장 예측 중...")
        
        # 그리드 생성
        grid_points, X, Y = self.create_prediction_grid(t)
        
        # 예측 수행
        with torch.no_grad():
            predictions = self.model(grid_points)
        
        # 결과 재구성
        ny, nx = X.shape
        u_pred = predictions[:, 0].cpu().numpy().reshape(ny, nx)
        v_pred = predictions[:, 1].cpu().numpy().reshape(ny, nx)
        p_pred = predictions[:, 2].cpu().numpy().reshape(ny, nx)
        
        return {
            'x_grid': X,
            'y_grid': Y,
            'u': u_pred,
            'v': v_pred,
            'p': p_pred
        }
    
    def plot_flow_field(self, flow_data: Dict[str, np.ndarray], save_path: str = None):
        """유동장 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        X, Y = flow_data['x_grid'], flow_data['y_grid']
        u, v, p = flow_data['u'], flow_data['v'], flow_data['p']
        
        # 속도 크기 계산
        velocity_magnitude = np.sqrt(u**2 + v**2)
        
        # 1. 속도 크기 contour
        im1 = axes[0, 0].contourf(X, Y, velocity_magnitude, levels=50, cmap='viridis')
        axes[0, 0].set_title('Velocity Magnitude |V|', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('x/C')
        axes[0, 0].set_ylabel('y/C')
        fig.colorbar(im1, ax=axes[0, 0])
        
        # 2. 압력 contour
        im2 = axes[0, 1].contourf(X, Y, p, levels=50, cmap='RdBu_r')
        axes[0, 1].set_title('Pressure p', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('x/C')
        axes[0, 1].set_ylabel('y/C')
        fig.colorbar(im2, ax=axes[0, 1])
        
        # 3. 유선 (Streamlines)
        axes[1, 0].streamplot(X, Y, u, v, density=2, color=velocity_magnitude, cmap='plasma')
        axes[1, 0].set_title('Streamlines', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('x/C')
        axes[1, 0].set_ylabel('y/C')
        
        # 4. 와도 (Vorticity)
        # 격자 간격 계산
        dx = X[0, 1] - X[0, 0]  # x 방향 격자 간격
        dy = Y[1, 0] - Y[0, 0]  # y 방향 격자 간격
        
        # 속도 구배 계산
        dudy = np.gradient(u, dy, axis=0)  # du/dy
        dvdx = np.gradient(v, dx, axis=1)  # dv/dx
        vorticity = dvdx - dudy
        
        im4 = axes[1, 1].contourf(X, Y, vorticity, levels=50, cmap='RdBu_r')
        axes[1, 1].set_title('Vorticity ωz', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('x/C')
        axes[1, 1].set_ylabel('y/C')
        fig.colorbar(im4, ax=axes[1, 1])
        
        # 에어포일 윤곽선 추가
        try:
            airfoil_x, airfoil_y = self.bc_manager.geometry.get_surface_points(100)
            for ax in axes.flat:
                ax.plot(airfoil_x, airfoil_y, 'k-', linewidth=2, label='NACA 0012')
                ax.set_xlim(-0.5, 1.5)
                ax.set_ylim(-0.8, 0.8)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
        except:
            print("⚠️ 에어포일 윤곽선 표시 실패")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 유동장 시각화 저장: {save_path}")
        
        plt.show()
    
    def create_swirling_strength_animation(self, time_points: np.ndarray = None):
        """와류 강도 애니메이션 생성"""
        if time_points is None:
            time_points = np.linspace(0, 1, 20)  # 20프레임
        
        print(f"🎬 와류 강도 애니메이션 생성 중... ({len(time_points)} 프레임)")
        
        # 각 시간 프레임별 데이터 생성
        frames_data = []
        for i, t in enumerate(time_points):
            print(f"  프레임 {i+1}/{len(time_points)}: t={t:.3f}")
            flow_data = self.predict_flow_field(t)
            if flow_data:
                frames_data.append(flow_data)
        
        if not frames_data:
            print("❌ 애니메이션 데이터 생성 실패")
            return
        
        # 애니메이션 저장 경로
        animation_path = self.results_dir / "swirling_strength_animation.mp4"
        
        try:
            import matplotlib.animation as animation
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            def animate(frame_idx):
                ax.clear()
                flow_data = frames_data[frame_idx]
                
                # 와류 강도 계산 (simplified)
                X, Y = flow_data['x_grid'], flow_data['y_grid']
                u, v = flow_data['u'], flow_data['v']
                
                # 속도 구배 계산 (와도)
                dx = X[0, 1] - X[0, 0]  # x 방향 격자 간격
                dy = Y[1, 0] - Y[0, 0]  # y 방향 격자 간격
                
                dudy = np.gradient(u, dy, axis=0)  # du/dy
                dvdx = np.gradient(v, dx, axis=1)  # dv/dx
                vorticity = dvdx - dudy
                
                # 와류 강도 시각화
                im = ax.contourf(X, Y, vorticity, levels=50, cmap='RdBu_r')
                ax.set_title(f'Vorticity at t={time_points[frame_idx]:.3f}', fontsize=14, fontweight='bold')
                ax.set_xlabel('x/C')
                ax.set_ylabel('y/C')
                ax.set_xlim(-0.5, 1.5)
                ax.set_ylim(-0.8, 0.8)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                
                # 에어포일 윤곽선
                try:
                    airfoil_x, airfoil_y = self.bc_manager.geometry.get_surface_points(100)
                    ax.plot(airfoil_x, airfoil_y, 'k-', linewidth=2)
                except:
                    pass
                
                return [im]
            
            anim = animation.FuncAnimation(
                fig, animate, frames=len(frames_data),
                interval=200, blit=False, repeat=True
            )
            
            # MP4로 저장
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=5, metadata=dict(artist='PINN'), bitrate=1800)
            anim.save(str(animation_path), writer=writer)
            
            plt.close()
            print(f"✅ 애니메이션 저장 완료: {animation_path}")
            
        except Exception as e:
            print(f"⚠️ 애니메이션 저장 실패: {e}")
            print("   (ffmpeg가 설치되지 않았거나 다른 문제일 수 있습니다)")
    
    def plot_loss_curves(self):
        """손실 곡선 시각화"""
        loss_path = self.results_dir / "loss_history.json"
        if not loss_path.exists():
            print("❌ loss_history.json 파일을 찾을 수 없습니다!")
            return
        
        with open(loss_path, 'r') as f:
            loss_data = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = loss_data['epochs']
        
        # 총 손실
        axes[0, 0].plot(epochs, loss_data['total'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # PDE 손실
        axes[0, 1].plot(epochs, loss_data['pde'], 'r-', linewidth=2)
        axes[0, 1].set_title('PDE Residual Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # 경계조건 손실
        axes[1, 0].plot(epochs, loss_data['bc'], 'g-', linewidth=2)
        axes[1, 0].set_title('Boundary Condition Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # FSI 손실
        axes[1, 1].plot(epochs, loss_data['fsi'], 'm-', linewidth=2)
        axes[1, 1].set_title('FSI Coupling Loss', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        # 저장
        save_path = self.results_dir / "loss_curves_detailed.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 상세 손실 곡선 저장: {save_path}")
    
    def create_complete_visualization(self):
        """완전한 시각화 생성"""
        print("🎨 완전한 PINN 결과 시각화 시작")
        print("="*50)
        
        # 1. 유동장 시각화
        flow_data = self.predict_flow_field(t=0.0)
        if flow_data:
            flow_save_path = self.results_dir / "flow_field_complete.png"
            self.plot_flow_field(flow_data, str(flow_save_path))
        
        # 2. 손실 곡선
        self.plot_loss_curves()
        
        # 3. 와류 애니메이션 (간단 버전)
        try:
            self.create_swirling_strength_animation()
        except Exception as e:
            print(f"⚠️ 애니메이션 생성 실패: {e}")
        
        # 4. 요약 보고서
        self.create_summary_report()
        
        print("\n🎉 모든 시각화 완료!")
        print(f"결과는 {self.results_dir} 폴더에서 확인할 수 있습니다.")
    
    def create_summary_report(self):
        """상세 요약 보고서 생성"""
        report = f"""
🎯 NACA 0012 Flutter PINN 완전한 결과 분석
========================================

📊 학습 정보:
- 총 에포크: 5000 (AdamW)
- 모델 파라미터: {sum(p.numel() for p in self.model.parameters()):,}개
- 배치 크기: {self.training_config.batch_size}
- 학습률: {self.training_config.adam_lr}

🎨 생성된 시각화:
- flow_field_complete.png: 완전한 유동장 시각화
- loss_curves_detailed.png: 상세 손실 곡선
- swirling_strength_animation.mp4: 와류 강도 애니메이션
- sampling_distribution.png: 샘플링 분포
- data_distribution.png: 데이터 분포

📈 물리적 파라미터:
- Reynolds 수: {self.phys_params.Re}
- 코드 길이: {self.phys_params.C_phys:.3f} m
- 도메인 크기: {self.domain_params.x_max_phys - self.domain_params.x_min_phys:.1f} × {self.domain_params.y_max_phys - self.domain_params.y_min_phys:.1f} m

🔬 분석 권장사항:
1. flow_field_complete.png에서 에어포일 주변 유동 패턴 확인
2. loss_curves_detailed.png에서 수렴성 분석
3. 애니메이션으로 시간에 따른 와류 변화 관찰
4. 추가 시간점에서의 예측 수행

🎉 Physics-Informed Neural Network 학습 및 시각화 완료!
========================================
"""
        
        report_path = self.results_dir / "complete_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"📄 상세 보고서 저장: {report_path}")

def main():
    """메인 함수"""
    # 시각화 도구 초기화
    visualizer = PINNVisualizer()
    
    # 완전한 시각화 실행
    visualizer.create_complete_visualization()

if __name__ == "__main__":
    main()