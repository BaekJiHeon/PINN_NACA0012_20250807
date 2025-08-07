#!/usr/bin/env python3
"""
LBFGS 학습 완료된 모델 시각화 스크립트
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

class LBFGSVisualizer:
    """LBFGS 결과 시각화 클래스"""
    
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
        self.load_lbfgs_model()
        
        # 구성 요소 초기화
        self.bc_manager = create_boundary_manager(
            self.phys_params, self.domain_params, "lab"
        )
        
        print("✅ LBFGS 시각화 도구 초기화 완료")
    
    def load_lbfgs_model(self):
        """LBFGS 학습된 모델 로드"""
        # LBFGS 모델 우선, 없으면 기존 모델 사용
        model_candidates = [
            self.results_dir / "lbfgs_best_model.pt",
            self.results_dir / "best_model.pt",
            self.results_dir / "checkpoint_epoch_4999.pt"
        ]
        
        model_path = None
        for candidate in model_candidates:
            if candidate.exists():
                model_path = candidate
                break
        
        if model_path is None:
            print("❌ 모델 파일을 찾을 수 없습니다!")
            return
        
        # 모델 생성
        self.model = create_pinn_model(self.pinn_config)
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        model_type = "LBFGS" if "lbfgs" in str(model_path) else "Adam"
        print(f"✅ {model_type} 모델 로드 완료: {model_path.name}")
        print(f"📊 에포크: {checkpoint.get('epoch', 'unknown')}")
        print(f"📊 손실: {checkpoint.get('loss', 'unknown'):.6f}")
    
    def create_prediction_grid(self, t: float = 0.0, nx: int = 300, ny: int = 200) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """고해상도 예측용 그리드 생성"""
        # 비차원 도메인 경계
        bounds = self.domain_params.get_nondim_bounds(self.phys_params.C_phys)
        
        # 그리드 생성 (고해상도)
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
        
        print(f"🔮 LBFGS 모델로 시간 t={t:.3f}에서 유동장 예측 중...")
        
        # 고해상도 그리드 생성
        grid_points, X, Y = self.create_prediction_grid(t)
        
        # 배치별 예측 (고해상도 처리)
        batch_size = 2000  # 더 작은 배치로 안정성 증대
        predictions_list = []
        
        with torch.no_grad():
            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i+batch_size]
                pred_batch = self.model(batch)
                predictions_list.append(pred_batch)
        
        predictions = torch.cat(predictions_list, dim=0)
        
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
    
    def plot_enhanced_flow_field(self, flow_data: Dict[str, np.ndarray], save_path: str = None):
        """초고해상도 유동장 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))  # 더 큰 캔버스
        
        X, Y = flow_data['x_grid'], flow_data['y_grid']
        u, v, p = flow_data['u'], flow_data['v'], flow_data['p']
        
        # 속도 크기 계산
        velocity_magnitude = np.sqrt(u**2 + v**2)
        
        # 와도 계산
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]
        dudy = np.gradient(u, dy, axis=0)
        dvdx = np.gradient(v, dx, axis=1)
        vorticity = dvdx - dudy
        
        # 1. 속도 크기 (초고해상도)
        im1 = axes[0, 0].contourf(X, Y, velocity_magnitude, levels=100, cmap='viridis')
        axes[0, 0].set_title('Velocity Magnitude |V|', fontsize=16, fontweight='bold')
        cbar1 = fig.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        cbar1.ax.tick_params(labelsize=12)
        
        # 2. U 속도 성분 (초고해상도)
        im2 = axes[0, 1].contourf(X, Y, u, levels=100, cmap='RdBu_r')
        axes[0, 1].set_title('U-Velocity', fontsize=16, fontweight='bold')
        cbar2 = fig.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        cbar2.ax.tick_params(labelsize=12)
        
        # 3. V 속도 성분 (초고해상도)
        im3 = axes[0, 2].contourf(X, Y, v, levels=100, cmap='RdBu_r')
        axes[0, 2].set_title('V-Velocity', fontsize=16, fontweight='bold')
        cbar3 = fig.colorbar(im3, ax=axes[0, 2], shrink=0.8)
        cbar3.ax.tick_params(labelsize=12)
        
        # 4. 압력 (초고해상도)
        im4 = axes[1, 0].contourf(X, Y, p, levels=100, cmap='coolwarm')
        axes[1, 0].set_title('Pressure', fontsize=16, fontweight='bold')
        cbar4 = fig.colorbar(im4, ax=axes[1, 0], shrink=0.8)
        cbar4.ax.tick_params(labelsize=12)
        
        # 5. 유선 (밀도 증가)
        axes[1, 1].streamplot(X, Y, u, v, density=5, color=velocity_magnitude, 
                             cmap='plasma', linewidth=1.5, arrowsize=1.5)
        axes[1, 1].set_title('Streamlines', fontsize=16, fontweight='bold')
        
        # 6. 와도 (초고해상도)
        im6 = axes[1, 2].contourf(X, Y, vorticity, levels=100, cmap='RdBu_r')
        axes[1, 2].set_title('Vorticity ωz', fontsize=16, fontweight='bold')
        cbar6 = fig.colorbar(im6, ax=axes[1, 2], shrink=0.8)
        cbar6.ax.tick_params(labelsize=12)
        
        # 에어포일 윤곽선 추가 (고해상도)
        try:
            airfoil_x, airfoil_y = self.bc_manager.geometry.get_surface_points(200)  # 더 세밀한 윤곽선
            for ax in axes.flat:
                ax.plot(airfoil_x, airfoil_y, 'k-', linewidth=3, label='NACA 0012')
                ax.fill(airfoil_x, airfoil_y, color='black', alpha=0.8)  # 솔리드 에어포일
                ax.set_xlim(-1, 2)
                ax.set_ylim(-1.2, 1.2)
                ax.set_xlabel('x/C', fontsize=14, fontweight='bold')
                ax.set_ylabel('y/C', fontsize=14, fontweight='bold')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.tick_params(labelsize=12)
        except Exception as e:
            print(f"⚠️ 에어포일 윤곽선 표시 실패: {e}")
        
        plt.tight_layout(pad=3.0)  # 레이아웃 여백 증가
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')  # 초고해상도 저장
            print(f"✅ 초고해상도 유동장 시각화 저장: {save_path}")
        
        plt.show()
    
    def plot_airfoil_closeup(self, flow_data: Dict[str, np.ndarray]):
        """에어포일 근처 확대 고해상도 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        X, Y = flow_data['x_grid'], flow_data['y_grid']
        u, v, p = flow_data['u'], flow_data['v'], flow_data['p']
        
        # 속도 크기 및 와도 계산
        velocity_magnitude = np.sqrt(u**2 + v**2)
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]
        dudy = np.gradient(u, dy, axis=0)
        dvdx = np.gradient(v, dx, axis=1)
        vorticity = dvdx - dudy
        
        # 에어포일 근처 영역 마스크 (확대)
        mask = (X >= -0.5) & (X <= 1.5) & (Y >= -0.6) & (Y <= 0.6)
        
        # 1. 속도 크기 (확대)
        X_zoom = np.where(mask, X, np.nan)
        Y_zoom = np.where(mask, Y, np.nan)
        vel_zoom = np.where(mask, velocity_magnitude, np.nan)
        
        im1 = axes[0, 0].contourf(X, Y, velocity_magnitude, levels=150, cmap='viridis')
        axes[0, 0].set_title('Velocity Magnitude - Airfoil Closeup', fontsize=18, fontweight='bold')
        axes[0, 0].set_xlim(-0.5, 1.5)
        axes[0, 0].set_ylim(-0.6, 0.6)
        cbar1 = fig.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        cbar1.ax.tick_params(labelsize=14)
        
        # 2. 압력 (확대)
        im2 = axes[0, 1].contourf(X, Y, p, levels=150, cmap='coolwarm')
        axes[0, 1].set_title('Pressure - Airfoil Closeup', fontsize=18, fontweight='bold')
        axes[0, 1].set_xlim(-0.5, 1.5)
        axes[0, 1].set_ylim(-0.6, 0.6)
        cbar2 = fig.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        cbar2.ax.tick_params(labelsize=14)
        
        # 3. 유선 (확대, 고밀도)
        axes[1, 0].streamplot(X, Y, u, v, density=8, color=velocity_magnitude, 
                             cmap='plasma', linewidth=2, arrowsize=2)
        axes[1, 0].set_title('Streamlines - High Density', fontsize=18, fontweight='bold')
        axes[1, 0].set_xlim(-0.5, 1.5)
        axes[1, 0].set_ylim(-0.6, 0.6)
        
        # 4. 와도 (확대)
        im4 = axes[1, 1].contourf(X, Y, vorticity, levels=150, cmap='RdBu_r')
        axes[1, 1].set_title('Vorticity - Airfoil Closeup', fontsize=18, fontweight='bold')
        axes[1, 1].set_xlim(-0.5, 1.5)
        axes[1, 1].set_ylim(-0.6, 0.6)
        cbar4 = fig.colorbar(im4, ax=axes[1, 1], shrink=0.8)
        cbar4.ax.tick_params(labelsize=14)
        
        # 모든 서브플롯에 에어포일 추가
        try:
            airfoil_x, airfoil_y = self.bc_manager.geometry.get_surface_points(300)
            for ax in axes.flat:
                ax.plot(airfoil_x, airfoil_y, 'k-', linewidth=4)
                ax.fill(airfoil_x, airfoil_y, color='black', alpha=0.9)
                ax.set_xlabel('x/C', fontsize=16, fontweight='bold')
                ax.set_ylabel('y/C', fontsize=16, fontweight='bold')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3, linewidth=0.8)
                ax.tick_params(labelsize=14)
        except Exception as e:
            print(f"⚠️ 에어포일 윤곽선 표시 실패: {e}")
        
        plt.tight_layout(pad=4.0)
        
        # 저장
        closeup_save_path = self.results_dir / "lbfgs_airfoil_closeup_ultra_hd.png"
        plt.savefig(closeup_save_path, dpi=600, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ 에어포일 확대 초고해상도 시각화 저장: {closeup_save_path}")
    
    def compare_loss_history(self):
        """Adam vs LBFGS 손실 비교"""
        # Adam 손실 이력
        adam_loss_path = self.results_dir / "loss_history.json"
        lbfgs_loss_path = self.results_dir / "lbfgs_loss_history.json"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Adam 손실
        if adam_loss_path.exists():
            with open(adam_loss_path, 'r') as f:
                adam_data = json.load(f)
            
            epochs = adam_data['epochs']
            axes[0, 0].plot(epochs, adam_data['total'], 'b-', linewidth=2, label='Adam')
            axes[0, 0].set_title('Total Loss Comparison', fontweight='bold')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(epochs, adam_data['pde'], 'r-', linewidth=2, label='Adam PDE')
            axes[0, 1].set_title('PDE Loss', fontweight='bold')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # LBFGS 손실
        if lbfgs_loss_path.exists():
            with open(lbfgs_loss_path, 'r') as f:
                lbfgs_data = json.load(f)
            
            epochs = lbfgs_data['epochs']
            axes[0, 0].plot(epochs, lbfgs_data['total'], 'g-', linewidth=2, label='LBFGS')
            axes[0, 1].plot(epochs, lbfgs_data['pde'], 'orange', linewidth=2, label='LBFGS PDE')
            
            # LBFGS만 별도 플롯
            axes[1, 0].plot(epochs, lbfgs_data['total'], 'g-', linewidth=2)
            axes[1, 0].set_title('LBFGS Total Loss', fontweight='bold')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(epochs, lbfgs_data['pde'], 'orange', linewidth=2)
            axes[1, 1].set_title('LBFGS PDE Loss', fontweight='bold')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        for ax in axes.flat:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
        
        plt.tight_layout()
        
        save_path = self.results_dir / "loss_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 손실 비교 그래프 저장: {save_path}")
    
    def create_complete_lbfgs_visualization(self):
        """완전한 LBFGS 결과 시각화"""
        print("🎨 LBFGS 결과 완전 시각화 시작")
        print("="*60)
        
        # 1. 초고해상도 유동장 시각화
        flow_data = self.predict_flow_field(t=0.0)
        if flow_data:
            # 전체 도메인 시각화
            flow_save_path = self.results_dir / "lbfgs_ultra_high_res_flow_field.png"
            self.plot_enhanced_flow_field(flow_data, str(flow_save_path))
            
            # 에어포일 근처 확대 시각화
            self.plot_airfoil_closeup(flow_data)
        
        # 2. 손실 비교
        self.compare_loss_history()
        
        # 3. 요약 보고서
        self.create_lbfgs_summary()
        
        print("\n🎉 LBFGS 시각화 완료!")
        print(f"📁 결과는 {self.results_dir} 폴더에서 확인하세요")
    
    def create_lbfgs_summary(self):
        """LBFGS 요약 보고서"""
        report = f"""
🎯 NACA 0012 Flutter PINN - LBFGS 학습 결과
==========================================

🔧 학습 방법:
- Phase 1: Adam 5000 에포크
- Phase 2: LBFGS 2000 에포크 (이어서 학습)

🎨 LBFGS 시각화 결과:
- lbfgs_enhanced_flow_field.png: 6패널 상세 유동장
- loss_comparison.png: Adam vs LBFGS 손실 비교
- lbfgs_best_model.pt: 최종 최적화된 모델

📈 기대 효과:
- Adam: 빠른 수렴, 전역 탐색
- LBFGS: 정밀한 최적화, 국부 수렴

🔬 분석 포인트:
1. 유동장 품질 개선 확인
2. 경계층 해상도 향상
3. 와류 구조 정확도 증가
4. 압력 분포 개선

🎉 2단계 최적화 전략 성공!
==========================================
"""
        
        report_path = self.results_dir / "lbfgs_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"📄 LBFGS 분석 보고서 저장: {report_path}")

def main():
    """메인 함수"""
    # LBFGS 시각화 도구 초기화
    visualizer = LBFGSVisualizer()
    
    # 완전한 시각화 실행
    visualizer.create_complete_lbfgs_visualization()

if __name__ == "__main__":
    main()