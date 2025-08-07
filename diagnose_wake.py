#!/usr/bin/env python3
"""
후류 문제 진단 스크립트
CFD 데이터와 PINN 예측에서 후류 영역 분석
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict

# 프로젝트 모듈 import
from config import *
from pinn_model import create_pinn_model
from data_io import DataProcessor
from boundary_conditions import create_boundary_manager

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class WakeDiagnoser:
    """후류 진단 클래스"""
    
    def __init__(self):
        """초기화"""
        self.phys_params = PhysicalParameters()
        self.domain_params = DomainParameters()
        self.pinn_config = PINNConfig()
        self.file_config = FileConfig()
        self.results_dir = Path(r"C:\Users\Master\Desktop\NACA0012_Flutter_PINN\results")
        
        # 모델 로드
        self.model = None
        self.load_model()
        
        # 데이터 처리기
        self.data_processor = DataProcessor(
            self.phys_params, self.domain_params, self.file_config
        )
        
        print("✅ 후류 진단 도구 초기화 완료")
    
    def load_model(self):
        """모델 로드"""
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
        
        self.model = create_pinn_model(self.pinn_config)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ 모델 로드: {model_path.name}")
    
    def analyze_original_cfd_data(self):
        """원본 CFD 데이터 후류 분석"""
        print("🔍 원본 CFD 데이터 후류 분석")
        print("="*50)
        
        try:
            # CFD 데이터 로드
            data = self.data_processor.load_all_data()
            cfd_data = data['cfd']
            
            # 후류 영역 정의 (에어포일 뒤쪽)
            wake_mask = (cfd_data['x'] > 1.0) & (cfd_data['x'] < 3.0) & \
                       (abs(cfd_data['y']) < 0.5)
            
            wake_data = cfd_data[wake_mask]
            
            print(f"📊 전체 CFD 점 개수: {len(cfd_data)}")
            print(f"📊 후류 영역 점 개수: {len(wake_data)}")
            print(f"📊 후류 비율: {len(wake_data)/len(cfd_data)*100:.1f}%")
            
            if len(wake_data) > 0:
                print(f"📈 후류 영역 속도 범위:")
                print(f"   U: {wake_data['u'].min():.3f} ~ {wake_data['u'].max():.3f}")
                print(f"   V: {wake_data['v'].min():.3f} ~ {wake_data['v'].max():.3f}")
                print(f"   |V|: {np.sqrt(wake_data['u']**2 + wake_data['v']**2).mean():.3f} (평균)")
                
                # 후류 시각화
                self.plot_cfd_wake_analysis(cfd_data, wake_data)
                return True
            else:
                print("❌ CFD 데이터에 후류 영역이 없습니다!")
                return False
                
        except Exception as e:
            print(f"❌ CFD 데이터 분석 실패: {e}")
            return False
    
    def plot_cfd_wake_analysis(self, cfd_data, wake_data):
        """CFD 후류 분석 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 전체 속도 분포
        scatter1 = axes[0, 0].scatter(cfd_data['x'], cfd_data['y'], 
                                     c=np.sqrt(cfd_data['u']**2 + cfd_data['v']**2), 
                                     s=1, cmap='viridis', alpha=0.6)
        axes[0, 0].set_title('CFD Data: Velocity Magnitude', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('x/C')
        axes[0, 0].set_ylabel('y/C')
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # 2. 후류 영역 강조
        axes[0, 1].scatter(cfd_data['x'], cfd_data['y'], c='lightgray', s=0.5, alpha=0.3)
        scatter2 = axes[0, 1].scatter(wake_data['x'], wake_data['y'], 
                                     c=np.sqrt(wake_data['u']**2 + wake_data['v']**2), 
                                     s=3, cmap='plasma')
        axes[0, 1].set_title('Wake Region Highlighted', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('x/C')
        axes[0, 1].set_ylabel('y/C')
        axes[0, 1].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Wake Start')
        axes[0, 1].legend()
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        # 3. 후류 중심선 속도 프로필
        if len(wake_data) > 0:
            # 중심선 근처 데이터 (|y| < 0.1)
            centerline_mask = abs(wake_data['y']) < 0.1
            centerline_data = wake_data[centerline_mask]
            
            if len(centerline_data) > 0:
                # x 위치별로 정렬
                centerline_sorted = centerline_data.sort_values('x')
                axes[1, 0].plot(centerline_sorted['x'], centerline_sorted['u'], 'bo-', 
                               markersize=4, linewidth=2, label='U velocity')
                axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Free stream')
                axes[1, 0].set_title('Wake Centerline Velocity Deficit', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('x/C')
                axes[1, 0].set_ylabel('U velocity')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 후류 폭 분석
        wake_cross_sections = []
        x_positions = np.linspace(1.2, 2.5, 8)
        
        for x_pos in x_positions:
            cross_section = wake_data[abs(wake_data['x'] - x_pos) < 0.1]
            if len(cross_section) > 5:
                cross_section_sorted = cross_section.sort_values('y')
                axes[1, 1].plot(cross_section_sorted['u'], cross_section_sorted['y'], 
                               'o-', alpha=0.7, markersize=3, 
                               label=f'x/C = {x_pos:.1f}')
        
        axes[1, 1].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Free stream')
        axes[1, 1].set_title('Wake Cross-sections', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('U velocity')
        axes[1, 1].set_ylabel('y/C')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        save_path = self.results_dir / "cfd_wake_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ CFD 후류 분석 저장: {save_path}")
    
    def compare_pinn_vs_cfd_wake(self):
        """PINN vs CFD 후류 비교"""
        print("🔄 PINN vs CFD 후류 비교")
        print("="*50)
        
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다!")
            return
        
        # 후류 영역 그리드 생성
        x_wake = np.linspace(1.0, 3.0, 100)
        y_wake = np.linspace(-0.5, 0.5, 50)
        X_wake, Y_wake = np.meshgrid(x_wake, y_wake)
        
        # 시간 좌표 (t=0)
        T_wake = np.zeros_like(X_wake)
        
        # PINN 예측
        grid_points = torch.tensor(
            np.stack([T_wake.flatten(), X_wake.flatten(), Y_wake.flatten()], axis=1),
            dtype=torch.float32
        )
        
        with torch.no_grad():
            predictions = self.model(grid_points)
        
        u_pinn = predictions[:, 0].cpu().numpy().reshape(X_wake.shape)
        v_pinn = predictions[:, 1].cpu().numpy().reshape(X_wake.shape)
        p_pinn = predictions[:, 2].cpu().numpy().reshape(X_wake.shape)
        
        # 비교 시각화
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # PINN 결과
        im1 = axes[0, 0].contourf(X_wake, Y_wake, u_pinn, levels=50, cmap='coolwarm')
        axes[0, 0].set_title('PINN: U Velocity in Wake', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].contourf(X_wake, Y_wake, v_pinn, levels=50, cmap='coolwarm')
        axes[0, 1].set_title('PINN: V Velocity in Wake', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=axes[0, 1])
        
        velocity_mag_pinn = np.sqrt(u_pinn**2 + v_pinn**2)
        im3 = axes[0, 2].contourf(X_wake, Y_wake, velocity_mag_pinn, levels=50, cmap='viridis')
        axes[0, 2].set_title('PINN: Velocity Magnitude', fontsize=14, fontweight='bold')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # CFD 데이터 (있는 경우)
        try:
            data = self.data_processor.load_all_data()
            cfd_data = data['cfd']
            
            wake_mask = (cfd_data['x'] > 1.0) & (cfd_data['x'] < 3.0) & \
                       (abs(cfd_data['y']) < 0.5)
            wake_cfd = cfd_data[wake_mask]
            
            if len(wake_cfd) > 0:
                scatter1 = axes[1, 0].scatter(wake_cfd['x'], wake_cfd['y'], c=wake_cfd['u'], 
                                            s=10, cmap='coolwarm')
                axes[1, 0].set_title('CFD: U Velocity in Wake', fontsize=14, fontweight='bold')
                plt.colorbar(scatter1, ax=axes[1, 0])
                
                scatter2 = axes[1, 1].scatter(wake_cfd['x'], wake_cfd['y'], c=wake_cfd['v'], 
                                            s=10, cmap='coolwarm')
                axes[1, 1].set_title('CFD: V Velocity in Wake', fontsize=14, fontweight='bold')
                plt.colorbar(scatter2, ax=axes[1, 1])
                
                vel_mag_cfd = np.sqrt(wake_cfd['u']**2 + wake_cfd['v']**2)
                scatter3 = axes[1, 2].scatter(wake_cfd['x'], wake_cfd['y'], c=vel_mag_cfd, 
                                            s=10, cmap='viridis')
                axes[1, 2].set_title('CFD: Velocity Magnitude', fontsize=14, fontweight='bold')
                plt.colorbar(scatter3, ax=axes[1, 2])
            else:
                for ax in axes[1, :]:
                    ax.text(0.5, 0.5, 'No CFD Wake Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=16)
                    ax.set_title('CFD Data (Not Available)', fontsize=14)
        
        except Exception as e:
            print(f"⚠️ CFD 데이터 로드 실패: {e}")
            for ax in axes[1, :]:
                ax.text(0.5, 0.5, f'CFD Load Error:\n{str(e)[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        for ax in axes.flat:
            ax.set_xlabel('x/C')
            ax.set_ylabel('y/C')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        save_path = self.results_dir / "pinn_vs_cfd_wake_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ PINN vs CFD 후류 비교 저장: {save_path}")
        
        # 정량적 분석
        print(f"\n📊 PINN 후류 분석:")
        print(f"   후류 중심선 속도 부족: {1.0 - u_pinn[25, 50:]:.3f} (평균)")
        print(f"   후류 최대 속도 부족: {1.0 - u_pinn.min():.3f}")
        
        if u_pinn.min() > 0.95:
            print("⚠️ 후류가 거의 형성되지 않음 - 학습 문제 가능성 높음")
        elif u_pinn.min() > 0.8:
            print("⚠️ 약한 후류 - 개선 필요")
        else:
            print("✅ 후류가 적절히 형성됨")
    
    def run_complete_wake_diagnosis(self):
        """완전한 후류 진단"""
        print("🔍 NACA 0012 후류 완전 진단")
        print("="*60)
        
        # 1. 원본 CFD 데이터 분석
        has_cfd_wake = self.analyze_original_cfd_data()
        
        # 2. PINN vs CFD 비교
        self.compare_pinn_vs_cfd_wake()
        
        # 3. 진단 결과 및 권장사항
        print(f"\n🎯 후류 진단 결과 및 권장사항:")
        print("="*60)
        
        if not has_cfd_wake:
            print("❌ 주요 문제: CFD 데이터에 후류 정보 부족")
            print("   → 해결책: 더 긴 도메인의 CFD 데이터 필요")
            print("   → 또는: 합성 후류 데이터 추가")
        else:
            print("✅ CFD 데이터에 후류 정보 존재")
            print("   → PINN 학습 방법 개선 필요:")
            print("     • 후류 영역 샘플링 가중치 증가")
            print("     • PDE 손실에서 운동량 보존 강화")
            print("     • 더 높은 Reynolds 수 학습")

def main():
    """메인 함수"""
    diagnoser = WakeDiagnoser()
    diagnoser.run_complete_wake_diagnosis()

if __name__ == "__main__":
    main()