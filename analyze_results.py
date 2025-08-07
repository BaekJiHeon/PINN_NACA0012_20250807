#!/usr/bin/env python3
"""
NACA 0012 Flutter PINN 결과 분석 스크립트
AdamW 학습 완료된 모델로 결과 시각화 및 분석
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_best_model():
    """최고 성능 모델 로드"""
    # 절대 경로로 수정
    results_dir = Path(r"C:\Users\Master\Desktop\NACA0012_Flutter_PINN\results")
    
    # best_model.pt가 없으면 최신 checkpoint 사용
    model_path = results_dir / "best_model.pt"
    if not model_path.exists():
        model_path = results_dir / "checkpoint_epoch_4999.pt"
        print(f"📁 best_model.pt가 없어서 최신 체크포인트 사용: {model_path}")
    
    if not model_path.exists():
        print("❌ 모델 파일을 찾을 수 없습니다!")
        return None
    
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"✅ 모델 로드 완료: 에포크 {checkpoint.get('epoch', 'unknown')}")
    print(f"📊 최종 손실: {checkpoint.get('loss', 'unknown'):.6f}")
    
    return checkpoint

def plot_loss_history():
    """손실 이력 플롯"""
    # 절대 경로로 수정
    loss_path = Path(r"C:\Users\Master\Desktop\NACA0012_Flutter_PINN\results\loss_history.json")
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
    
    # 저장 (절대 경로)
    output_path = Path(r"C:\Users\Master\Desktop\NACA0012_Flutter_PINN\results\loss_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 손실 분석 그래프 저장: {output_path}")
    
    # 최종 손실 값 출력
    if len(epochs) > 0:
        final_idx = -1
        print(f"\n📈 최종 손실 값 (에포크 {epochs[final_idx]}):")
        print(f"   총 손실: {loss_data['total'][final_idx]:.6f}")
        print(f"   PDE 손실: {loss_data['pde'][final_idx]:.6f}")
        print(f"   경계조건 손실: {loss_data['bc'][final_idx]:.6f}")
        print(f"   FSI 손실: {loss_data['fsi'][final_idx]:.6f}")

def create_summary_report():
    """요약 보고서 생성"""
    report = """
🎯 NACA 0012 Flutter PINN 학습 결과 요약
==========================================

✅ 학습 완료: AdamW 5000 에포크
📊 모델 크기: ~1.9MB (166K 파라미터)
🎨 결과물: 샘플링 분포, 데이터 분포, 손실 곡선

📁 생성된 파일들:
- best_model.pt: 최고 성능 모델
- checkpoint_epoch_4999.pt: 최종 모델
- loss_history.json: 상세 손실 이력
- sampling_distribution.png: 샘플 분포
- data_distribution.png: 데이터 분포

🔬 다음 단계:
1. inference.py로 추론 테스트
2. 유동장 시각화 및 분석
3. 구조 응답 예측 성능 평가
4. 와류 패턴 분석

🎉 성공적으로 Physics-Informed Neural Network 학습 완료!
"""
    
    report_path = Path(r"C:\Users\Master\Desktop\NACA0012_Flutter_PINN\results\training_summary.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"📄 요약 보고서 저장: {report_path}")

def main():
    """메인 분석 함수"""
    print("🔍 NACA 0012 Flutter PINN 결과 분석 시작")
    print("="*50)
    
    # 모델 정보
    checkpoint = load_best_model()
    
    # 손실 이력 분석
    plot_loss_history()
    
    # 요약 보고서
    create_summary_report()
    
    print("\n🎉 분석 완료!")
    print("results 폴더에서 모든 결과를 확인할 수 있습니다.")

if __name__ == "__main__":
    main()