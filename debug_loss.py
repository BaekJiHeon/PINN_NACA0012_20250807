#!/usr/bin/env python3
"""
PINN 손실 분석 및 디버깅 스크립트
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_loss_components():
    """손실 성분 분석"""
    results_dir = Path(r"C:\Users\Master\Desktop\NACA0012_Flutter_PINN\results")
    loss_path = results_dir / "loss_history.json"
    
    if not loss_path.exists():
        print("❌ loss_history.json 파일을 찾을 수 없습니다!")
        return
    
    with open(loss_path, 'r') as f:
        loss_data = json.load(f)
    
    epochs = np.array(loss_data['epochs'])
    total_loss = np.array(loss_data['total'])
    pde_loss = np.array(loss_data['pde'])
    bc_loss = np.array(loss_data['bc'])
    fsi_loss = np.array(loss_data['fsi'])
    
    print("🔍 손실 성분 분석")
    print("="*50)
    print(f"총 에포크: {len(epochs)}")
    
    if len(epochs) > 0:
        print(f"\n📊 최종 손실 값 (에포크 {epochs[-1]}):")
        print(f"   총 손실: {total_loss[-1]:.6f}")
        print(f"   PDE 손실: {pde_loss[-1]:.6f}")
        print(f"   BC 손실: {bc_loss[-1]:.6f}")
        print(f"   FSI 손실: {fsi_loss[-1]:.6f}")
        
        print(f"\n📈 손실 비율:")
        total_final = total_loss[-1]
        print(f"   PDE: {pde_loss[-1]/total_final*100:.1f}%")
        print(f"   BC: {bc_loss[-1]/total_final*100:.1f}%")
        print(f"   FSI: {fsi_loss[-1]/total_final*100:.1f}%")
        
        print(f"\n📉 손실 감소율 (처음 대비 마지막):")
        if total_loss[0] > 0:
            print(f"   총 손실: {total_loss[-1]/total_loss[0]:.3f}x")
        if pde_loss[0] > 0:
            print(f"   PDE: {pde_loss[-1]/pde_loss[0]:.3f}x")
        if bc_loss[0] > 0:
            print(f"   BC: {bc_loss[-1]/bc_loss[0]:.3f}x")
        if fsi_loss[0] > 0:
            print(f"   FSI: {fsi_loss[-1]/fsi_loss[0]:.3f}x")
    
    # 문제 진단
    print(f"\n🚨 문제 진단:")
    
    if fsi_loss[-1] > 1000:
        print("   ❌ FSI 손실이 너무 높음 (>1000)")
        print("      → 구조 동역학 가중치 조정 필요")
        print("      → 구조 데이터 스케일링 문제")
    
    if bc_loss[-1] > pde_loss[-1] * 10:
        print("   ⚠️ 경계조건 손실이 PDE보다 10배 이상 높음")
        print("      → 경계조건 가중치 조정 필요")
    
    if np.any(np.isnan(total_loss)) or np.any(np.isinf(total_loss)):
        print("   ❌ NaN 또는 Inf 값 발견")
        print("      → 수치적 불안정성 존재")
    
    # 수렴성 분석
    if len(epochs) > 100:
        recent_loss = total_loss[-100:]
        loss_std = np.std(recent_loss)
        loss_mean = np.mean(recent_loss)
        
        print(f"\n📊 수렴성 분석 (최근 100 에포크):")
        print(f"   평균 손실: {loss_mean:.6f}")
        print(f"   표준편차: {loss_std:.6f}")
        print(f"   변동계수: {loss_std/loss_mean:.4f}")
        
        if loss_std/loss_mean < 0.01:
            print("   ✅ 잘 수렴됨")
        elif loss_std/loss_mean < 0.1:
            print("   ⚠️ 부분적으로 수렴")
        else:
            print("   ❌ 수렴하지 않음")

def suggest_fixes():
    """수정 제안"""
    print(f"\n🔧 권장 수정사항:")
    print("="*50)
    
    print("1️⃣ FSI 손실 문제 해결:")
    print("   - lambda_fsi를 0.001로 감소")
    print("   - 구조 데이터 정규화")
    print("   - 구조 동역학 손실 단순화")
    
    print("\n2️⃣ 경계조건 개선:")
    print("   - lambda_bc를 0.1로 감소") 
    print("   - 표면 경계조건 검증")
    print("   - 경계점 샘플링 증가")
    
    print("\n3️⃣ 학습 안정화:")
    print("   - 학습률을 5e-4로 감소")
    print("   - 그래디언트 클리핑 추가")
    print("   - 조기 종료 추가")
    
    print("\n4️⃣ 데이터 전처리:")
    print("   - 모든 데이터 정규화")
    print("   - 차원 분석 확인")
    print("   - 물리적 단위 통일")

def create_improved_config():
    """개선된 설정 파일 생성"""
    config_content = '''
# 개선된 PINN 설정
# FSI 손실 문제 해결을 위한 수정된 파라미터들

class TrainingConfig:
    # 학습률 감소
    adam_lr: float = 5e-4  # 1e-3에서 감소
    
    # 손실 가중치 재조정
    lambda_data: float = 1.0
    lambda_pde: float = 1.0  
    lambda_bc: float = 0.1   # 1.0에서 감소
    lambda_fsi: float = 0.001  # 1.0에서 대폭 감소
    
    # 그래디언트 클리핑
    grad_clip: float = 1.0
    
    # 조기 종료
    early_stopping: bool = True
    patience: int = 500
'''
    
    config_path = Path(r"C:\Users\Master\Desktop\NACA0012_Flutter_PINN\improved_config.txt")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"📄 개선된 설정 저장: {config_path}")

def main():
    """메인 함수"""
    print("🔍 PINN 손실 분석 및 디버깅")
    print("="*50)
    
    analyze_loss_components()
    suggest_fixes()
    create_improved_config()
    
    print(f"\n💡 다음 단계:")
    print("1. config.py에서 lambda_fsi = 0.001로 수정")
    print("2. lambda_bc = 0.1로 수정") 
    print("3. adam_lr = 5e-4로 수정")
    print("4. 모델 재학습 실행")

if __name__ == "__main__":
    main()