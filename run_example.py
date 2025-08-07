#!/usr/bin/env python3
"""
NACA 0012 Flutter PINN - 실행 예제 스크립트
Quick start examples for the NACA 0012 Flutter PINN project
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description=""):
    """명령어 실행"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"명령어: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print("출력:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 오류 발생: {e}")
        if e.stderr:
            print("오류 메시지:")
            print(e.stderr)
        return False

def check_dependencies():
    """의존성 확인"""
    print("📦 의존성 확인 중...")
    
    required_packages = [
        'torch', 'numpy', 'matplotlib', 'scipy', 'pandas',
        'scikit-learn', 'tqdm', 'loguru'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (누락)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 누락된 패키지: {', '.join(missing_packages)}")
        print("다음 명령으로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ 모든 의존성이 설치되어 있습니다!")
    return True

def example_1_quick_training():
    """예제 1: 빠른 학습 테스트"""
    cmd = """python main.py \
        --Re 1000 \
        --coord_sys lab \
        --adam_epochs 100 \
        --lbfgs_epochs 50 \
        --batch_size 1024 \
        --num_layers 6 \
        --num_neurons 64"""
    
    return run_command(cmd, "예제 1: 빠른 학습 테스트 (5분 내외)")

def example_2_full_training():
    """예제 2: 완전한 학습"""
    cmd = """python main.py \
        --Re 1000 \
        --coord_sys lab \
        --activation tanh \
        --num_layers 10 \
        --num_neurons 128 \
        --adam_epochs 5000 \
        --lbfgs_epochs 2000 \
        --batch_size 2048 \
        --lr 1e-3"""
    
    return run_command(cmd, "예제 2: 완전한 학습 (2-3시간) - 5000 에포크 강화!")

def example_3_siren_network():
    """예제 3: SIREN 네트워크"""
    cmd = """python main.py \
        --Re 1500 \
        --coord_sys body \
        --activation siren \
        --num_layers 8 \
        --num_neurons 96 \
        --adam_epochs 4000 \
        --lbfgs_epochs 1500 \
        --lr 5e-4"""
    
    return run_command(cmd, "예제 3: SIREN 활성화 함수 (고주파 특징 포착) - 강화!")

def example_4_parameter_identification():
    """예제 4: 파라미터 식별"""
    cmd = """python main.py \
        --Re 1000 \
        --inverse_id \
        --adam_epochs 6000 \
        --lbfgs_epochs 3000 \
        --lr 1e-4"""
    
    return run_command(cmd, "예제 4: 구조 파라미터 식별 (초고정밀 6000 에포크!)")

def example_5_inference_flow():
    """예제 5: 유동장 추론"""
    # 먼저 모델이 있는지 확인
    model_path = "results/model.pt"
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        print("먼저 학습을 완료하세요.")
        return False
    
    cmd = f"""python inference.py \
        --model {model_path} \
        --mode flow \
        --time_start 0 \
        --time_end 5.0 \
        --n_time_points 50 \
        --resolution 100 80"""
    
    return run_command(cmd, "예제 5: 유동장 예측")

def example_6_flutter_analysis():
    """예제 6: 플러터 분석"""
    model_path = "results/model.pt"
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        return False
    
    cmd = f"""python inference.py \
        --model {model_path} \
        --mode flutter \
        --time_start 0 \
        --time_end 10.0 \
        --n_time_points 1000"""
    
    return run_command(cmd, "예제 6: 플러터 응답 분석")

def example_7_parameter_study():
    """예제 7: 파라미터 스터디"""
    model_path = "results/model.pt"
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        return False
    
    cmd = f"""python inference.py \
        --model {model_path} \
        --mode param_study \
        --param_name k_h \
        --param_range 400 600 21"""
    
    return run_command(cmd, "예제 7: 강성 파라미터 스터디")

def example_8_animation():
    """예제 8: 애니메이션 생성"""
    model_path = "results/model.pt"
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        return False
    
    cmd = f"""python inference.py \
        --model {model_path} \
        --mode animation \
        --time_start 0 \
        --time_end 5.0 \
        --n_time_points 100 \
        --resolution 120 100"""
    
    return run_command(cmd, "예제 8: 플러터 애니메이션 생성")

def example_9_multi_reynolds():
    """예제 9: 다중 Reynolds 수 연구"""
    reynolds_numbers = [500, 1000, 1500, 2000]
    
    for re in reynolds_numbers:
        output_dir = f"results_Re{re}"
        cmd = f"""python main.py \
            --Re {re} \
            --adam_epochs 3000 \
            --lbfgs_epochs 1500 \
            --batch_size 1024 \
            --output_dir {output_dir}"""
        
        success = run_command(cmd, f"Reynolds 수 {re} 학습")
        if not success:
            print(f"⚠️ Re={re} 학습 실패")
            break
        
        time.sleep(2)  # 짧은 휴식

def example_10_custom_data():
    """예제 10: 사용자 데이터 사용"""
    # 더미 데이터 생성 예제
    print("🔧 사용자 정의 데이터 예제")
    print("✅ Wind_turnel_DATA 폴더의 실제 데이터 파일들:")
    print("   - mesh center postion (CFD 데이터)")
    print("   - Node postion (메시 데이터)")  
    print("   - Damping_data.csv (구조 응답)")
    
    example_cmd = """python main.py \\
        --cfd_csv "Wind_turnel_DATA/mesh center postion" \\
        --mesh_csv "Wind_turnel_DATA/Node postion" \\
        --damping_csv "Wind_turnel_DATA/Damping_data.csv" \\
        --Re 1200 \\
        --coord_sys lab"""
    
    print(f"\n기본 실행 (자동으로 위 파일들 사용):\npython main.py")
    print(f"\n또는 수동 지정:\n{example_cmd}")
    print("\n예상 파일 형식:")
    print("CFD: cell_id, x, y, p, u, v")
    print("MESH: node_id, x, y, p, u, v")
    print("DAMPING: time, h, theta, h_vel, theta_vel, Lift, Moment")

def main():
    """메인 함수"""
    print("""
    🚁 NACA 0012 Flutter PINN - 실행 예제
    =====================================
    
    이 스크립트는 다양한 학습 및 추론 예제를 제공합니다.
    원하는 예제를 선택하여 실행하세요.
    """)
    
    # 의존성 확인
    if not check_dependencies():
        print("\n❌ 의존성을 먼저 설치하세요.")
        return
    
    examples = {
        "1": ("빠른 학습 테스트 (5분)", example_1_quick_training),
        "2": ("완전한 학습 (2-3시간) 🔥5000 에포크!", example_2_full_training),
        "3": ("SIREN 네트워크 (4000 에포크 강화)", example_3_siren_network),
        "4": ("파라미터 식별 (6000 에포크 초고정밀)", example_4_parameter_identification),
        "5": ("유동장 추론", example_5_inference_flow),
        "6": ("플러터 분석", example_6_flutter_analysis),
        "7": ("파라미터 스터디", example_7_parameter_study),
        "8": ("애니메이션 생성", example_8_animation),
        "9": ("다중 Reynolds 수 (3000 에포크)", example_9_multi_reynolds),
        "10": ("사용자 데이터 예제", example_10_custom_data),
    }
    
    print("\n📋 사용 가능한 예제:")
    for key, (desc, _) in examples.items():
        print(f"  {key}. {desc}")
    
    print("\n선택 옵션:")
    print("  'all'  : 모든 예제 순차 실행")
    print("  'train': 학습 예제들 (1-4)")
    print("  'infer': 추론 예제들 (5-8)")
    print("  'q'    : 종료")
    
    while True:
        choice = input("\n예제를 선택하세요 (1-10, all, train, infer, q): ").strip().lower()
        
        if choice == 'q':
            print("👋 프로그램을 종료합니다.")
            break
        
        elif choice == 'all':
            print("🔄 모든 예제를 순차적으로 실행합니다...")
            for key in ['1', '2', '5', '6', '7', '8']:  # 시간이 오래 걸리는 것들 제외
                print(f"\n⏳ 예제 {key} 실행 중...")
                examples[key][1]()
                time.sleep(3)
        
        elif choice == 'train':
            print("🎓 학습 예제들을 실행합니다...")
            for key in ['1', '2']:  # 빠른 예제들만
                examples[key][1]()
                time.sleep(2)
        
        elif choice == 'infer':
            print("🔮 추론 예제들을 실행합니다...")
            for key in ['5', '6', '7', '8']:
                examples[key][1]()
                time.sleep(2)
        
        elif choice in examples:
            examples[choice][1]()
        
        else:
            print("❌ 잘못된 선택입니다. 다시 시도하세요.")

if __name__ == "__main__":
    main()