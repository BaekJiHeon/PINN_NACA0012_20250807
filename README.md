# NACA 0012 Flutter PINN 🚁

**Physics-Informed Neural Networks for 2-DOF Flutter Analysis**

NACA 0012 에어포일의 2자유도 플러터 현상을 Physics-Informed Neural Network(PINN)으로 해석하는 고급 프로젝트입니다. 유동 PDE와 구조 ODE를 결합하여 와류 방출(vortex shedding) 및 양력/모멘트 응답을 동시에 재현합니다.

## 🎯 주요 특징

### 🔬 물리 모델링
- **2D 비압축성 Navier-Stokes 방정식** (ALE 기법 지원)
- **2-DOF 구조 동역학** (heave-pitch 결합 운동)
- **유동-구조 상호작용** (FSI) 결합
- **NACA 0012 에어포일** 정밀 형상 모델링

### 🧠 PINN 아키텍처
- **Fourier Features / SIREN** 활성화 함수 지원
- **적응적 손실 가중치** (GradNorm 기반)
- **다중 스케일 샘플링** 전략
- **앙상블 학습** (불확실성 정량화)

### 🎨 고급 기능
- **실시간 애니메이션** 생성
- **파라미터 식별** (inverse problem)
- **플러터 경계** 예측
- **상세한 시각화** 도구

## 📊 프로젝트 구조

```
NACA0012_Flutter_PINN/
│
├── 📋 requirements.txt          # 의존성
├── ⚙️ config.py                # 설정 관리
├── 📥 data_io.py               # 데이터 입출력 & 비차원화
├── 🧠 pinn_model.py            # PINN 모델 아키텍처
├── 📐 loss_functions.py        # 복합 손실 함수
├── 🔲 boundary_conditions.py   # 경계조건 관리
├── 🏗️ structure_dynamics.py    # 구조 동역학
├── 🎯 samplers.py              # 샘플링 전략
├── 🔧 utils.py                 # 유틸리티
├── 🚀 main.py                  # 메인 실행
├── 🔮 inference.py             # 추론 & 분석
└── 📖 README.md                # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 기본 학습 실행

```bash
# 기본 설정으로 학습 시작 (5000 에포크로 강화!)
python main.py --Re 1000 --coord_sys lab --adam_epochs 5000 --lbfgs_epochs 2000

# 고급 설정
python main.py \
    --Re 1500 \
    --coord_sys body \
    --activation siren \
    --num_layers 12 \
    --num_neurons 128 \
    --batch_size 4096 \
    --adam_epochs 6000 \
    --lr 5e-4
```

### 3. 추론 및 분석

```bash
# 유동장 예측
python inference.py --model results/model.pt --mode flow --time_end 10.0

# 플러터 응답 시뮬레이션
python inference.py --model results/model.pt --mode flutter --n_time_points 1000

# 파라미터 스터디
python inference.py --model results/model.pt --mode param_study \
    --param_name k_h --param_range 400 600 21

# 애니메이션 생성
python inference.py --model results/model.pt --mode animation \
    --time_end 5.0 --n_time_points 100 --resolution 120 100
```

## 📋 설정 옵션

### 물리 파라미터
- `--Re`: Reynolds 수 (기본값: 1000)
- `--coord_sys`: 좌표계 (`lab` 또는 `body`)

### 네트워크 구조
- `--activation`: 활성화 함수 (`tanh`, `siren`, `fourier`)
- `--num_layers`: 은닉층 수 (기본값: 10)
- `--num_neurons`: 뉴런 수 (기본값: 128)

### 학습 설정
- `--adam_epochs`: Adam 에포크 수 (기본값: 2000)
- `--lbfgs_epochs`: LBFGS 에포크 수 (기본값: 1000)
- `--lr`: 학습률 (기본값: 1e-3)
- `--batch_size`: 배치 크기 (기본값: 2048)

### 파일 경로
- `--cfd_csv`: CFD 데이터 경로
- `--mesh_csv`: 메시 데이터 경로  
- `--damping_csv`: 댐핑 데이터 경로 (UDF data_2dof.txt)

## 🎨 출력 파일

학습 완료 후 `results/` 디렉터리에 다음 파일들이 생성됩니다:

### 📊 분석 결과
- `loss_curves.png`: 손실 함수 곡선
- `data_distribution.png`: 입력 데이터 분포
- `sampling_distribution.png`: 샘플링 분포
- `flow_field.png`: 예측된 유동장
- `structural_response.png`: 구조 응답 분석

### 🎬 애니메이션
- `swirling_strength.mp4`: 와류 강도 애니메이션
- `flutter_animation.mp4`: 플러터 운동 애니메이션

### 📈 모니터링
- `system_monitoring.png`: 시스템 리소스 사용량
- `training.log`: 상세 학습 로그

### 💾 모델 파일
- `model.pt`: 최종 학습된 모델
- `best_model.pt`: 최고 성능 모델
- `checkpoint_epoch_*.pt`: 에포크별 체크포인트

## 🔬 고급 사용법

### 1. 실제 Wind Tunnel 데이터 사용 ✅

```bash
# 🎉 기본 실행 - 자동으로 Wind_turnel_DATA 폴더의 데이터 사용
python main.py

# 실제 파일 형식:
# CFD: Wind_turnel_DATA/mesh center postion (바이너리)
# MESH: Wind_turnel_DATA/Node postion 
#       (cellnumber, x-coordinate, y-coordinate, pressure, x-velocity, y-velocity)
# DAMPING: Wind_turnel_DATA/Damping_data.csv
#          (step, time, Lift, Moment, heave, theta, heave_vel, theta_vel)

# 수동으로 다른 파일 지정:
python main.py \
    --cfd_csv "your_path/your_cfd_data.csv" \
    --mesh_csv "your_path/your_mesh_data.csv" \
    --damping_csv "your_path/your_damping_data.csv"
```

### 2. 파라미터 식별

```python
# 구조 파라미터 학습
python main.py --inverse_id --adam_epochs 5000
```

### 3. 앙상블 학습

```python
# config.py에서 앙상블 설정 후
from pinn_model import create_pinn_model
ensemble_model = create_pinn_model(config, ensemble=True)
```

### 4. 배치 실행

```bash
# 여러 Reynolds 수 동시 실행
for re in 500 1000 1500 2000; do
    python main.py --Re $re --output_dir "results_Re$re" &
done
wait
```

## 📐 이론적 배경

### 지배 방정식

**2D 비압축성 Navier-Stokes:**
```
∂u/∂t + (u-w_x)∂u/∂x + (v-w_y)∂u/∂y = -∂p/∂x + ν∇²u
∂v/∂t + (u-w_x)∂v/∂x + (v-w_y)∂v/∂y = -∂p/∂y + ν∇²v
∂u/∂x + ∂v/∂y = 0
```

**2-DOF 구조 동역학:**
```
m ḧ + c_h ḣ + k_h h = -Lift
Iα θ̈ + c_α θ̇ + k_α θ = Moment
```

### 비차원화

- **길이**: `x' = x / C_phys` (C_phys = 0.156 m)
- **속도**: `u' = u / U_inf`
- **압력**: `p' = p / (ρ U_inf²)`
- **시간**: `t' = t * U_inf / C_phys`

### 손실 함수

```
L_total = λ_data·‖(u,v,p)_pred - data‖²
        + λ_PDE·‖R_NS‖²
        + λ_BC·‖BC 오류‖²
        + λ_FSI·[‖구조 ODE 잔차‖²]
```

## 🛠️ 문제 해결

### 일반적인 문제들

1. **메모리 부족**
   ```bash
   # 배치 크기 감소
   python main.py --batch_size 1024
   ```

2. **수렴 안됨**
   ```bash
   # 학습률 조정
   python main.py --lr 1e-4 --adam_epochs 5000
   ```

3. **CUDA 오류**
   ```bash
   # CPU 강제 사용
   export CUDA_VISIBLE_DEVICES=""
   python main.py
   ```

### 로그 확인

```bash
# 실시간 로그 모니터링
tail -f results/training.log

# 에러 패턴 검색
grep -i error results/training.log
```

## 🔍 성능 최적화

### GPU 최적화
- **AMP (Automatic Mixed Precision)** 활성화
- **배치 크기** 최대화 (메모리 허용 범위 내)
- **멀티 GPU** 지원

### 샘플링 최적화
- **적응적 샘플링** 활용
- **계층화 샘플링** (경계층/후류 집중)
- **Latin Hypercube** 샘플링

### 수치적 안정성
- **기울기 클리핑**
- **학습률 스케줄링**
- **가중치 정규화**

## 📚 참고 문헌

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. Dowell, E. H., & Hall, K. C. (2001). Modeling of fluid-structure interaction. *Annual Review of Fluid Mechanics*, 33(1), 445-490.

3. Sitzmann, V., Martel, J., Bergman, A., Lindell, D., & Wetzstein, G. (2020). Implicit neural representations with periodic activation functions. *Advances in Neural Information Processing Systems*, 33.

## 🤝 기여하기

이 프로젝트에 기여하고 싶으시다면:

1. 이슈 리포트
2. 기능 제안
3. 코드 개선
4. 문서화 향상

## 📄 라이선스

이 프로젝트는 연구 및 교육 목적으로 자유롭게 사용할 수 있습니다.

## 🎉 완성된 프로젝트!

이 NACA 0012 Flutter PINN 프로젝트는 다음을 모두 포함하는 완전한 구현입니다:

✅ **물리 기반 모델링**: 정확한 Navier-Stokes + 구조 동역학  
✅ **고급 PINN 아키텍처**: Fourier/SIREN, 적응적 샘플링  
✅ **실용적 도구들**: 시각화, 애니메이션, 모니터링  
✅ **확장 가능한 구조**: 모듈화된 설계, 쉬운 커스터마이징  
✅ **상세한 문서화**: 완전한 사용 가이드 및 이론 설명

학습을 시작하려면 단순히 `python main.py`를 실행하세요! 🚀

---

*Made with ❤️ for the CFD and Machine Learning community*