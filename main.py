"""
NACA 0012 Flutter PINN - 메인 실행 파일
Physics-Informed Neural Networks for 2-DOF Flutter Analysis

전체 PINN 학습 파이프라인:
1. 데이터 로딩 및 전처리
2. 모델 초기화
3. 샘플링 전략 설정
4. 학습 루프 실행
5. 결과 분석 및 저장
"""

import torch
import torch.optim as optim
import numpy as np
import os
import argparse
from typing import Dict, Tuple, List, Optional

# 프로젝트 모듈들
from config import (
    PhysicalParameters, DomainParameters, PINNConfig, 
    TrainingConfig, FileConfig, parse_args, create_config_from_args
)
from data_io import DataProcessor, load_and_process_data
from pinn_model import PINNModel, create_pinn_model, save_model
from loss_functions import CompositeLoss
from boundary_conditions import BoundaryConditionManager, create_boundary_manager
from structure_dynamics import TwoDOFStructure, AerodynamicLoads, create_structure_system
from samplers import CompositeSampler, create_composite_sampler
from utils import (
    initialize_utils, Timer, system_monitor, loss_tracker,
    get_device, print_model_summary, create_progress_bar,
    FlowFieldVisualizer, ResultsManager
)

class PINNTrainer:
    """PINN 학습 관리자"""
    
    def __init__(self, phys_params: PhysicalParameters,
                 domain_params: DomainParameters,
                 pinn_config: PINNConfig,
                 training_config: TrainingConfig,
                 file_config: FileConfig,
                 coord_sys: str = "lab"):
        
        self.phys_params = phys_params
        self.domain_params = domain_params
        self.pinn_config = pinn_config
        self.training_config = training_config
        self.file_config = file_config
        self.coord_sys = coord_sys
        
        # 디바이스 설정
        self.device = get_device()
        
        # 결과 관리자
        self.results_manager = ResultsManager(file_config.output_dir)
        
        # 구성 요소 초기화
        self._initialize_components()
        
    def _initialize_components(self):
        """구성 요소 초기화"""
        from loguru import logger
        from pathlib import Path
        
        # 출력 폴더 생성 (Windows 안전)
        output_path = Path(self.file_config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("🔧 구성 요소 초기화 중...")
        
        # 1. 데이터 처리기
        self.data_processor = DataProcessor(
            self.phys_params, self.domain_params, self.file_config
        )
        
        # 2. 경계조건 관리자
        self.bc_manager = create_boundary_manager(
            self.phys_params, self.domain_params, self.coord_sys
        )
        
        # 3. 구조 시스템
        self.structure = create_structure_system(self.phys_params)
        
        # 4. 공력 하중 계산기
        self.aero_loads = AerodynamicLoads(self.phys_params)
        
        # 5. 샘플러
        self.sampler = create_composite_sampler(
            self.phys_params, self.domain_params, self.training_config
        )
        
        # 6. PINN 모델
        self.model = create_pinn_model(self.pinn_config).to(self.device)
        print_model_summary(self.model)
        
        # 7. 손실 함수
        domain_bounds = self.domain_params.get_nondim_bounds(self.phys_params.C_phys)
        self.loss_function = CompositeLoss(
            self.phys_params, self.training_config, domain_bounds, self.coord_sys
        )
        
        # 8. 시각화 도구
        self.visualizer = FlowFieldVisualizer(domain_bounds)
        
        logger.info("✅ 구성 요소 초기화 완료")
    
    def prepare_data(self) -> Dict:
        """데이터 준비"""
        from loguru import logger
        
        with Timer("데이터 준비"):
            # CSV 데이터 로드
            processor, processed_data = load_and_process_data(
                self.phys_params, self.domain_params, self.file_config
            )
            
            # 데이터 분포 시각화
            processor.visualize_data_distribution(
                os.path.join(self.file_config.output_dir, "data_distribution.png")
            )
            
            logger.info("데이터 준비 완료")
            return processed_data
    
    def generate_training_samples(self) -> Dict[str, torch.Tensor]:
        """학습용 샘플 생성"""
        from loguru import logger
        
        with Timer("샘플링"):
            samples = self.sampler.generate_training_samples(
                n_collocation=self.training_config.batch_size,
                n_boundary=200,
                n_surface=150,
                strategy="stratified"
            )
            
            # 디바이스로 이동
            for key in samples:
                if isinstance(samples[key], torch.Tensor):
                    samples[key] = samples[key].to(self.device)
                elif isinstance(samples[key], dict):
                    for subkey in samples[key]:
                        if isinstance(samples[key][subkey], torch.Tensor):
                            samples[key][subkey] = samples[key][subkey].to(self.device)
            
            # 샘플 분포 시각화 (임시 비활성화)
            try:
                from pathlib import Path
                save_path = Path(self.file_config.output_dir) / "sampling_distribution.png"
                self.sampler.visualize_samples(
                    samples, save_path=str(save_path)
                )
            except Exception as e:
                logger.warning(f"⚠️ 시각화 저장 실패 (무시하고 진행): {e}")
            
            logger.info("Sampling completed")
            return samples
    
    def compute_physics_residuals(self, x_collocation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """물리 법칙 잔차 계산"""
        x_collocation.requires_grad_(True)
        
        # 모델 예측
        output = self.model(x_collocation)
        
        # 기울기 계산
        gradients = self.model.compute_gradients(x_collocation, output)
        
        return output, gradients
    
    def compute_boundary_residuals(self, boundary_points: Dict[str, torch.Tensor],
                                 structure_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """경계조건 잔차 계산"""
        bc_residuals = {}
        
        for bc_type, points in boundary_points.items():
            if len(points) > 0:
                points.requires_grad_(True)
                predictions = self.model(points)
                
                residual = self.bc_manager.apply_boundary_conditions(
                    points, predictions, bc_type, structure_state
                )
                bc_residuals[bc_type] = residual
        
        return bc_residuals
    
    def update_structure_state(self, t: float, damping_data: Dict) -> Dict[str, torch.Tensor]:
        """구조 상태 업데이트"""
        # 댐핑 데이터에서 현재 시간의 상태 보간
        time_data = damping_data['time']
        
        # 현재 시간에 가장 가까운 인덱스 찾기
        time_diff = np.abs(time_data - t)
        closest_idx = np.argmin(time_diff)
        
        structure_state = {
            'time': torch.tensor([t], device=self.device, requires_grad=True),
            'h': torch.tensor([damping_data['h'][closest_idx]], device=self.device, requires_grad=True),
            'theta': torch.tensor([damping_data['theta'][closest_idx]], device=self.device, requires_grad=True),
            'h_vel': torch.tensor([damping_data['h_vel'][closest_idx]], device=self.device, requires_grad=True),
            'theta_vel': torch.tensor([damping_data['theta_vel'][closest_idx]], device=self.device, requires_grad=True),
            'lift': torch.tensor([damping_data['Lift'][closest_idx]], device=self.device, requires_grad=True),
            'moment': torch.tensor([damping_data['Moment'][closest_idx]], device=self.device, requires_grad=True)
        }
        
        return structure_state
    
    def training_step(self, samples: Dict[str, torch.Tensor], 
                     processed_data: Dict, epoch: int) -> Dict[str, torch.Tensor]:
        """단일 학습 스텝"""
        
        # 현재 시간 (에포크 기반으로 순환)
        time_range = self.domain_params.t_end - self.domain_params.t_start
        current_time = (epoch % 100) / 100.0 * time_range + self.domain_params.t_start
        
        # 구조 상태 업데이트 - LBFGS용 gradient 보장
        structure_state = self.update_structure_state(current_time, processed_data['damping_nd'])
        
        # 모든 샘플에 대해 requires_grad 재설정 (LBFGS 안정성)
        for key, value in samples.items():
            if isinstance(value, torch.Tensor):
                value.requires_grad_(True)
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        subvalue.requires_grad_(True)
        
        # 1. PDE 잔차 계산
        output, gradients = self.compute_physics_residuals(samples['collocation'])
        
        # 2. 경계조건 잔차 계산
        boundary_dict = samples['boundary'].copy()
        boundary_dict['airfoil'] = samples['surface']  # surface를 airfoil boundary로 추가
        bc_residuals = self.compute_boundary_residuals(boundary_dict, structure_state)
        
        # 3. 총 손실 계산
        losses = self.loss_function.compute_total_loss(
            model=self.model,
            model_output=output,
            x_collocation=samples['collocation'],
            x_boundary=samples['boundary'],
            x_surface=samples['surface'],
            structure_data=structure_state,
            cfd_data={'bc_residuals': bc_residuals},
            gradients=gradients
        )
        
        return losses
    
    def train(self, processed_data: Dict):
        """메인 학습 루프"""
        from loguru import logger
        
        logger.info("PINN 학습 시작")
        
        # 옵티마이저 설정
        optimizer_adam = optim.AdamW(
            self.model.parameters(), 
            lr=self.training_config.adam_lr,
            weight_decay=1e-6
        )
        
        # 학습률 스케줄러
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_adam, mode='min', factor=0.8, patience=200
        )
        
        # AMP 설정
        scaler = torch.cuda.amp.GradScaler() if self.training_config.use_amp else None
        
        # Adam 학습
        logger.info(f"Phase 1: Adam 최적화 ({self.training_config.adam_epochs} epochs)")
        self._train_phase(optimizer_adam, self.training_config.adam_epochs, 
                         processed_data, scaler, scheduler, "Adam")
        
        # LBFGS 학습 (gradient 오류 방지)
        logger.info(f"Phase 2: LBFGS 최적화 ({self.training_config.lbfgs_epochs} epochs)")
        try:
            optimizer_lbfgs = optim.LBFGS(
                self.model.parameters(),
                lr=0.1,
                max_iter=20,
                max_eval=None,
                tolerance_grad=1e-7,
                tolerance_change=1e-9
            )
            self._train_phase(optimizer_lbfgs, self.training_config.lbfgs_epochs,
                             processed_data, None, None, "LBFGS")
        except RuntimeError as e:
            if "does not require grad" in str(e):
                logger.warning("⚠️ LBFGS gradient 오류 발생 - Adam 결과로 진행합니다")
                logger.warning(f"오류 내용: {e}")
            else:
                raise e
        
        logger.info("🎉 학습 완료!")
    
    def _train_phase(self, optimizer, num_epochs: int, processed_data: Dict,
                    scaler: Optional[torch.cuda.amp.GradScaler], 
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                    phase_name: str):
        """학습 페이즈 실행"""
        from loguru import logger
        
        progress_bar = create_progress_bar(num_epochs, f"{phase_name} Training")
        
        for epoch in range(num_epochs):
            system_monitor.update()
            
            # 새로운 샘플 생성 (매 에포크마다)
            samples = self.generate_training_samples()
            
            def closure():
                optimizer.zero_grad()
                
                if scaler and phase_name == "Adam":
                    with torch.cuda.amp.autocast():
                        losses = self.training_step(samples, processed_data, epoch)
                    scaler.scale(losses['total']).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses = self.training_step(samples, processed_data, epoch)
                    losses['total'].backward()
                    if phase_name == "Adam":
                        optimizer.step()
                
                return losses['total']
            
            if phase_name == "LBFGS":
                optimizer.step(closure)
                # LBFGS: gradient 유지한 채로 loss 계산
                losses = self.training_step(samples, processed_data, epoch)
            else:
                total_loss = closure()
                losses = self.training_step(samples, processed_data, epoch)
            
            # 손실 기록
            loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                        for k, v in losses.items()}
            loss_tracker.update(loss_dict, epoch)
            
            # 학습률 스케줄링
            if scheduler and phase_name == "Adam":
                scheduler.step(losses['total'])
            
            # 진행률 업데이트
            progress_bar.set_postfix({
                'Total': f"{losses['total'].item():.2e}",
                'PDE': f"{losses['pde'].item():.2e}",
                'BC': f"{losses['bc'].item():.2e}",
                'FSI': f"{losses['fsi'].item():.2e}"
            })
            progress_bar.update(1)
            
            # 체크포인트 저장 (매 100 에포크)
            if (epoch + 1) % 100 == 0:
                self.results_manager.save_model_checkpoint(
                    self.model, optimizer, epoch, losses['total'].item(),
                    metadata={'phase': phase_name, 'lr': optimizer.param_groups[0]['lr']}
                )
        
        progress_bar.close()
    
    def evaluate(self, processed_data: Dict):
        """모델 평가"""
        from loguru import logger
        
        with Timer("모델 평가"):
            self.model.eval()
            
            with torch.no_grad():
                # 평가용 그리드 생성
                eval_grid = self._create_evaluation_grid()
                
                # 유동장 예측
                flow_predictions = self.model(eval_grid)
                
                # 구조 응답 분석
                self._analyze_structural_response(processed_data)
                
                # 결과 시각화
                self._visualize_results(eval_grid, flow_predictions, processed_data)
        
        logger.info("📈 평가 완료")
    
    def _create_evaluation_grid(self) -> torch.Tensor:
        """평가용 그리드 생성"""
        domain_bounds = self.domain_params.get_nondim_bounds(self.phys_params.C_phys)
        
        # 공간 그리드
        nx, ny = 100, 80
        x = np.linspace(domain_bounds['x_min'], domain_bounds['x_max'], nx)
        y = np.linspace(domain_bounds['y_min'], domain_bounds['y_max'], ny)
        X, Y = np.meshgrid(x, y)
        
        # 시간 (중간값)
        t_eval = (self.domain_params.t_start + self.domain_params.t_end) / 2
        T = np.full_like(X, t_eval)
        
        # 텐서로 변환
        grid_points = torch.tensor(
            np.stack([T.flatten(), X.flatten(), Y.flatten()], axis=1),
            dtype=torch.float32, device=self.device
        )
        
        return grid_points
    
    def _analyze_structural_response(self, processed_data: Dict):
        """구조 응답 분석"""
        from structure_dynamics import ResponseAnalysis
        
        analyzer = ResponseAnalysis()
        damping_data = processed_data['damping_nd']
        
        # FFT 분석
        h_fft = analyzer.fft_analysis(damping_data['time'], damping_data['h'])
        theta_fft = analyzer.fft_analysis(damping_data['time'], damping_data['theta'])
        
        # 주파수 분석 결과 저장
        results_summary = {
            'h_dominant_freq': analyzer.identify_dominant_frequency(
                damping_data['time'], damping_data['h']
            ),
            'theta_dominant_freq': analyzer.identify_dominant_frequency(
                damping_data['time'], damping_data['theta']
            ),
            'h_rms': analyzer.compute_rms(damping_data['h']),
            'theta_rms': analyzer.compute_rms(damping_data['theta'])
        }
        
        self.results_manager.save_results_summary(results_summary)
        
        # 응답 분석 플롯
        analyzer.plot_response_analysis(
            damping_data['time'], damping_data['h'], damping_data['theta'],
            damping_data['Lift'], damping_data['Moment'],
            save_path=os.path.join(self.file_config.output_dir, "structural_response.png")
        )
    
    def _visualize_results(self, grid_points: torch.Tensor, 
                          predictions: torch.Tensor, processed_data: Dict):
        """결과 시각화"""
        
        # 그리드 차원 복원
        nx, ny = 100, 80
        
        # 예측값을 그리드로 재구성
        u_pred = predictions[:, 0].cpu().numpy().reshape(ny, nx)
        v_pred = predictions[:, 1].cpu().numpy().reshape(ny, nx)
        p_pred = predictions[:, 2].cpu().numpy().reshape(ny, nx)
        
        x_grid = grid_points[:, 1].cpu().numpy().reshape(ny, nx)
        y_grid = grid_points[:, 2].cpu().numpy().reshape(ny, nx)
        
        # 에어포일 윤곽선
        airfoil_x, airfoil_y = self.bc_manager.geometry.get_surface_points(100)
        
        # 유동장 시각화
        self.visualizer.plot_flow_field(
            x_grid, y_grid, u_pred, v_pred, p_pred,
            airfoil_x, airfoil_y,
            save_path=os.path.join(self.file_config.output_dir, "flow_field.png")
        )
        
        # 손실 곡선 플롯
        loss_tracker.plot_curves(
            save_path=os.path.join(self.file_config.output_dir, "loss_curves.png")
        )
        
        # 시스템 모니터링 결과
        system_monitor.plot_history(
            save_path=os.path.join(self.file_config.output_dir, "system_monitoring.png")
        )

def main():
    """메인 함수"""
    
    # 명령행 인수 파싱
    args = parse_args()
    
    # 설정 생성
    phys_params, domain_params, pinn_config, training_config, file_config = \
        create_config_from_args(args)
    
    # 유틸리티 초기화
    initialize_utils("INFO", file_config.output_dir)
    
    # 학습기 생성
    trainer = PINNTrainer(
        phys_params, domain_params, pinn_config, 
        training_config, file_config, args.coord_sys
    )
    
    try:
        # 데이터 준비
        processed_data = trainer.prepare_data()
        
        # 학습 실행
        trainer.train(processed_data)
        
        # 평가 및 결과 저장
        trainer.evaluate(processed_data)
        
        # 최종 모델 저장
        save_model(trainer.model, file_config.model_save_path, {
            'training_config': training_config,
            'phys_params': phys_params,
            'final_loss': loss_tracker.best_loss
        })
        
        from loguru import logger
        logger.info(f"🎯 최종 모델 저장: {file_config.model_save_path}")
        logger.info(f"🏆 최고 성능: {loss_tracker.best_loss:.2e} (Epoch {loss_tracker.best_epoch})")
        
    except KeyboardInterrupt:
        from loguru import logger
        logger.warning("⚠️ 사용자에 의해 학습이 중단되었습니다.")
        
        # 현재 상태 저장
        save_model(trainer.model, file_config.model_save_path.replace('.pt', '_interrupted.pt'))
        
    except Exception as e:
        from loguru import logger
        logger.error(f"❌ 학습 중 오류 발생: {str(e)}")
        raise
    
    finally:
        # 정리 작업
        loss_tracker.save_history(
            os.path.join(file_config.output_dir, "loss_history.json")
        )

if __name__ == "__main__":
    main()