#!/usr/bin/env python3
"""
NACA 0012 Flutter PINN - LBFGS 이어서 학습 스크립트
Adam 학습 완료된 모델에서 LBFGS만 추가 학습
"""

import torch
import torch.optim as optim
import os
from pathlib import Path
from tqdm import tqdm

# 프로젝트 모듈 import
from config import *
from pinn_model import create_pinn_model
from boundary_conditions import create_boundary_manager
from structure_dynamics import create_structure_system
from data_io import DataProcessor
from utils import *
from main import PINNTrainer

class LBFGSContinueTrainer(PINNTrainer):
    """LBFGS 이어서 학습을 위한 트레이너"""
    
    def __init__(self, phys_params, domain_params, pinn_config, 
                 training_config, file_config, coord_sys: str = "lab"):
        super().__init__(phys_params, domain_params, pinn_config, 
                        training_config, file_config, coord_sys)
        
    def load_best_model(self):
        """최고 성능 모델 로드"""
        results_dir = Path(r"C:\Users\Master\Desktop\NACA0012_Flutter_PINN\results")
        
        # 가능한 모델 파일들 (우선순위 순)
        model_candidates = [
            results_dir / "best_model.pt",
            results_dir / "checkpoint_epoch_4999.pt",
            results_dir / "checkpoint_epoch_4899.pt",
        ]
        
        model_path = None
        for candidate in model_candidates:
            if candidate.exists():
                model_path = candidate
                break
        
        if model_path is None:
            raise FileNotFoundError("❌ 로드할 수 있는 모델 파일을 찾을 수 없습니다!")
        
        print(f"📁 모델 로드: {model_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('loss', float('inf'))
        
        print(f"✅ 모델 로드 완료")
        print(f"   시작 에포크: {start_epoch}")
        print(f"   최고 손실: {best_loss:.6f}")
        
        return start_epoch, best_loss
    
    def lbfgs_only_training(self, processed_data: Dict):
        """LBFGS만 학습"""
        from loguru import logger
        
        print("🔧 LBFGS 이어서 학습 시작")
        print("="*50)
        
        # 모델 로드
        start_epoch, best_loss = self.load_best_model()
        
        # LBFGS 최적화기 생성
        logger.info(f"LBFGS 최적화 시작 ({self.training_config.lbfgs_epochs} epochs)")
        
        try:
            optimizer_lbfgs = optim.LBFGS(
                self.model.parameters(),
                lr=0.1,
                max_iter=20,
                max_eval=None,
                tolerance_grad=1e-7,
                tolerance_change=1e-9
            )
            
            # LBFGS 학습 실행
            self._train_lbfgs_phase(optimizer_lbfgs, self.training_config.lbfgs_epochs, 
                                   processed_data, start_epoch)
            
            logger.info("🎉 LBFGS 학습 완료!")
            
        except RuntimeError as e:
            if "does not require grad" in str(e):
                logger.warning("⚠️ LBFGS gradient 오류 발생")
                logger.warning(f"오류 내용: {e}")
                logger.info("💡 Adam 모델이 이미 좋은 성능이므로 그대로 사용하세요")
            else:
                raise e
    
    def _train_lbfgs_phase(self, optimizer, num_epochs: int, processed_data: Dict, 
                          start_epoch: int):
        """LBFGS 학습 페이즈"""
        from loguru import logger
        
        # 손실 추적기
        loss_tracker = LossTracker()
        
        # 진행률 표시
        progress_bar = tqdm(range(num_epochs), desc="LBFGS 학습")
        
        best_loss = float('inf')
        
        for epoch in progress_bar:
            self.model.train()
            
            # 샘플 생성
            samples = self.generate_training_samples()
            
            # 모든 샘플에 대해 requires_grad 재설정
            for key, value in samples.items():
                if isinstance(value, torch.Tensor):
                    value.requires_grad_(True)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, torch.Tensor):
                            subvalue.requires_grad_(True)
            
            def closure():
                optimizer.zero_grad()
                losses = self.training_step(samples, processed_data, start_epoch + epoch)
                losses['total'].backward()
                return losses['total']
            
            # LBFGS 스텝
            optimizer.step(closure)
            
            # 손실 계산 (기록용)
            with torch.no_grad():
                losses = self.training_step(samples, processed_data, start_epoch + epoch)
            
            # 손실 기록
            loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                        for k, v in losses.items()}
            loss_tracker.update(loss_dict, start_epoch + epoch)
            
            # 진행률 업데이트
            progress_bar.set_postfix({
                'Total': f"{losses['total'].item():.2e}",
                'PDE': f"{losses['pde'].item():.2e}",
                'BC': f"{losses['bc'].item():.2e}",
                'FSI': f"{losses['fsi'].item():.2e}" if 'fsi' in losses else "0.0e+00"
            })
            
            # 최고 모델 저장
            current_loss = losses['total'].item()
            if current_loss < best_loss:
                best_loss = current_loss
                self.save_checkpoint(start_epoch + epoch, current_loss, "lbfgs_best_model.pt")
            
            # 주기적 체크포인트
            if (epoch + 1) % 100 == 0:
                checkpoint_name = f"lbfgs_checkpoint_epoch_{start_epoch + epoch}.pt"
                self.save_checkpoint(start_epoch + epoch, current_loss, checkpoint_name)
        
        # 손실 곡선 저장
        loss_tracker.save_history(os.path.join(self.file_config.output_dir, "lbfgs_loss_history.json"))
        
        print(f"\n✅ LBFGS 학습 완료!")
        print(f"   최종 손실: {best_loss:.6f}")
        print(f"   최고 모델: lbfgs_best_model.pt")
    
    def save_checkpoint(self, epoch: int, loss: float, filename: str):
        """체크포인트 저장"""
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'model_config': self.pinn_config
        }
        
        save_path = os.path.join(self.file_config.output_dir, filename)
        torch.save(save_dict, save_path)

def main():
    """메인 함수"""
    print("🚀 LBFGS 이어서 학습 시작")
    print("="*50)
    
    # 설정 로드
    phys_params = PhysicalParameters()
    domain_params = DomainParameters()
    pinn_config = PINNConfig()
    training_config = TrainingConfig()
    file_config = FileConfig()
    
    # 유틸리티 초기화
    initialize_utils("INFO", file_config.output_dir)
    
    # 트레이너 생성
    trainer = LBFGSContinueTrainer(
        phys_params, domain_params, pinn_config, 
        training_config, file_config, "lab"
    )
    
    try:
        # 데이터 준비
        processed_data = trainer.prepare_data()
        
        # LBFGS만 학습
        trainer.lbfgs_only_training(processed_data)
        
        print("\n🎉 LBFGS 이어서 학습 완료!")
        print("📁 결과물:")
        print("   - lbfgs_best_model.pt: 최고 성능 모델")
        print("   - lbfgs_loss_history.json: 손실 이력")
        print("   - lbfgs_checkpoint_epoch_*.pt: 체크포인트들")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()