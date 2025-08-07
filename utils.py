"""
NACA 0012 Flutter PINN - 유틸리티 모듈
Physics-Informed Neural Networks for 2-DOF Flutter Analysis

유틸리티 함수들:
- 로깅 및 모니터링
- 시각화 도구
- 성능 측정
- 파일 관리
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import seaborn as sns
import time
import psutil
import os
import json
from typing import Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
from loguru import logger
import imageio
from tqdm import tqdm

# 로깅 설정
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """로깅 시스템 설정"""
    logger.remove()  # 기본 핸들러 제거
    
    # 콘솔 출력
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # 파일 출력
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="10 MB"
        )
    
    logger.info("로깅 시스템 초기화 완료")

class Timer:
    """시간 측정 유틸리티"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"{self.name} 시작...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logger.info(f"{self.name} 완료 (소요시간: {elapsed:.2f}초)")
        
    def elapsed(self) -> float:
        """경과 시간 반환"""
        if self.start_time is None:
            return 0.0
        current_time = time.time() if self.end_time is None else self.end_time
        return current_time - self.start_time

class SystemMonitor:
    """시스템 리소스 모니터링"""
    
    def __init__(self):
        self.cpu_history = []
        self.memory_history = []
        self.gpu_history = []
        self.timestamps = []
        
    def update(self):
        """시스템 상태 업데이트"""
        current_time = time.time()
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        self.timestamps.append(current_time)
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_info.percent)
        
        # GPU 모니터링 (NVIDIA-SMI 사용)
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                self.gpu_history.append(gpu_memory)
            else:
                self.gpu_history.append(0.0)
        except:
            self.gpu_history.append(0.0)
    
    def get_current_status(self) -> Dict[str, float]:
        """현재 시스템 상태 반환"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'available_memory_gb': psutil.virtual_memory().available / 1024**3
        }
    
    def plot_history(self, save_path: Optional[str] = None):
        """시스템 사용 이력 플롯"""
        if not self.timestamps:
            logger.warning("모니터링 데이터가 없습니다.")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 시간 정규화
        start_time = self.timestamps[0]
        time_elapsed = [(t - start_time) / 60 for t in self.timestamps]  # 분 단위
        
        # CPU 사용률
        axes[0].plot(time_elapsed, self.cpu_history, 'b-', linewidth=2)
        axes[0].set_ylabel('CPU 사용률 (%)')
        axes[0].set_title('시스템 리소스 모니터링')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)
        
        # 메모리 사용률
        axes[1].plot(time_elapsed, self.memory_history, 'g-', linewidth=2)
        axes[1].set_ylabel('메모리 사용률 (%)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
        
        # GPU 메모리 사용률
        axes[2].plot(time_elapsed, self.gpu_history, 'r-', linewidth=2)
        axes[2].set_ylabel('GPU 메모리 (%)')
        axes[2].set_xlabel('시간 (분)')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 100)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

class LossTracker:
    """손실 함수 추적"""
    
    def __init__(self):
        self.history = {
            'total': [],
            'data': [],
            'pde': [],
            'bc': [],
            'fsi': [],
            'epochs': []
        }
        self.best_loss = float('inf')
        self.best_epoch = 0
        
    def update(self, losses: Dict[str, float], epoch: int):
        """손실 업데이트"""
        self.history['epochs'].append(epoch)
        
        for key, value in losses.items():
            if key in self.history:
                loss_val = value.item() if isinstance(value, torch.Tensor) else value
                self.history[key].append(loss_val)
                
                # 최고 성능 추적
                if key == 'total' and loss_val < self.best_loss:
                    self.best_loss = loss_val
                    self.best_epoch = epoch
    
    def plot_curves(self, save_path: Optional[str] = None, log_scale: bool = True):
        """손실 곡선 플롯"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = {
            'total': 'black',
            'data': 'blue', 
            'pde': 'red',
            'bc': 'green',
            'fsi': 'orange'
        }
        
        for i, (loss_type, history) in enumerate(self.history.items()):
            if loss_type == 'epochs' or not history:
                continue
                
            ax = axes[i]
            epochs = self.history['epochs']
            
            if log_scale:
                ax.semilogy(epochs, history, color=colors.get(loss_type, 'gray'), 
                           linewidth=2, label=loss_type.upper())
            else:
                ax.plot(epochs, history, color=colors.get(loss_type, 'gray'),
                       linewidth=2, label=loss_type.upper())
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'{loss_type.upper()} Loss')
            ax.set_title(f'{loss_type.upper()} Loss Curve')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 마지막 축 제거
        if len(self.history) - 1 < len(axes):
            for j in range(len(self.history) - 1, len(axes)):
                fig.delaxes(axes[j])
        
        plt.suptitle(f'Training Loss Curves (Best: {self.best_loss:.2e} @ Epoch {self.best_epoch})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_history(self, filepath: str):
        """손실 이력 저장"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"손실 이력 저장: {filepath}")
    
    def load_history(self, filepath: str):
        """손실 이력 로드"""
        with open(filepath, 'r') as f:
            self.history = json.load(f)
        logger.info(f"손실 이력 로드: {filepath}")

class FlowFieldVisualizer:
    """유동장 시각화"""
    
    def __init__(self, domain_bounds: Dict[str, float]):
        self.domain_bounds = domain_bounds
        
    def plot_flow_field(self, x: np.ndarray, y: np.ndarray, 
                       u: np.ndarray, v: np.ndarray, p: np.ndarray,
                       airfoil_x: Optional[np.ndarray] = None,
                       airfoil_y: Optional[np.ndarray] = None,
                       save_path: Optional[str] = None):
        """유동장 시각화"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 압력 필드
        im1 = axes[0, 0].contourf(x, y, p, levels=50, cmap='RdBu_r')
        axes[0, 0].set_title('압력장 (p)')
        axes[0, 0].set_xlabel('x/C')
        axes[0, 0].set_ylabel('y/C')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 속도 크기
        velocity_magnitude = np.sqrt(u**2 + v**2)
        im2 = axes[0, 1].contourf(x, y, velocity_magnitude, levels=50, cmap='viridis')
        axes[0, 1].set_title('속도 크기 (|V|)')
        axes[0, 1].set_xlabel('x/C')
        axes[0, 1].set_ylabel('y/C')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 속도 벡터
        stride = max(1, len(x) // 20)  # 벡터 밀도 조정
        axes[1, 0].quiver(x[::stride, ::stride], y[::stride, ::stride], 
                         u[::stride, ::stride], v[::stride, ::stride],
                         velocity_magnitude[::stride, ::stride], 
                         cmap='plasma', alpha=0.8)
        axes[1, 0].set_title('속도 벡터')
        axes[1, 0].set_xlabel('x/C')
        axes[1, 0].set_ylabel('y/C')
        
        # 와도
        dy, dx = np.gradient(y), np.gradient(x)
        du_dy = np.gradient(u, axis=0) / dy[0]
        dv_dx = np.gradient(v, axis=1) / dx[0]
        vorticity = dv_dx - du_dy
        
        im4 = axes[1, 1].contourf(x, y, vorticity, levels=50, cmap='RdBu')
        axes[1, 1].set_title('와도 (ω)')
        axes[1, 1].set_xlabel('x/C')
        axes[1, 1].set_ylabel('y/C')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # 에어포일 윤곽선 추가
        if airfoil_x is not None and airfoil_y is not None:
            for ax in axes.flat:
                ax.plot(airfoil_x, airfoil_y, 'k-', linewidth=2)
                ax.fill(airfoil_x, airfoil_y, color='white', alpha=0.8)
        
        # 축 설정
        for ax in axes.flat:
            ax.set_xlim(self.domain_bounds['x_min'], self.domain_bounds['x_max'])
            ax.set_ylim(self.domain_bounds['y_min'], self.domain_bounds['y_max'])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_flow_animation(self, time_series: List[Dict], 
                            airfoil_motion: Optional[Dict] = None,
                            save_path: str = "flow_animation.mp4",
                            fps: int = 10):
        """유동장 애니메이션 생성"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        def animate(frame):
            for ax in axes.flat:
                ax.clear()
            
            # 현재 프레임 데이터
            data = time_series[frame]
            x, y = data['x'], data['y']
            u, v, p = data['u'], data['v'], data['p']
            
            # 압력장
            im1 = axes[0, 0].contourf(x, y, p, levels=20, cmap='RdBu_r')
            axes[0, 0].set_title(f'압력장 (t={data.get("time", frame):.2f})')
            
            # 속도 크기
            vel_mag = np.sqrt(u**2 + v**2)
            im2 = axes[0, 1].contourf(x, y, vel_mag, levels=20, cmap='viridis')
            axes[0, 1].set_title('속도 크기')
            
            # 속도 벡터
            stride = max(1, len(x) // 15)
            axes[1, 0].quiver(x[::stride, ::stride], y[::stride, ::stride],
                             u[::stride, ::stride], v[::stride, ::stride])
            axes[1, 0].set_title('속도 벡터')
            
            # 와도
            vorticity = data.get('vorticity', np.zeros_like(p))
            im4 = axes[1, 1].contourf(x, y, vorticity, levels=20, cmap='RdBu')
            axes[1, 1].set_title('와도')
            
            # 에어포일 추가
            if airfoil_motion and frame < len(airfoil_motion['x']):
                airfoil_x = airfoil_motion['x'][frame]
                airfoil_y = airfoil_motion['y'][frame]
                for ax in axes.flat:
                    ax.plot(airfoil_x, airfoil_y, 'k-', linewidth=2)
                    ax.fill(airfoil_x, airfoil_y, color='white')
            
            # 축 설정
            for ax in axes.flat:
                ax.set_xlim(self.domain_bounds['x_min'], self.domain_bounds['x_max'])
                ax.set_ylim(self.domain_bounds['y_min'], self.domain_bounds['y_max'])
                ax.set_aspect('equal')
        
        # 애니메이션 생성
        anim = animation.FuncAnimation(
            fig, animate, frames=len(time_series), 
            interval=1000//fps, blit=False
        )
        
        # 저장
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(save_path, writer=writer)
        plt.close()
        
        logger.info(f"애니메이션 저장: {save_path}")
        return anim

class ModelAnalyzer:
    """모델 분석 도구"""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        
    def count_parameters(self) -> Dict[str, int]:
        """파라미터 수 계산"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    def analyze_gradients(self, loss: torch.Tensor) -> Dict[str, float]:
        """기울기 분석"""
        loss.backward()
        
        total_norm = 0.0
        param_count = 0
        max_grad = 0.0
        min_grad = float('inf')
        
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                max_grad = max(max_grad, p.grad.abs().max().item())
                min_grad = min(min_grad, p.grad.abs().min().item())
        
        total_norm = total_norm ** (1. / 2)
        
        return {
            'total_norm': total_norm,
            'average_norm': total_norm / max(param_count, 1),
            'max_gradient': max_grad,
            'min_gradient': min_grad,
            'param_with_gradients': param_count
        }
    
    def memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 분석"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            cached = torch.cuda.memory_reserved() / 1024**2     # MB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            return {
                'allocated_mb': allocated,
                'cached_mb': cached,
                'max_allocated_mb': max_allocated,
                'utilization_percent': (allocated / max_allocated) * 100 if max_allocated > 0 else 0
            }
        else:
            return {'message': 'CUDA not available'}

class ResultsManager:
    """결과 관리"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_model_checkpoint(self, model: torch.nn.Module, 
                            optimizer: torch.optim.Optimizer,
                            epoch: int, loss: float,
                            metadata: Optional[Dict] = None):
        """모델 체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        filepath = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, filepath)
        logger.info(f"체크포인트 저장: {filepath}")
        
        # 최고 성능 모델도 별도 저장
        best_filepath = os.path.join(self.output_dir, 'best_model.pt')
        if not os.path.exists(best_filepath) or loss < self._get_best_loss():
            torch.save(checkpoint, best_filepath)
            logger.info(f"최고 성능 모델 업데이트: {best_filepath}")
    
    def _get_best_loss(self) -> float:
        """저장된 최고 성능 손실값 반환"""
        best_filepath = os.path.join(self.output_dir, 'best_model.pt')
        if os.path.exists(best_filepath):
            checkpoint = torch.load(best_filepath, map_location='cpu')
            return checkpoint.get('loss', float('inf'))
        return float('inf')
    
    def save_results_summary(self, results: Dict):
        """결과 요약 저장"""
        summary_file = os.path.join(self.output_dir, 'results_summary.json')
        
        # 기존 결과와 병합
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                existing_results = json.load(f)
        else:
            existing_results = {}
        
        # 타임스탬프 추가
        results['timestamp'] = datetime.now().isoformat()
        existing_results[results['timestamp']] = results
        
        with open(summary_file, 'w') as f:
            json.dump(existing_results, f, indent=2, default=str)
        
        logger.info(f"결과 요약 저장: {summary_file}")

# 편의 함수들
def ensure_dir(directory: str):
    """디렉터리 존재 확인 및 생성"""
    os.makedirs(directory, exist_ok=True)

def get_device() -> torch.device:
    """최적 디바이스 선택"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"CUDA 디바이스 사용: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("CPU 디바이스 사용")
    return device

def set_random_seeds(seed: int = 42):
    """재현성을 위한 시드 설정"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"랜덤 시드 설정: {seed}")

def print_model_summary(model: torch.nn.Module):
    """모델 요약 출력"""
    analyzer = ModelAnalyzer(model)
    params = analyzer.count_parameters()
    
    logger.info("=" * 50)
    logger.info("모델 요약")
    logger.info("=" * 50)
    logger.info(f"총 파라미터 수: {params['total']:,}")
    logger.info(f"학습 가능 파라미터: {params['trainable']:,}")
    logger.info(f"고정 파라미터: {params['non_trainable']:,}")
    
    if hasattr(model, 'get_model_summary'):
        summary = model.get_model_summary()
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
    
    logger.info("=" * 50)

def create_progress_bar(total: int, desc: str = "Processing") -> tqdm:
    """진행률 표시줄 생성"""
    return tqdm(
        total=total,
        desc=desc,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        colour='green'
    )

# 전역 유틸리티 인스턴스
system_monitor = SystemMonitor()
loss_tracker = LossTracker()

def initialize_utils(log_level: str = "INFO", output_dir: str = "results"):
    """유틸리티 초기화"""
    setup_logging(log_level, os.path.join(output_dir, "training.log"))
    ensure_dir(output_dir)
    set_random_seeds(42)
    
    logger.info("🚀 NACA 0012 Flutter PINN 프로젝트 시작")
    logger.info(f"출력 디렉터리: {output_dir}")
    
    # 시스템 상태 출력
    status = system_monitor.get_current_status()
    logger.info(f"시스템 상태 - CPU: {status['cpu_percent']:.1f}%, "
                f"Memory: {status['memory_percent']:.1f}%, "
                f"Available Memory: {status['available_memory_gb']:.1f}GB")

if __name__ == "__main__":
    # 유틸리티 테스트
    initialize_utils()
    
    # 시스템 모니터링 테스트
    logger.info("시스템 모니터링 테스트 시작...")
    for i in range(10):
        system_monitor.update()
        time.sleep(0.1)
    
    system_monitor.plot_history("results/system_monitor_test.png")
    
    # 손실 추적 테스트
    logger.info("손실 추적 테스트 시작...")
    for epoch in range(100):
        fake_losses = {
            'total': np.exp(-epoch/50) + 0.1 * np.random.random(),
            'data': np.exp(-epoch/30) + 0.05 * np.random.random(),
            'pde': np.exp(-epoch/40) + 0.08 * np.random.random(),
            'bc': np.exp(-epoch/60) + 0.03 * np.random.random(),
            'fsi': np.exp(-epoch/45) + 0.06 * np.random.random()
        }
        loss_tracker.update(fake_losses, epoch)
    
    loss_tracker.plot_curves("results/loss_curves_test.png")
    loss_tracker.save_history("results/loss_history_test.json")
    
    logger.info("유틸리티 테스트 완료 ✅")