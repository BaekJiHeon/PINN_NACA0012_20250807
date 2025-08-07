"""
NACA 0012 Flutter PINN - 데이터 입출력 및 비차원화 모듈
Physics-Informed Neural Networks for 2-DOF Flutter Analysis
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional, Union
from scipy.interpolate import interp1d, griddata
from config import PhysicalParameters, DomainParameters, FileConfig
import matplotlib.pyplot as plt

class DataProcessor:
    """CFD 데이터 처리 및 비차원화 클래스"""
    
    def __init__(self, phys_params: PhysicalParameters, 
                 domain_params: DomainParameters,
                 file_config: FileConfig):
        self.phys_params = phys_params
        self.domain_params = domain_params
        self.file_config = file_config
        
        # 비차원화 스케일
        self.C_phys = phys_params.C_phys
        self.U_inf = phys_params.U_inf
        self.rho = phys_params.rho
        
        # 로드된 데이터 저장
        self.cfd_data = None
        self.mesh_data = None
        self.damping_data = None
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """모든 CSV 데이터 로드"""
        print("📂 CSV 데이터 로딩 중...")
        
        # CFD 데이터 로드 (실제로는 mesh center postion - 바이너리 파일로 추정)
        try:
            # 텍스트 파일로 시도
            self.cfd_data = pd.read_csv(self.file_config.cfd_csv_path, sep=None, engine='python')
            # 컬럼명 표준화
            self.cfd_data = self._standardize_mesh_columns(self.cfd_data)
            print(f"✅ CFD 데이터 로드: {len(self.cfd_data)} 점")
        except:
            print(f"⚠️ CFD 데이터 파일 로드 실패: {self.file_config.cfd_csv_path}")
            print("더미 데이터로 대체합니다.")
            self.cfd_data = self._generate_dummy_cfd_data()
            
        # 메시 데이터 로드 (Node postion)
        try:
            # 공백으로 구분된 파일 로드
            self.mesh_data = pd.read_csv(self.file_config.mesh_csv_path, sep=None, engine='python')
            # 컬럼명 표준화
            self.mesh_data = self._standardize_mesh_columns(self.mesh_data)
            print(f"✅ 메시 데이터 로드: {len(self.mesh_data)} 점")
        except Exception as e:
            print(f"⚠️ 메시 데이터 파일 로드 실패: {self.file_config.mesh_csv_path}")
            print(f"오류: {e}")
            self.mesh_data = self._generate_dummy_mesh_data()
            
        # 댐핑 데이터 로드 (Damping_data.csv)
        try:
            # 콤마 구분자로 로드
            self.damping_data = pd.read_csv(self.file_config.damping_csv_path)
            # 컬럼명 표준화
            self.damping_data = self._standardize_damping_columns(self.damping_data)
            print(f"✅ 댐핑 데이터 로드: {len(self.damping_data)} 시간 스텝")
        except Exception as e:
            print(f"⚠️ 댐핑 데이터 파일 로드 실패: {self.file_config.damping_csv_path}")
            print(f"오류: {e}")
            self.damping_data = self._generate_dummy_damping_data()
            
        return {
            'cfd': self.cfd_data,
            'mesh': self.mesh_data,
            'damping': self.damping_data
        }
    
    def _generate_dummy_cfd_data(self) -> pd.DataFrame:
        """더미 CFD 데이터 생성 (테스트용)"""
        print("🔧 더미 CFD 데이터 생성 중...")
        
        # 물리적 좌표에서 그리드 생성
        x_phys = np.linspace(self.domain_params.x_min_phys, 
                            self.domain_params.x_max_phys, 100)
        y_phys = np.linspace(self.domain_params.y_min_phys,
                            self.domain_params.y_max_phys, 80)
        X_phys, Y_phys = np.meshgrid(x_phys, y_phys)
        
        # 단순한 유동장 생성 (potential flow + perturbation)
        U = np.ones_like(X_phys)  # 균등류
        V = np.zeros_like(Y_phys)
        P = np.ones_like(X_phys) - X_phys**2  # 단순한 압력 분포
        
        # 데이터프레임 생성
        dummy_data = pd.DataFrame({
            'cell_id': range(len(X_phys.flatten())),
            'x': X_phys.flatten(),
            'y': Y_phys.flatten(),
            'p': P.flatten(),
            'u': U.flatten(),
            'v': V.flatten()
        })
        
        return dummy_data
    
    def _generate_dummy_mesh_data(self) -> pd.DataFrame:
        """더미 메시 데이터 생성"""
        print("🔧 더미 메시 데이터 생성 중...")
        
        # CFD 데이터와 유사하게 생성
        if self.cfd_data is not None:
            mesh_data = self.cfd_data.copy()
            mesh_data['node_id'] = mesh_data['cell_id']
            return mesh_data
        else:
            return self._generate_dummy_cfd_data()
    
    def _generate_dummy_damping_data(self) -> pd.DataFrame:
        """더미 댐핑 데이터 생성 (2-DOF 플러터 시뮬레이션)"""
        print("🔧 더미 댐핑 데이터 생성 중...")
        
        # 시간 배열
        t = np.linspace(self.domain_params.t_start, 
                       self.domain_params.t_end, 1000)
        
        # 플러터 주파수 및 댐핑
        omega_h = np.sqrt(self.phys_params.k_h / self.phys_params.m)
        omega_alpha = np.sqrt(self.phys_params.k_alpha / self.phys_params.I_alpha)
        zeta = 0.02  # 낮은 댐핑비 (플러터 조건)
        
        # 2-DOF 플러터 응답 시뮬레이션
        h = 0.01 * np.exp(zeta * omega_h * t) * np.sin(omega_h * t)
        theta = 0.05 * np.exp(zeta * omega_alpha * t) * np.sin(omega_alpha * t + np.pi/4)
        
        h_vel = np.gradient(h, t)
        theta_vel = np.gradient(theta, t)
        
        # 단순한 양력/모멘트 (실제로는 CFD에서 계산됨)
        Lift = -self.phys_params.k_h * h - self.phys_params.c_h * h_vel
        Moment = -self.phys_params.k_alpha * theta - self.phys_params.c_alpha * theta_vel
        
        dummy_damping = pd.DataFrame({
            'time': t,
            'h': h,
            'theta': theta,
            'h_vel': h_vel,
            'theta_vel': theta_vel,
            'Lift': Lift,
            'Moment': Moment
        })
        
        return dummy_damping
    
    def _standardize_mesh_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """메시 데이터 컬럼명 표준화"""
        # 실제 컬럼: cellnumber, x-coordinate, y-coordinate, pressure, x-velocity, y-velocity
        # 표준 컬럼: cell_id, x, y, p, u, v
        
        column_mapping = {
            'cellnumber': 'cell_id',
            'x-coordinate': 'x',
            'y-coordinate': 'y', 
            'pressure': 'p',
            'x-velocity': 'u',
            'y-velocity': 'v'
        }
        
        # 공백 제거 후 매핑
        df.columns = df.columns.str.strip()
        df = df.rename(columns=column_mapping)
        
        print(f"   📋 메시 데이터 컬럼: {list(df.columns)}")
        return df
    
    def _standardize_damping_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """댐핑 데이터 컬럼명 표준화"""
        # 실제 컬럼: step,time,Lift,Moment,heave,theta,heave_vel,theta_vel
        # 표준 컬럼: time, h, theta, h_vel, theta_vel, Lift, Moment
        
        column_mapping = {
            'heave': 'h',
            'heave_vel': 'h_vel'
            # step, time, Lift, Moment, theta, theta_vel은 그대로 유지
        }
        
        df = df.rename(columns=column_mapping)
        print(f"   📋 댐핑 데이터 컬럼: {list(df.columns)}")
        return df
    
    def nondim(self, data: Union[Dict, pd.DataFrame], data_type: str = 'flow') -> Union[Dict, pd.DataFrame]:
        """비차원화 함수
        
        Args:
            data: 물리적 단위의 데이터
            data_type: 'flow', 'structure', 'coords' 중 하나
        """
        if data_type == 'flow':
            return self._nondim_flow(data)
        elif data_type == 'structure':
            return self._nondim_structure(data)
        elif data_type == 'coords':
            return self._nondim_coords(data)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    def _nondim_flow(self, data: Union[Dict, pd.DataFrame]) -> Union[Dict, pd.DataFrame]:
        """유동 데이터 비차원화"""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key in ['x', 'y']:
                    result[key] = (value - getattr(self.domain_params, f"{key}0_phys")) / self.C_phys
                elif key in ['u', 'v']:
                    result[key] = value / self.U_inf
                elif key == 'p':
                    result[key] = value / (self.rho * self.U_inf**2)
                else:
                    result[key] = value
            return result
        
        elif isinstance(data, pd.DataFrame):
            result = data.copy()
            
            # 좌표 비차원화
            if 'x' in result.columns:
                result['x'] = (result['x'] - self.domain_params.x0_phys) / self.C_phys
            if 'y' in result.columns:
                result['y'] = (result['y'] - self.domain_params.y0_phys) / self.C_phys
                
            # 속도 비차원화
            if 'u' in result.columns:
                result['u'] = result['u'] / self.U_inf
            if 'v' in result.columns:
                result['v'] = result['v'] / self.U_inf
                
            # 압력 비차원화
            if 'p' in result.columns:
                result['p'] = result['p'] / (self.rho * self.U_inf**2)
                
            return result
        
        else:
            raise TypeError("Data must be dict or DataFrame")
    
    def _nondim_structure(self, data: Union[Dict, pd.DataFrame]) -> Union[Dict, pd.DataFrame]:
        """구조 데이터 비차원화"""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key in ['h']:
                    result[key] = value / self.C_phys
                elif key in ['theta']:
                    result[key] = value  # 각도는 무차원
                elif key in ['h_vel']:
                    result[key] = value / (self.C_phys * np.sqrt(self.phys_params.k_h / self.phys_params.m))
                elif key in ['theta_vel']:
                    result[key] = value / np.sqrt(self.phys_params.k_alpha / self.phys_params.I_alpha)
                elif key == 'Lift':
                    result[key] = value / (0.5 * self.rho * self.U_inf**2 * self.C_phys)
                elif key == 'Moment':
                    result[key] = value / (0.5 * self.rho * self.U_inf**2 * self.C_phys**2)
                else:
                    result[key] = value
            return result
        
        elif isinstance(data, pd.DataFrame):
            result = data.copy()
            
            # 변위 비차원화
            if 'h' in result.columns:
                result['h'] = result['h'] / self.C_phys
            if 'theta' in result.columns:
                result['theta'] = result['theta']  # 무차원
                
            # 속도 비차원화
            if 'h_vel' in result.columns:
                omega_h = np.sqrt(self.phys_params.k_h / self.phys_params.m)
                result['h_vel'] = result['h_vel'] / (self.C_phys * omega_h)
            if 'theta_vel' in result.columns:
                omega_alpha = np.sqrt(self.phys_params.k_alpha / self.phys_params.I_alpha)
                result['theta_vel'] = result['theta_vel'] / omega_alpha
                
            # 힘/모멘트 비차원화
            if 'Lift' in result.columns:
                result['Lift'] = result['Lift'] / (0.5 * self.rho * self.U_inf**2 * self.C_phys)
            if 'Moment' in result.columns:
                result['Moment'] = result['Moment'] / (0.5 * self.rho * self.U_inf**2 * self.C_phys**2)
                
            return result
        
        else:
            raise TypeError("Data must be dict or DataFrame")
    
    def _nondim_coords(self, coords: np.ndarray) -> np.ndarray:
        """좌표만 비차원화"""
        coords_nd = coords.copy()
        coords_nd[:, 1] = (coords[:, 1] - self.domain_params.x0_phys) / self.C_phys  # x
        coords_nd[:, 2] = (coords[:, 2] - self.domain_params.y0_phys) / self.C_phys  # y
        return coords_nd
    
    def get_airfoil_surface_points(self, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """NACA 0012 에어포일 표면점 생성"""
        
        # NACA 0012 수식 (대칭 에어포일)
        x_c = np.linspace(0, 1, n_points//2)
        
        # 두께 분포 (NACA 0012)
        t = 0.12  # 최대 두께비
        y_t = 5 * t * (0.2969 * np.sqrt(x_c) - 0.1260 * x_c - 
                       0.3516 * x_c**2 + 0.2843 * x_c**3 - 0.1015 * x_c**4)
        
        # 상면과 하면
        x_upper = x_c
        y_upper = y_t
        x_lower = x_c[::-1]
        y_lower = -y_t[::-1]
        
        # 전체 윤곽선
        x_airfoil = np.concatenate([x_upper, x_lower])
        y_airfoil = np.concatenate([y_upper, y_lower])
        
        return x_airfoil, y_airfoil
    
    def interpolate_damping_data(self, t_query: np.ndarray) -> Dict[str, np.ndarray]:
        """시간에 따른 댐핑 데이터 보간"""
        if self.damping_data is None:
            raise ValueError("댐핑 데이터가 로드되지 않았습니다.")
        
        result = {}
        t_data = self.damping_data['time'].values
        
        for col in ['h', 'theta', 'h_vel', 'theta_vel', 'Lift', 'Moment']:
            if col in self.damping_data.columns:
                f = interp1d(t_data, self.damping_data[col].values, 
                           kind='cubic', fill_value='extrapolate')
                result[col] = f(t_query)
        
        return result
    
    def to_torch_tensors(self, data: Dict[str, np.ndarray], 
                        device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
        """NumPy 배열을 PyTorch 텐서로 변환"""
        tensors = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                tensors[key] = torch.tensor(value, dtype=torch.float32, device=device)
            else:
                tensors[key] = torch.tensor([value], dtype=torch.float32, device=device)
        return tensors
    
    def export_results(self, results: Dict[str, np.ndarray], 
                      filename: str = "results/pinn_results.csv"):
        """결과를 CSV로 내보내기"""
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(filename, index=False)
        print(f"✅ 결과 저장: {filename}")
    
    def visualize_data_distribution(self, save_path: str = "results/data_distribution.png"):
        """데이터 분포 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if self.cfd_data is not None:
            # CFD 데이터 분포
            cfd_nd = self.nondim(self.cfd_data, 'flow')
            axes[0, 0].scatter(cfd_nd['x'], cfd_nd['y'], c=cfd_nd['u'], 
                             s=1, cmap='viridis', alpha=0.6)
            axes[0, 0].set_title('CFD 데이터 분포 (u 속도)')
            axes[0, 0].set_xlabel('x/C')
            axes[0, 0].set_ylabel('y/C')
            
            axes[0, 1].scatter(cfd_nd['x'], cfd_nd['y'], c=cfd_nd['p'], 
                             s=1, cmap='RdBu_r', alpha=0.6)
            axes[0, 1].set_title('CFD 데이터 분포 (압력)')
            axes[0, 1].set_xlabel('x/C')
            axes[0, 1].set_ylabel('y/C')
        
        if self.damping_data is not None:
            # 구조 응답
            damping_nd = self.nondim(self.damping_data, 'structure')
            axes[1, 0].plot(damping_nd['time'], damping_nd['h'], 'b-', label='h/C')
            axes[1, 0].plot(damping_nd['time'], damping_nd['theta'], 'r-', label='θ')
            axes[1, 0].set_title('구조 응답')
            axes[1, 0].set_xlabel('시간')
            axes[1, 0].legend()
            
            # 힘/모멘트
            axes[1, 1].plot(damping_nd['time'], damping_nd['Lift'], 'g-', label='CL')
            axes[1, 1].plot(damping_nd['time'], damping_nd['Moment'], 'm-', label='CM')
            axes[1, 1].set_title('공력 하중')
            axes[1, 1].set_xlabel('시간')
            axes[1, 1].legend()
        
        plt.tight_layout()
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 데이터 분포 시각화 저장: {save_path}")

# 편의 함수들
def load_and_process_data(phys_params: PhysicalParameters,
                         domain_params: DomainParameters, 
                         file_config: FileConfig) -> Tuple[DataProcessor, Dict]:
    """데이터 로딩 및 전처리 수행"""
    processor = DataProcessor(phys_params, domain_params, file_config)
    all_data = processor.load_all_data()
    
    # 비차원화 수행
    processed_data = {
        'cfd_nd': processor.nondim(all_data['cfd'], 'flow'),
        'mesh_nd': processor.nondim(all_data['mesh'], 'flow'),
        'damping_nd': processor.nondim(all_data['damping'], 'structure')
    }
    
    return processor, processed_data