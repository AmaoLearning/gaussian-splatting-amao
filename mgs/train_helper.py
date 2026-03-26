"""
MGS 训练集成辅助模块

提供简化的 MGS 多尺度训练功能，直接集成到 3DGS 训练循环中。
支持自主选择 MRL 或 Diffusion 风格的 subset scheduler。
"""
import torch
from typing import Optional, List, Dict, Any, Union, Tuple


class MGSTrainingHelper:
    """MGS 训练辅助类 - 简化版
    
    提供 MGS 多尺度训练所需的辅助功能，直接在 train.py 中使用。
    不创建额外的抽象层，保持代码简洁。
    
    支持两种 scheduler:
        1. MRLSubsetScheduler: MRL 风格的固定嵌套子集（默认）
        2. DiffusionSubsetScheduler: Diffusion 风格的随机子集（可选）
    """
    
    def __init__(
        self,
        use_mgs: bool = False,
        scheduler_type: str = "mrl",
        # MRL 参数
        cap_max: int = 5_000_000,
        min_splats: int = 100_000,
        nesting_base_max: Optional[int] = None,
        # Diffusion 参数（仅在 scheduler_type="diffusion" 时使用）
        diffusion_num_timesteps: int = 1000,
        diffusion_num_subsets: int = 4,
        diffusion_schedule: str = "cosine",
        diffusion_min_keep_ratio: float = 0.0,
        diffusion_max_keep_ratio: float = 0.9,
        diffusion_include_full_subset: bool = True,
        # 通用参数
        sort_strategy: str = "by_volume_descending",
        update_interval: int = 100,
    ):
        """
        Args:
            use_mgs: 是否启用 MGS 训练
            scheduler_type: scheduler 类型，"mrl" 或 "diffusion"
            
            # MRL 参数（仅在 scheduler_type="mrl" 时使用）
            cap_max: 最大高斯点数量（用于计算 MRL 嵌套大小）
            min_splats: 最小高斯点数量
            nesting_base_max: 可选，MRL 论文缩放基准（默认 None 使用一致减半）
            
            # Diffusion 参数（仅在 scheduler_type="diffusion" 时使用）
            diffusion_num_timesteps: 扩散时间步数量
            diffusion_num_subsets: 每个训练步的子集数量
            diffusion_schedule: 调度类型 ("cosine", "linear", "uniform")
            diffusion_min_keep_ratio: 最小保留比例
            diffusion_max_keep_ratio: 最大保留比例
            diffusion_include_full_subset: 是否包含完整子集
            
            # 通用参数
            sort_strategy: 排序策略
            update_interval: 排序索引更新间隔（步数）
        """
        self.use_mgs = use_mgs
        self.scheduler_type = scheduler_type
        self.sort_strategy = sort_strategy
        self.update_interval = update_interval
        
        # 存储 MRL 参数
        self.cap_max = cap_max
        self.min_splats = min_splats
        self.nesting_base_max = nesting_base_max
        
        # 存储 Diffusion 参数
        self.diffusion_num_timesteps = diffusion_num_timesteps
        self.diffusion_num_subsets = diffusion_num_subsets
        self.diffusion_schedule = diffusion_schedule
        self.diffusion_min_keep_ratio = diffusion_min_keep_ratio
        self.diffusion_max_keep_ratio = diffusion_max_keep_ratio
        self.diffusion_include_full_subset = diffusion_include_full_subset
        
        self.scheduler = None
        if use_mgs:
            self._init_scheduler()
    
    def _init_scheduler(self):
        """初始化 scheduler"""
        from mgs.subset_scheduler import MRLSubsetScheduler, DiffusionSubsetScheduler
        
        if self.scheduler_type == "mrl":
            self.scheduler = MRLSubsetScheduler(
                cap_max=self.cap_max,
                min_splats=self.min_splats,
                nesting_base_max=self.nesting_base_max,
            )
        elif self.scheduler_type == "diffusion":
            self.scheduler = DiffusionSubsetScheduler(
                num_timesteps=self.diffusion_num_timesteps,
                num_subsets=self.diffusion_num_subsets,
                schedule=self.diffusion_schedule,
                min_keep_ratio=self.diffusion_min_keep_ratio,
                max_keep_ratio=self.diffusion_max_keep_ratio,
                include_full_subset=self.diffusion_include_full_subset,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}. Must be 'mrl' or 'diffusion'.")
    
    def should_update_sort_indices(self, iteration: int) -> bool:
        """判断是否需要更新排序索引"""
        if not self.use_mgs:
            return False
        return (iteration % self.update_interval == 0)
    
    def update_sort_indices(self, gaussians, iteration: int):
        """更新排序索引"""
        if not self.use_mgs:
            return
        
        self._sort_indices = gaussians.get_mgs_sort_indices(self.sort_strategy)
        self._last_update_iteration = iteration
    
    def get_cached_sort_indices(self) -> Optional[torch.Tensor]:
        """
        获取缓存的排序索引
        
        Returns:
            排序索引张量
        """
        if not hasattr(self, '_sort_indices') or self._sort_indices is None:
            raise RuntimeError("Sort indices not cached. Call update_sort_indices() first.")
        return self._sort_indices
    
    def get_subsets(
        self,
        gaussians,
        iteration: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        获取当前迭代的子集列表
        
        Args:
            gaussians: GaussianModel 实例
            iteration: 当前迭代次数
            
        Returns:
            子集列表，每个子集包含 indices, keep_ratio, weight 等
        """
        if not self.use_mgs:
            return None
        
        # 获取排序索引
        sort_indices = self.get_cached_sort_indices()
        num_splats = len(sort_indices)
        
        # 采样子集
        subsets = self.scheduler.sample_subsets(
            num_splats=num_splats,
            sort_indices=sort_indices,
            device="cuda",
        )
        
        return subsets
    
    def get_subset_for_ratio(
        self,
        gaussians,
        ratio: float,
    ) -> Dict[str, Any]:
        """
        获取指定比例的子集参数（用于渲染评估）
        
        Args:
            gaussians: GaussianModel 实例
            ratio: 保留比例 (0.0 ~ 1.0)
            
        Returns:
            子集参数字典，包含 indices 和 subset_params
        """
        # 获取排序索引
        sort_indices = self.get_cached_sort_indices()
        num_splats = len(sort_indices)
        
        # 计算保留数量
        keep_count = max(1, int(num_splats * ratio))
        
        # 获取前 keep_count 个索引
        subset_indices = sort_indices[:keep_count]
        
        # 获取子集参数
        subset_params = gaussians.get_mgs_subset_params(subset_indices)
        
        return {
            "indices": subset_indices,
            "subset_params": subset_params,
            "keep_count": keep_count,
            "total_count": num_splats,
            "ratio": ratio,
        }
    
    def compute_multi_subset_loss(
        self,
        viewpoint_cam,
        gaussians,
        pipe,
        background: torch.Tensor,
        gt_image: torch.Tensor,
        subsets: List[Dict[str, Any]],
        use_ssim: bool = True,
        lambda_dssim: float = 0.2,
        use_trained_exp: bool = False,
        separate_sh: bool = False,
        use_fused_ssim: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        计算多子集损失
        
        Args:
            viewpoint_cam: 视点相机
            gaussians: GaussianModel 实例
            pipe: 渲染管线配置
            background: 背景颜色
            gt_image: GT 图像
            subsets: 子集列表
            use_ssim: 是否使用 SSIM
            lambda_dssim: SSIM 权重
            use_trained_exp: 是否使用训练的曝光
            separate_sh: 是否分离处理 DC 和高频 SH
            use_fused_ssim: 是否使用 Fused SSIM
            
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典，包含 total_loss, l1_loss, ssim_loss 等
            full_render_pkg: 最大子集（通常是完整点云）的渲染结果包
        """
        from gaussian_renderer import render
        from utils.loss_utils import l1_loss, ssim
        
        total_loss = 0.0
        Ll1_accum = 0.0
        ssim_accum = 0.0
        loss_dict = {}
        full_render_pkg = None
        
        for idx, subset in enumerate(subsets):
            subset_indices = subset["indices"]
            weight = subset.get("weight", 1.0)
            
            # 获取子集参数
            subset_params = gaussians.get_mgs_subset_params(subset_indices)
            
            # 渲染子集
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                overrides=subset_params,
                use_trained_exp=use_trained_exp,
                separate_sh=separate_sh,
            )
            
            rendered_image = render_pkg["render"]
            
            # 保存最大子集（最后一个）的渲染结果
            if idx == len(subsets) - 1:
                full_render_pkg = render_pkg
            
            # 应用 alpha mask（如果有）
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                rendered_image *= alpha_mask
            
            # 计算损失
            Ll1 = l1_loss(rendered_image, gt_image)
            
            if use_ssim:
                if use_fused_ssim:
                    # 使用 Fused SSIM（需要额外导入）
                    try:
                        from fused_ssim import fused_ssim
                        ssim_value = fused_ssim(rendered_image.unsqueeze(0), gt_image.unsqueeze(0))
                    except:
                        ssim_value = ssim(rendered_image, gt_image)
                else:
                    ssim_value = ssim(rendered_image, gt_image)
                
                loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value)
            else:
                loss = Ll1
                ssim_value = torch.tensor(0.0)
            
            # 累加损失
            total_loss += loss * weight
            Ll1_accum += Ll1 * weight
            ssim_accum += ssim_value * weight
            
            # 记录每个子集的损失
            loss_dict[f"l1_subset_{idx}"] = Ll1.item()
            if use_ssim:
                loss_dict[f"ssim_subset_{idx}"] = ssim_value.item()
        
        # 平均化（如果多个子集）
        if len(subsets) > 1:
            total_loss = total_loss / len(subsets)
            Ll1_accum = Ll1_accum / len(subsets)
            ssim_accum = ssim_accum / len(subsets)
        
        loss_dict["total_loss"] = total_loss.item()
        loss_dict["l1_loss"] = Ll1_accum.item()
        if use_ssim:
            loss_dict["ssim_loss"] = ssim_accum.item()
        
        return total_loss, loss_dict, full_render_pkg
