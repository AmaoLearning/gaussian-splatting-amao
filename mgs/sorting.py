import torch
from typing import Optional

FIXED_ORDER_POLICIES = {
    "fixed_append": "append",
    "fixed_prepend": "prepend",
    "fixed_random": "random",
}


def resolve_fixed_order_policy(strategy: str) -> Optional[str]:
    return FIXED_ORDER_POLICIES.get(strategy)


class SplatSorter:
    def __init__(self, strategy: str = "by_volume_descending"):
        self.strategy = strategy

    def _parse_strategy(self) -> tuple[str, bool]:
        strategy = self.strategy
        if strategy.endswith("_ascending"):
            base = strategy[: -len("_ascending")]
            return base, False
        if strategy.endswith("_descending"):
            base = strategy[: -len("_descending")]
            return base, True
        if strategy in {"size", "volume", "by_volume"}:
            return "by_volume", True
        if strategy in {"by_opacity", "by_sh_energy", "by_color_variance"}:
            return strategy, True
        return strategy, True

    def _fixed_order_policy(self) -> Optional[str]:
        return resolve_fixed_order_policy(self.strategy)

    def _sh0_to_rgb(self, sh0: torch.Tensor) -> torch.Tensor:
        # sh0 is DC SH coefficient; approximate RGB in [0,1].
        C0 = 0.28209479177387814
        return sh0 * C0 + 0.5

    def _get_rgb(self, splats) -> torch.Tensor:
        if hasattr(splats, 'get_opacity'):
            # 3DGS GaussianModel format
            sh0 = splats.get_features_dc.squeeze(1)
            return self._sh0_to_rgb(sh0)
        if "colors" in splats:
            return torch.sigmoid(splats["colors"])
        if "sh0" in splats:
            sh0 = splats["sh0"].squeeze(1)
            return self._sh0_to_rgb(sh0)
        raise ValueError("Color variance sorting requires 'colors' or 'sh0'.")

    def argsort(self, splats) -> torch.Tensor:
        """
        对高斯点进行排序
        
        Args:
            splats: 可以是以下任一格式：
                1. 3DGS GaussianModel 实例（推荐）
                2. torch.nn.ParameterDict (MGS 原始格式，向后兼容)
                
        Returns:
            sort_indices: 排序后的索引
        """
        # === 自动检测输入类型 ===
        is_gaussian_model = hasattr(splats, 'get_xyz') and callable(getattr(splats, 'get_xyz'))
        
        if is_gaussian_model:
            # 3DGS GaussianModel 格式 - 直接调用其方法
            means = splats.get_xyz
            scales = splats.get_scaling
            quats = splats.get_rotation
            opacities = splats.get_opacity
            sh0 = splats.get_features_dc
            shN = splats.get_features_rest
        else:
            # ParameterDict 格式 - 保持向后兼容
            means = splats["means"]
            scales = splats["scales"]
            quats = splats["quats"]
            opacities = splats["opacities"]
            sh0 = splats["sh0"]
            shN = splats["shN"]
        fixed_policy = self._fixed_order_policy()
        if fixed_policy is not None:
            if is_gaussian_model:
                # 3DGS format doesn't have order_key, skip fixed order
                pass
            elif "order_key" not in splats:
                raise ValueError(
                    "Fixed-order sorting requires 'order_key' in splats."
                )
            else:
                order_key = splats["order_key"]
                return torch.argsort(order_key, descending=False)
        strategy, descending = self._parse_strategy()
        if strategy in {"size", "by_volume", "volume"}:
            # splats["scales"] are log-scales
            scales_exp = torch.exp(scales)
            # Calculate volume: product of x,y,z scales
            volume = scales_exp.prod(dim=1)
            # Sort descending: Largest (Coarse) -> Smallest (Fine)
            return torch.argsort(volume, descending=descending)
        if strategy == "by_opacity":
            opacities_sig = torch.sigmoid(opacities)
            return torch.argsort(opacities_sig, descending=descending)
        if strategy == "by_sh_energy":
            if shN is not None and sh0 is not None:
                sh0_squeeze = sh0.squeeze(1)
                high_energy = (shN ** 2).sum(dim=(1, 2))
                dc_energy = (sh0_squeeze ** 2).sum(dim=1)
                score = high_energy / (dc_energy + 1e-6)
            elif shN is not None:
                score = (shN ** 2).sum(dim=(1, 2))
            elif hasattr(splats, 'get_features'):
                # 3DGS GaussianModel
                colors = torch.sigmoid(splats.get_features.squeeze(1))
                score = (colors ** 2).sum(dim=1)
            else:
                raise ValueError("SH energy sorting requires SH coeffs or colors.")
            return torch.argsort(score, descending=descending)
        if strategy == "by_color_variance":
            rgb = self._get_rgb(splats if not is_gaussian_model else {"sh0": sh0})
            mean_rgb = rgb.mean(dim=1, keepdim=True)
            variance = ((rgb - mean_rgb) ** 2).mean(dim=1)
            return torch.argsort(variance, descending=descending)
        if strategy == "random":
            return torch.randperm(means.shape[0], device=means.device)
        raise NotImplementedError(f"Unknown sort strategy: {self.strategy}")
