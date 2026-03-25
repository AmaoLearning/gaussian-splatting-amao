"""
MGS 多比例评估脚本

评估不同高斯数量比例的渲染质量（PSNR/SSIM/LPIPS）
"""
from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from argparse import ArgumentParser
import json
from tqdm import tqdm


def read_images(renders_dir, gt_dir):
    """读取渲染图像和 GT 图像"""
    renders = []
    gts = []
    image_names = []
    
    for fname in sorted(os.listdir(renders_dir)):
        if not fname.endswith('.png'):
            continue
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    
    return renders, gts, image_names


def evaluate_ratios(model_paths, eval_ratios=None):
    """
    评估不同比例的渲染结果
    """
    if eval_ratios is None:
        eval_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    
    full_dict = {}
    per_view_dict = {}
    
    for scene_dir in model_paths:
        print(f"\n{'='*80}")
        print(f"评估场景：{scene_dir}")
        print(f"{'='*80}")
        
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        
        # 为每个比例进行评估
        for ratio in tqdm(eval_ratios, desc="评估比例"):
            ratio_dir = f"ratio_{ratio:.2f}".replace('.', '_')
            renders_dir = Path(scene_dir) / "test" / f"ours_30000" / f"renders_{ratio_dir}" / "renders"
            gt_dir = Path(scene_dir) / "test" / f"ours_30000" / f"renders_{ratio_dir}" / "gt"
            
            if not renders_dir.exists():
                print(f"  跳过比例 {ratio:.2f}：渲染目录不存在")
                continue
            
            # 读取图像
            renders, gts, image_names = read_images(renders_dir, gt_dir)
            
            psnr_list = []
            ssim_list = []
            lpips_list = []
            
            # 逐图像评估
            for i in range(len(renders)):
                psnr_val = psnr(renders[i], gts[i]).mean().double()
                ssim_val = ssim(renders[i], gts[i]).mean().double()
                lpips_val = lpips(renders[i], gts[i]).mean().double()
                
                psnr_list.append(float(psnr_val))
                ssim_list.append(float(ssim_val))
                lpips_list.append(float(lpips_val))
            
            # 计算平均值
            psnr_avg = sum(psnr_list) / len(psnr_list) if psnr_list else 0.0
            ssim_avg = sum(ssim_list) / len(ssim_list) if ssim_list else 0.0
            lpips_avg = sum(lpips_list) / len(lpips_list) if lpips_list else 0.0
            
            # 存储结果
            key = f"ratio_{ratio:.2f}"
            full_dict[scene_dir][key] = {
                "PSNR": psnr_avg,
                "SSIM": ssim_avg,
                "LPIPS": lpips_avg,
                "num_images": len(renders),
            }
            
            # 每视角结果
            per_view_dict[scene_dir][key] = {
                "PSNR": psnr_list,
                "SSIM": ssim_list,
                "LPIPS": lpips_list,
                "image_names": image_names,
            }
        
        # 打印该场景的汇总
        print(f"\n{scene_dir} 评估结果:")
        print(f"{'比例':<10} {'PSNR↑':<12} {'SSIM↑':<12} {'LPIPS↓':<12}")
        print(f"{'-'*50}")
        for ratio in eval_ratios:
            key = f"ratio_{ratio:.2f}"
            if key in full_dict[scene_dir]:
                result = full_dict[scene_dir][key]
                print(f"{ratio:<10.2f} {result['PSNR']:<12.4f} "
                      f"{result['SSIM']:<12.4f} {result['LPIPS']:<12.4f}")
    
    # 保存汇总报告
    if model_paths:
        report_path = os.path.join(model_paths[0], "mgs_ratios_eval.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(full_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估报告已保存至：{report_path}")
        
        # 保存每视角详细结果
        per_view_path = os.path.join(model_paths[0], "mgs_ratios_per_view.json")
        with open(per_view_path, 'w', encoding='utf-8') as f:
            json.dump(per_view_dict, f, indent=2, ensure_ascii=False)
        
        print(f"每视角详细结果已保存至：{per_view_path}")
    
    return full_dict, per_view_dict


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="MGS 多比例评估")
    parser.add_argument("--model_paths", nargs='+', required=True,
                        help="模型路径列表")
    parser.add_argument("--eval_ratios", type=float, nargs='+', 
                        default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
                        help="要评估的比例列表")
    
    args = parser.parse_args()
    
    # 评估
    evaluate_ratios(args.model_paths, args.eval_ratios)
