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
import numpy as np


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


def get_gaussian_count(scene_dir, ratio):
    """
    从点云文件获取高斯数量
    
    尝试从点云ply文件读取实际的高斯点数量
    """
    import glob
    
    # 尝试找到点云文件
    point_cloud_dir = Path(scene_dir) / "point_cloud"
    if point_cloud_dir.exists():
        # 查找 iteration 文件夹
        iter_dirs = list(point_cloud_dir.glob("iteration_*"))
        if iter_dirs:
            # 使用最新的 iteration
            iter_dir = sorted(iter_dirs)[-1]
            ply_files = list(iter_dir.glob("*.ply"))
            if ply_files:
                try:
                    # 读取 ply 文件获取点数
                    from plyfile import PlyData
                    plydata = PlyData.read(str(ply_files[0]))
                    num_points = len(plydata['vertex'])
                    # 计算该比例下的高斯数量
                    return num_points
                except:
                    pass
    
    # 如果无法读取，返回 None
    return None


def evaluate_ratios(model_paths, eval_ratios=None):
    """
    评估不同比例的渲染结果
    """
    if eval_ratios is None:
        eval_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    
    for scene_dir in model_paths:
        print(f"\n{'='*80}")
        print(f"评估场景：{scene_dir}")
        print(f"{'='*80}")
        
        # 结果字典
        results = {
            "scene": scene_dir,
            "evaluations": []
        }

        # 获取高斯数量
        total_count = get_gaussian_count(scene_dir)
        
        # 为每个比例进行评估
        for ratio in tqdm(eval_ratios, desc="评估比例"):
            ratio_dir = f"ratio_{ratio:.2f}".replace('.', '_')
            
            # 新的路径结构：test/ratio_* 直接包含渲染图，test/gt 包含GT
            renders_dir = Path(scene_dir) / "test" / ratio_dir
            gt_dir = Path(scene_dir) / "test" / "gt"
            
            if not renders_dir.exists():
                print(f"  跳过比例 {ratio:.2f}：渲染目录不存在 {renders_dir}")
                continue
            
            if not gt_dir.exists():
                print(f"  跳过比例 {ratio:.2f}：GT目录不存在 {gt_dir}")
                continue
            
            # 读取图像
            try:
                renders, gts, image_names = read_images(renders_dir, gt_dir)
            except Exception as e:
                print(f"  读取图像失败：{e}")
                continue
            
            if len(renders) == 0:
                print(f"  跳过比例 {ratio:.2f}：未找到渲染图像")
                continue
            
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
            
            gaussian_count = max(1, int(total_count * ratio))
            
            # 存储结果
            eval_result = {
                "ratio": ratio,
                "gaussian_count": gaussian_count,
                "total_gaussians": total_count,
                "num_images": len(renders),
                "PSNR": round(psnr_avg, 4),
                "SSIM": round(ssim_avg, 4),
                "LPIPS": round(lpips_avg, 4),
                "per_view": {
                    "PSNR": [round(x, 4) for x in psnr_list],
                    "SSIM": [round(x, 4) for x in ssim_list],
                    "LPIPS": [round(x, 4) for x in lpips_list],
                    "image_names": image_names,
                }
            }
            results["evaluations"].append(eval_result)
        
        # 打印该场景的汇总
        print(f"\n{scene_dir} 评估结果:")
        print(f"{'比例':<10} {'高斯数量':<12} {'PSNR↑':<12} {'SSIM↑':<12} {'LPIPS↓':<12}")
        print(f"{'-'*60}")
        for eval_result in results["evaluations"]:
            ratio = eval_result["ratio"]
            count = eval_result["gaussian_count"] if eval_result["gaussian_count"] else "N/A"
            count_str = f"{count:,}" if isinstance(count, int) else str(count)
            print(f"{ratio:<10.2f} {count_str:<12} {eval_result['PSNR']:<12.4f} "
                  f"{eval_result['SSIM']:<12.4f} {eval_result['LPIPS']:<12.4f}")
        
        # 保存结果到场景文件夹内的 results.json
        results_path = os.path.join(scene_dir, "results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至：{results_path}")
    
    return results


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
