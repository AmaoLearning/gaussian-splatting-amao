"""
MGS 多比例渲染脚本

渲染不同高斯数量比例的子集，默认比例：[1.0, 0.9, ..., 0.01]
"""
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from mgs import MGSTrainingHelper

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set_with_ratios(
    model_path, 
    name, 
    iteration, 
    views, 
    gaussians, 
    pipeline, 
    background,
    train_test_exp,
    separate_sh,
    eval_ratios,
    sort_strategy="by_opacity_descending",
    quiet=False,
):
    """
    按不同比例渲染测试集
    
    Args:
        eval_ratios: 要评估的比例列表
        sort_strategy: 排序策略
        quiet: 是否禁用进度条
    """
    # 初始化 MGS 辅助
    mgs_helper = MGSTrainingHelper(
        use_mgs=True,
        sort_strategy=sort_strategy,
        update_interval=1,
    )
    
    # 更新排序索引
    if not quiet:
        print(f"\n{'='*60}")
        print(f"渲染多比例子集")
        print(f"{'='*60}")
        print(f"评估比例：{eval_ratios}")
        print(f"排序策略：{sort_strategy}")
    
    mgs_helper.update_sort_indices(gaussians, iteration=0)
    total_gaussians = len(mgs_helper._sort_indices)
    
    if not quiet:
        print(f"\n总高斯点数量：{total_gaussians:,}")
        print(f"\n渲染目录：{model_path}/{name}/ours_{iteration}")
    
    # 为每个比例渲染
    for ratio_idx, ratio in enumerate(tqdm(eval_ratios, desc="Rendering ratios", disable=quiet)):
        ratio_dir = f"ratio_{ratio:.2f}".replace('.', '_')
        render_path = os.path.join(
            model_path, 
            name, 
            f"ours_{iteration}", 
            f"renders_{ratio_dir}",
            "renders"
        )
        gts_path = os.path.join(
            model_path, 
            name, 
            f"ours_{iteration}", 
            f"renders_{ratio_dir}",
            "gt"
        )
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        
        # 计算该比例的高斯数量
        keep_count = max(1, int(total_gaussians * ratio))
        
        if not quiet:
            print(f"\n比例 {ratio:.2f}: {keep_count:,} / {total_gaussians:,} 个点 ({ratio*100:.1f}%)")
        
        # 获取子集参数
        subset_params = mgs_helper.get_subset_for_ratio(gaussians, ratio, iteration=0)
        
        # 渲染所有视图
        for idx, view in enumerate(tqdm(views, desc=f"Ratio {ratio:.2f}", disable=quiet, leave=False)):
            rendering = render(
                view, 
                gaussians, 
                pipeline, 
                background, 
                overrides=subset_params,
                use_trained_exp=train_test_exp, 
                separate_sh=separate_sh
            )["render"]
            
            gt = view.original_image[0:3, :, :]
            
            if train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
                gt = gt[..., gt.shape[-1] // 2:]
            
            torchvision.utils.save_image(
                rendering, 
                os.path.join(render_path, '{0:05d}'.format(idx) + ".png")
            )
            
            # GT 只需保存一次（第一个比例）
            if ratio_idx == 0:
                torchvision.utils.save_image(
                    gt, 
                    os.path.join(gts_path, '{0:05d}'.format(idx) + ".png")
                )
    
    if not quiet:
        print(f"\n{'='*60}")
        print(f"渲染完成！")
        print(f"{'='*60}")


def render_sets_with_mgs(
    dataset : ModelParams, 
    iteration : int, 
    pipeline : PipelineParams, 
    skip_train : bool, 
    skip_test : bool, 
    separate_sh: bool,
    eval_ratios: list = None,
    sort_strategy: str = "by_opacity_descending",
    quiet: bool = False,
):
    """
    渲染不同比例的高斯子集
    """
    if eval_ratios is None:
        eval_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set_with_ratios(
                dataset.model_path, 
                "train", 
                scene.loaded_iter, 
                scene.getTrainCameras(), 
                gaussians, 
                pipeline, 
                background, 
                dataset.train_test_exp, 
                separate_sh,
                eval_ratios,
                sort_strategy,
                quiet,
            )

        if not skip_test:
            render_set_with_ratios(
                dataset.model_path, 
                "test", 
                scene.loaded_iter, 
                scene.getTestCameras(), 
                gaussians, 
                pipeline, 
                background, 
                dataset.train_test_exp, 
                separate_sh,
                eval_ratios,
                sort_strategy,
                quiet,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="MGS 多比例渲染")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--eval_ratios", type=float, nargs='+', 
                        default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
                        help="要评估的比例列表")
    parser.add_argument("--sort_strategy", type=str, default="by_opacity_descending",
                        help="排序策略")
    
    args = get_combined_args(parser)
    print("MGS 多比例渲染 " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # 渲染
    render_sets_with_mgs(
        model.extract(args), 
        args.iteration, 
        pipeline.extract(args), 
        args.skip_train, 
        args.skip_test, 
        SPARSE_ADAM_AVAILABLE,
        eval_ratios=args.eval_ratios,
        sort_strategy=args.sort_strategy,
        quiet=args.quiet,
    )
