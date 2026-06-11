import argparse
import json
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune

warnings.filterwarnings("ignore")

PRUNE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PRUNE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_WEIGHTS = PROJECT_ROOT / "yolo26n.pt"
DEFAULT_DATA = PROJECT_ROOT / "data.yaml"
DEFAULT_OUTPUT_DIR = PRUNE_ROOT / "yolo26_structured_prune"


def parse_args():
    parser = argparse.ArgumentParser(description="Structured pruning for a trained Ultralytics YOLO26 model.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="训练好的 .pt 权重路径。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="数据集 data.yaml 路径。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="剪枝结果保存目录。")
    parser.add_argument("--amount", type=float, default=0.20, help="每个可剪 Conv 层的结构化通道剪枝比例。")
    parser.add_argument("--imgsz", type=int, default=640, help="验证/微调图片尺寸。")
    parser.add_argument("--device", default="0", help="设备，例如 0 或 cpu。")
    parser.add_argument("--batch", type=int, default=4, help="验证/微调 batch。")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers。")
    parser.add_argument("--finetune-epochs", type=int, default=30, help="剪枝后微调 epoch；0 表示不微调。")
    parser.add_argument("--lr0", type=float, default=0.001, help="微调初始学习率，建议小于原训练学习率。")
    parser.add_argument("--optimizer", default="SGD", help="微调优化器。")
    parser.add_argument("--no-val", action="store_true", help="跳过剪枝前后验证。")
    parser.add_argument("--include-detect-inner", action="store_true", help="同时剪检测头内部中间 Conv，仍会跳过最终输出 Conv。")
    parser.add_argument("--dry-run", action="store_true", help="只打印可剪层和统计，不保存权重。")
    return parser.parse_args()


def ensure_path(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在: {path}")


def is_detect_output_conv(name, module):
    if not isinstance(module, nn.Conv2d):
        return False
    if not name.startswith("model.23."):
        return False
    return name.endswith(".2") and module.out_channels <= 4


def is_prunable_conv(name, module, include_detect_inner=False):
    if not isinstance(module, nn.Conv2d):
        return False
    if module.groups != 1:
        return False
    if module.out_channels < 8:
        return False
    if is_detect_output_conv(name, module):
        return False
    if name.startswith("model.23.") and not include_detect_inner:
        return False
    return True


def conv_stats(model, include_detect_inner=False):
    layers = []
    for name, module in model.named_modules():
        if is_prunable_conv(name, module, include_detect_inner):
            layers.append(
                {
                    "name": name,
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": list(module.kernel_size),
                }
            )
    return layers


def count_zero_output_channels(module):
    weight = module.weight.detach()
    flat = weight.abs().flatten(1)
    return int((flat.sum(dim=1) == 0).sum().item())


def summarize_model(model, include_detect_inner=False):
    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum((p.detach() == 0).sum().item() for p in model.parameters())
    prunable_layers = conv_stats(model, include_detect_inner)
    zero_channels = {}
    for name, module in model.named_modules():
        if is_prunable_conv(name, module, include_detect_inner):
            zeros = count_zero_output_channels(module)
            if zeros:
                zero_channels[name] = {"zero_channels": zeros, "out_channels": module.out_channels}
    return {
        "parameters": int(total_params),
        "zero_parameters": int(zero_params),
        "zero_parameter_ratio": float(zero_params / total_params) if total_params else 0.0,
        "prunable_conv_layers": len(prunable_layers),
        "structured_zero_channels": zero_channels,
    }


def apply_structured_filter_pruning(model, amount, include_detect_inner=False):
    pruned = []
    for name, module in model.named_modules():
        if not is_prunable_conv(name, module, include_detect_inner):
            continue
        before = count_zero_output_channels(module)
        torch_prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
        torch_prune.remove(module, "weight")
        after = count_zero_output_channels(module)
        pruned.append(
            {
                "name": name,
                "out_channels": module.out_channels,
                "new_zero_channels": after - before,
                "total_zero_channels": after,
            }
        )
    return pruned


def run_val(yolo_model, args, name):
    metrics = yolo_model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.output_dir / "val"),
        name=name,
        exist_ok=True,
        verbose=False,
    )
    box = getattr(metrics, "box", None)
    if box is None:
        return {}
    return {
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
    }


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    ensure_path(args.weights, "权重文件")
    ensure_path(args.data, "数据集配置")
    if not 0.0 <= args.amount < 1.0:
        raise ValueError("--amount 必须在 [0, 1) 范围内。")

    from ultralytics import YOLO

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pruned_path = args.output_dir / f"{args.weights.stem}_structured_pruned_{args.amount:.2f}_{timestamp}.pt"
    report_path = pruned_path.with_suffix(".json")

    yolo = YOLO(str(args.weights))
    yolo.model.eval()

    prunable_layers = conv_stats(yolo.model, args.include_detect_inner)
    print(f"权重: {args.weights}")
    print(f"数据: {args.data}")
    print(f"剪枝比例: {args.amount:.2f}")
    print(f"可结构化剪枝 Conv 层数: {len(prunable_layers)}")
    if args.dry_run:
        for layer in prunable_layers:
            print(f"- {layer['name']}: {layer['in_channels']} -> {layer['out_channels']}, k={layer['kernel_size']}")
        return

    report = {
        "weights": str(args.weights),
        "data": str(args.data),
        "amount": args.amount,
        "include_detect_inner": args.include_detect_inner,
        "before": summarize_model(yolo.model, args.include_detect_inner),
        "baseline_metrics": None,
        "pruned_metrics": None,
        "finetuned_weights": None,
        "finetuned_metrics": None,
    }

    if not args.no_val:
        print("开始剪枝前验证...")
        report["baseline_metrics"] = run_val(yolo, args, "baseline")

    print("执行结构化滤波器剪枝...")
    report["pruned_layers"] = apply_structured_filter_pruning(yolo.model, args.amount, args.include_detect_inner)
    report["after_prune"] = summarize_model(yolo.model, args.include_detect_inner)

    yolo.save(str(pruned_path))
    print(f"剪枝权重已保存: {pruned_path}")

    reloaded = YOLO(str(pruned_path))
    with torch.no_grad():
        device = torch.device("cpu")
        reloaded.model.to(device).eval()(torch.zeros(1, 3, min(args.imgsz, 640), min(args.imgsz, 640), device=device))
    print("剪枝权重加载/前向检查通过。")

    if not args.no_val:
        print("开始剪枝后验证...")
        report["pruned_metrics"] = run_val(reloaded, args, "pruned")

    if args.finetune_epochs > 0:
        print(f"开始剪枝后微调 {args.finetune_epochs} epochs...")
        finetuned = YOLO(str(pruned_path))
        finetuned.train(
            data=str(args.data),
            imgsz=args.imgsz,
            epochs=args.finetune_epochs,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            optimizer=args.optimizer,
            lr0=args.lr0,
            project=str(args.output_dir / "finetune"),
            name=f"finetune_{timestamp}",
            exist_ok=True,
        )
        best = args.output_dir / "finetune" / f"finetune_{timestamp}" / "weights" / "best.pt"
        report["finetuned_weights"] = str(best) if best.exists() else None
        if best.exists() and not args.no_val:
            report["finetuned_metrics"] = run_val(YOLO(str(best)), args, "finetuned")

    save_report(report_path, report)
    print(f"剪枝报告已保存: {report_path}")


if __name__ == "__main__":
    main()
