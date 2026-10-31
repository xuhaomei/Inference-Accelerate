"""命令行入口；env 子命令可在未安装模型依赖时运行。"""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    # 解析命令行参数，并在子命令中调用对应的函数，子命令包括：
    # - env：检查环境，不加载模型
    # - synth：生成确定性的检索/三跳关联检索任务
    # - run：运行实验
    # - profile：独立 PyTorch Profiler，不用于性能结论
    # - report：从已有原始记录重新生成报告
    parser = argparse.ArgumentParser(prog="kvlab", description="推理状态管理与数据访问优化实验平台")
    sub = parser.add_subparsers(dest="command", required=True)
    env = sub.add_parser("env", help="检查环境，不加载模型")
    env.add_argument("--config", help="额外验证配置并检查目标设备")
    env.add_argument("--output", help="保存环境 JSON")
    synth = sub.add_parser("synth", help="生成确定性的检索/三跳关联检索任务")
    synth.add_argument("--output", default="data/synthetic.jsonl")
    synth.add_argument("--count", type=int, default=8)
    synth.add_argument("--records", type=int, default=128)
    synth.add_argument("--seed", type=int, default=42)
    for command, help_text in (("run", "运行实验"), ("profile", "独立 PyTorch Profiler，不用于性能结论"),
                               ("validate", "独立固定轨迹预检，不执行正式测量")):
        p = sub.add_parser(command, help=help_text)
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖配置，可重复")
    report = sub.add_parser("report", help="从已有原始记录重新生成报告")
    report.add_argument("directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "env":
            from .environment import environment
            info = environment()
            if args.config:
                from .config import load
                c = load(args.config)
                info["config_valid"] = True
                info["requested_model"] = c["model"]
                info["device_ready"] = info["cuda_available"] if c["model"]["device"].startswith("cuda") else True
            text = json.dumps(info, ensure_ascii=False, indent=2)
            print(text)
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(text + "\n")
            if args.config and not info["device_ready"]:
                return 2
        elif args.command == "synth":
            from .data import synthetic
            rows = synthetic(args.output, args.count, args.records, args.seed)
            print(f"已保存 {len(rows)} 个样本：{args.output}")
        elif args.command in {"run", "profile", "validate"}:
            from .config import load
            from .runner import run
            run(load(args.config, args.set), args.config, args.set, profile=args.command == "profile",
                validation_only=args.command == "validate")
        elif args.command == "report":
            from .report import build_report
            print(build_report(args.directory).resolve())
        return 0
    except Exception as exc:
        print(f"失败 [{type(exc).__name__}]：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
