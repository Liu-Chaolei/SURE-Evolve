#!/usr/bin/env python3
"""Small task components; never starts XLab, a search loop, or full evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from playground.sure_master.runtime.accelerator import runtime_environment
from playground.sure_master.tasks import get_adapter


def expand(value):
    if isinstance(value, str):
        result = os.path.expandvars(value)
        if "${" in result:
            raise ValueError(f"Unset configuration variable: {value}")
        return result
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value


def model_probe(adapter, settings, backend, component, workspace):
    """Synthetic tiny-batch check; this is not a dataset training experiment."""
    from playground.sure_master.runtime.accelerator import RuntimeBackend
    from playground.sure_master.runtime.model_source import (
        snapshot_source,
        prepare_f5_source,
    )

    runtime = RuntimeBackend(backend)
    runtime.seed(42)
    torch = runtime.torch
    resources = {k: Path(v) for k, v in settings["resources"].items()}
    root = snapshot_source(resources["source"], Path("working/probe_source"))
    if adapter.name == "tts.f5tts":
        prepare_f5_source(root)
        sys.path.insert(0, str(root / "src"))
        from hydra.utils import get_class
        from f5_tts.infer.utils_infer import load_model

        config = yaml.safe_load(
            (root / "src/f5_tts/configs/F5TTS_v1_Base.yaml").read_text()
        )
        arch = config["model"]["arch"]
        if component == "arch":
            arch["depth"] = 20
        if component == "arch":
            from f5_tts.model import CFM
            from safetensors.torch import load_file
            from f5_tts.model.utils import get_tokenizer

            tokenizer, size = get_tokenizer(str(resources["vocab"]), "custom")
            model = CFM(
                transformer=get_class("f5_tts.model." + config["model"]["backbone"])(
                    **arch, text_num_embeds=size
                ),
                vocab_char_map=tokenizer,
            )
            state = load_file(str(resources["checkpoint"]), device="cpu")
            state = {
                k.removeprefix("ema_model."): v
                for k, v in state.items()
                if k not in {"initted", "step"}
            }
            own = model.state_dict()
            matched = {
                k: v for k, v in state.items() if k in own and v.shape == own[k].shape
            }
            if (
                sum(v.numel() for v in matched.values())
                / sum(v.numel() for v in own.values())
                < 0.7
            ):
                raise ValueError("F5 probe partial-load match below .70")
            model.load_state_dict(matched, strict=False)
            model.to(runtime.device)
        else:
            model = load_model(
                get_class("f5_tts.model." + config["model"]["backbone"]),
                arch,
                str(resources["checkpoint"]),
                vocab_file=str(resources["vocab"]),
                device=str(runtime.device),
            )

        def loss_fn():
            return model(
                inp=torch.randn(1, 32, 100, device=runtime.device), text=["你好"]
            )[0]
    else:
        import toml
        from playground.sure_master.tasks.training_resources import validate_wavlm_provenance
        validate_wavlm_provenance(resources["wavlm"])
        sys.path[:0] = [str(root), str(root / "pyannote-audio")]
        config = toml.load(root / "recipes/diar_ssl/conf/wavlm_updated_conformer.toml")
        args = config["model"]["args"]
        args["wavlm_src"] = str(resources["wavlm"])
        if component == "arch":
            args["num_layer"] = 3
        module, cls = config["model"]["path"].rsplit(".", 1)
        model = getattr(importlib.import_module(module), cls)(**args)
        state = torch.load(resources["wavlm"], map_location="cpu", weights_only=True)
        model.wavlm_model.load_state_dict(state["state_dict"], strict=True)
        model.to(runtime.device)

        def loss_fn():
            return -model(torch.randn(1, 1, 16000, device=runtime.device))[
                ..., 0
            ].mean()

    model.train()
    runtime.verify_model(model)
    loss = loss_fn()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite component loss")
    loss.backward()
    runtime.verify_model(model, gradients=True)
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not all(torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError("Nonfinite gradients")
    parameter = next(p for p in parameters if p.grad.abs().max() > 0)
    before = parameter.detach().clone()
    torch.optim.SGD(parameters, lr=1e-3).step()
    if torch.equal(before, parameter):
        raise ValueError("No parameter update")
    checkpoint = workspace / "probe_checkpoint.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, checkpoint)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    return {
        "status": "passed",
        "component": component,
        "adapter": adapter.name,
        "accelerator": backend,
        "synthetic_batch": True,
        "optimizer_steps": 1,
        "loss": float(loss.detach().cpu()),
        "save_restore": True,
    }


def child(request, workspace):
    adapter = get_adapter(request["adapter"])
    sure = request["sure"]
    settings = sure.get("task", {})
    source = settings.get("resources", {}).get("source")
    if source:
        sys.path[:0] = [
            source,
            str(Path(source) / "src"),
            str(Path(source) / "pyannote-audio"),
        ]
    component = request["component"]
    if component == "imports":
        from playground.sure_master.runtime.accelerator import check_accelerator

        info = check_accelerator(sure["runtime"]["accelerator"])
        for name in adapter.modules:
            importlib.import_module(name)
        return {
            "status": "passed",
            "component": component,
            "adapter": adapter.name,
            **info,
        }
    if component in {"train", "arch"}:
        if adapter.task == "asr":
            root = sure["base_models"][sure["task_id"]]["source_paths"]["root"]
            script = Path(__file__).with_name("probe_zipformer.py")
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--icefall",
                    root,
                    "--backend",
                    sure["runtime"]["accelerator"],
                    "--output",
                    str(workspace / "asr_probe"),
                ],
                check=True,
            )
            return json.loads((workspace / "asr_probe/probe_result.json").read_text())
        return model_probe(
            adapter, settings, sure["runtime"]["accelerator"], component, workspace
        )
    if component == "inference" and adapter.task == "asr":
        raise ValueError(
            "Use the existing ASR smoke_baseline.py for audio decoding; this tool provides the bounded forward/backward probe"
        )
    from playground.sure_master.tools.run_task_candidate import run

    env = adapter.environment(sure)
    env.update({str(k): str(v) for k, v in sure.get("execution_env", {}).items()})
    raw = request.get("manifest") or env.get("SURE_EVAL_MANIFEST")
    if not raw:
        raise ValueError("Inference/replay probe requires --manifest")
    data = [
        json.loads(line) for line in Path(raw).read_text().splitlines() if line.strip()
    ][:1]
    if not data:
        raise ValueError("Empty probe manifest")
    selected = workspace / "probe_manifest.jsonl"
    selected.write_text(json.dumps(data[0], ensure_ascii=False) + "\n")
    env["SURE_EVAL_MANIFEST"] = str(selected)
    env.update(SURE_COMPONENT_TEST="1", SURE_CANDIDATE_PHASE="component_test")
    os.environ.update(env)
    run("infer", {}, request.get("artifact", ""))
    bundle = {"model_artifact": request.get("artifact", ""), "synthetic_or_reference_probe": True}
    return {
        "status": "passed",
        "component": component,
        "adapter": adapter.name,
        "samples": 1,
        **bundle,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--component",
        choices=["imports", "inference", "train", "arch", "replay"],
        default="imports",
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-artifact", default="")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if args.child:
        request = json.loads(args.child.read_text())
        os.chdir(workspace)
        result = child(request, workspace)
        (workspace / "probe_result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return
    if not args.config:
        parser.error("--config is required")
    if args.component == "replay" and not args.model_artifact:
        parser.error("--component replay requires --model-artifact")
    raw = yaml.safe_load(args.config.read_text())["sure"]
    # Component probes intentionally do not resolve LLM credentials or survey settings.
    sure = expand(
        {
            k: v
            for k, v in raw.items()
            if k
            in {
                "runtime",
                "adapter",
                "task",
                "task_id",
                "execution_env",
                "datasets",
                "base_models",
            }
        }
    )
    request = {
        "sure": sure,
        "adapter": sure["adapter"],
        "component": args.component,
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "artifact": args.model_artifact,
    }
    path = workspace / "probe_request.json"
    path.write_text(json.dumps(request, indent=2))
    env = {**os.environ, **runtime_environment(sure["runtime"])}
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[3])
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    try:
        with (workspace / "probe.log").open("w") as log:
            from playground.sure_master.runtime.process import run_bounded

            run_bounded(
                [
                    sure["runtime"]["python"],
                    str(Path(__file__).resolve()),
                    "--child",
                    str(path),
                    "--workspace",
                    str(workspace),
                ],
                env=env,
                output=log,
                timeout=args.timeout,
            )
        result = json.loads((workspace / "probe_result.json").read_text())
    except (OSError, subprocess.SubprocessError) as exc:
        result = {
            "status": "unverified",
            "component": args.component,
            "adapter": sure["adapter"],
            "accelerator": sure["runtime"]["accelerator"],
            "error": str(exc),
            "log": str(workspace / "probe.log"),
        }
        (workspace / "probe_result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
