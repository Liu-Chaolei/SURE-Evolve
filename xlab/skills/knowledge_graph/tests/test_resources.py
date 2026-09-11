from __future__ import annotations

import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from knowledge_graph_lib.mineru import _run_batch
from knowledge_graph_lib.resources import build_resource_plan


class ResourcePlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.options = {
            "output_path": self.root / "resources.json",
            "requested_backend": "pipeline",
            "requested_gpus": [],
            "max_mineru_workers": 4,
            "llm_workers": 2,
            "min_gpu_free_gib": 24.0,
            "max_gpu_utilization": 20,
        }
        self.command_patch = patch("knowledge_graph_lib.resources.resolve_mineru_command", return_value=self.root / "mineru")
        self.command_patch.start()

    def tearDown(self) -> None:
        self.command_patch.stop()
        self.temporary.cleanup()

    def test_cpu_workers_respect_allocation(self) -> None:
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "16"}), patch("os.sched_getaffinity", return_value=set(range(64))):
            plan = build_resource_plan(**self.options, requested_device="cpu")
            self.assertEqual(plan["mineru"]["workers"], 4)
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "16"}), patch("os.sched_getaffinity", return_value=set(range(8))):
            plan = build_resource_plan(**self.options, requested_device="cpu")
            self.assertEqual(plan["mineru"]["workers"], 1)

    def test_idle_cuda_device_is_eligible(self) -> None:
        gpu = {"index": "0", "memory_free_mib": 32768, "utilization_percent": 0}
        self.options["requested_backend"] = "auto"
        with patch("knowledge_graph_lib.resources.probe_gpus", return_value=([gpu], None)):
            plan = build_resource_plan(**self.options)
        self.assertEqual(plan["mineru"]["selected_gpus"], ["0"])
        self.assertEqual(plan["mineru"]["device"], "cuda")

    def test_npu_workers_only_use_visible_allocation(self) -> None:
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "2,5"}), patch("shutil.which", return_value="/usr/bin/npu-smi"):
            plan = build_resource_plan(**self.options, requested_device="npu")
        self.assertEqual(plan["mineru"]["selected_npus"], ["2", "5"])
        self.assertEqual(plan["mineru"]["workers"], 2)
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": ""}):
            with self.assertRaisesRegex(ValueError, "Ascend allocation"):
                build_resource_plan(**self.options, requested_device="npu")

    def test_timeout_terminates_worker_process_group(self) -> None:
        process = Mock()
        process.pid = 12345
        process.wait.side_effect = [subprocess.TimeoutExpired("mineru", 5), 0]
        with patch("subprocess.Popen", return_value=process) as launch, patch("os.killpg") as kill:
            code, error = _run_batch(command=self.root / "mineru", input_dir=self.root, output_dir=self.root / "output", backend="pipeline", method="auto", language="en", timeout=5, gpu=None, log_path=self.root / "worker.log")
        self.assertEqual(code, 124)
        self.assertIn("exceeded", error)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertEqual(kill.call_args.args[0], process.pid)

    def test_npu_launcher_bounds_onnx_threads_without_changing_provider(self) -> None:
        calls = []

        class Options:
            intra_op_num_threads = 0
            inter_op_num_threads = 0

            def add_session_config_entry(self, key: str, value: str) -> None:
                return None

        class Session:
            def __init__(self, path, options, providers, provider_options, **kwargs):
                calls.append((path, options, providers, provider_options, kwargs))

        ort = SimpleNamespace(InferenceSession=Session, SessionOptions=Options)
        npu = SimpleNamespace(npu=SimpleNamespace(set_compile_mode=Mock()))
        main = Mock()
        modules = {"onnxruntime": ort, "torch_npu": npu, "mineru.cli.client": SimpleNamespace(main=main)}
        script = Path(__file__).resolve().parents[1] / "scripts/mineru_npu.py"
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"OMP_NUM_THREADS": "16", "ASCEND_CACHE_PATH": str(self.root / "cache")}), patch("os.sched_getaffinity", return_value=set(range(8))):
            runpy.run_path(str(script), run_name="npu_launcher_test")
            Session("model.onnx", providers=["CPUExecutionProvider"])
        npu.npu.set_compile_mode.assert_called_once_with(jit_compile=False)
        main.assert_not_called()
        self.assertTrue((self.root / "cache").is_dir())
        self.assertEqual(calls[0][1].intra_op_num_threads, 8)
        self.assertEqual(calls[0][1].inter_op_num_threads, 1)
        self.assertEqual(calls[0][2], ["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
