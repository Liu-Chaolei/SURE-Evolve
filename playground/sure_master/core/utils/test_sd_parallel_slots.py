"""Local processes exercise the same leases as the Slurm candidate launcher."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from .sd_allocations import validate_jobs
from .slurm_allocations import run_in_allocations


class ParallelSlotTests(unittest.TestCase):
    def test_sse_done_does_not_wait_for_proxy_disconnect(self):
        from playground.sure_master.tools.completion_transport import completion_transport
        response = SimpleNamespace(headers={"Content-Type": "text/event-stream"}, getcode=lambda: 200)
        lines = iter([b'data: {"choices":[]}\n', b'data: [DONE]\n'])
        response.readline = lambda: next(lines)
        from unittest.mock import MagicMock
        manager = MagicMock()
        manager.__enter__.return_value = response
        with patch('urllib.request.urlopen', return_value=manager) as send:
            status, _, body = completion_transport('https://example.invalid/chat/completions', {}, b'{}', 1)
        self.assertEqual(send.call_args.args[0].get_header('User-agent'), 'SURE-Evolve/1.0')
        self.assertEqual(status, 200)
        self.assertTrue(body.endswith(b'data: [DONE]\n'))

    def settings(self, root):
        return {"existing_allocations": ["10085", "10086"], "sd_restricted_pool": True,
                "allocation_slots_per_job": 2, "allocation_overlap": False,
                "allocation_cpus": 32, "allocation_step_memory": "120G",
                "allocation_lock_dir": str(root / "locks")}

    def test_rejects_oversubscribed_profiles(self):
        settings = self.settings(Path("/tmp/test-slots"))
        validate_jobs(settings)
        for change in ({"allocation_cpus": 64}, {"allocation_step_memory": "256G"},
                       {"allocation_overlap": True}, {"allocation_slots_per_job": 3},
                       {"existing_allocations": ["10085", "10086", "10087", "10088", "10094"]}):
            with self.assertRaises(ValueError):
                validate_jobs({**settings, **change})

    def test_four_live_processes_and_fifth_waits(self):
        self._check_parallel_limit(["10085", "10086"])

    def test_eight_live_processes_and_ninth_waits(self):
        self._check_parallel_limit(["10085", "10086", "10087", "10088"])

    def _check_parallel_limit(self, jobs):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "srun"
            fake.write_text('''#!/usr/bin/env python3
import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[-2]);result=Path(sys.argv[-1])
job=next(x.split('=',1)[1] for x in sys.argv if x.startswith('--jobid='))
assert '--exclusive' in sys.argv and '--overlap' not in sys.argv
assert '--cpus-per-task=32' in sys.argv and '--mem=120G' in sys.argv
assert '--gres=gpu:ascend910b3:4' in sys.argv
marker=root/(job+'-'+str(os.getpid())+'.active');marker.touch()
while not (root/'release').exists():time.sleep(.05)
result.write_text(json.dumps({'job':job,'args':sys.argv[1:]}));marker.unlink()
''')
            fake.chmod(0o700)
            settings = self.settings(root)
            settings["existing_allocations"] = jobs
            capacity = len(jobs) * 2
            with patch('playground.sure_master.core.utils.slurm_allocations.owned_running', return_value=True), \
                 patch('playground.sure_master.core.utils.sd_allocations.acquire', return_value=True):
                with ThreadPoolExecutor(max_workers=capacity + 1) as pool:
                    futures = []
                    for index in range(capacity + 1):
                        candidate = root / str(index)
                        candidate.mkdir()
                        futures.append(pool.submit(run_in_allocations, settings,
                            [str(fake), '--gres=gpu:ascend910b3:4', str(root), str(candidate/'result.json')],
                            candidate, candidate/'result.json'))
                    try:
                        deadline = time.monotonic() + 12
                        while len(list(root.glob('*.active'))) < capacity and time.monotonic() < deadline:
                            time.sleep(.05)
                        markers = list(root.glob('*.active'))
                        self.assertEqual(len(markers), capacity)
                        for job in jobs:
                            self.assertEqual(len(list(root.glob(job+'-*.active'))), 2)
                            with (root/'locks'/f'{job}.lock').open('a') as gate:
                                with self.assertRaises(BlockingIOError):
                                    fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        self.assertFalse(any(f.done() for f in futures))
                    finally:
                        (root/'release').touch()
                    self.assertTrue(all(f.result(timeout=15) for f in futures))
                    receipts = [json.loads((root/str(i)/'allocation.json').read_text()) for i in range(capacity + 1)]
                    self.assertTrue(all(r['slot'] in {0,1} for r in receipts))


if __name__ == '__main__':
    unittest.main()
