"""Observe the real training container; never run calibration or model work."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time


def start_usage_monitor(output: Path) -> None:
    def number(paths):
        for path in paths:
            try:
                return int(Path(path).read_text().strip())
            except (OSError, ValueError):
                continue
        return None

    def observe():
        while True:
            try:
                rss = 0
                processes = 0
                for process in Path('/proc').iterdir():
                    if not process.name.isdigit():
                        continue
                    try:
                        rss += int((process / 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
                        processes += 1
                    except (OSError, ValueError, IndexError):
                        continue
                sample = {'time': time.time(), 'job_id': os.environ.get('SLURM_JOB_ID'),
                    'step_id': os.environ.get('SLURM_STEP_ID'), 'cpu_affinity_count': len(os.sched_getaffinity(0)),
                    'container_rss_sum_bytes': rss, 'container_processes': processes,
                    'cgroup_memory_bytes': number(['/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes']),
                    'cgroup_memory_limit_bytes': number(['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'])}
                with output.open('a') as stream:
                    stream.write(json.dumps(sample) + '\n')
            except OSError:
                pass
            time.sleep(30)

    threading.Thread(target=observe, name='asr-resource-monitor', daemon=True).start()
