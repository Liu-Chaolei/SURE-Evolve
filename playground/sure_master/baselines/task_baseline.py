"""Fixed task baseline; implementation and resources are selected by the adapter."""

import os
import subprocess


def main():
    subprocess.run(
        [
            os.environ["SURE_WORKER_PYTHON"],
            os.environ["SURE_TASK_WRAPPER"],
            "--action",
            "baseline",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
