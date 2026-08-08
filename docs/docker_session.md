# Running a Playground in Docker Session

EvoMaster agents drive their tools (bash, editor, skill scripts, …) through a
**Session**. By default the session is `local`, i.e. commands run on your host.
Switching to a `docker` session runs every tool call inside an isolated
container instead, giving you:

* A clean, reproducible runtime that does not pollute the host (great for
  untrusted agent output, long-running science workloads, or CI).
* Per-run resource caps (memory, CPU set, GPU set, network mode).
* A writable workspace that is automatically mounted back to
  `runs/<run_dir>/workspace/` on the host, so you can inspect artifacts from
  outside the container exactly as the agent saw them.

All of this only requires editing one block of YAML — the agent and tool
code is untouched.

> The canonical end-to-end examples live under `configs/<playground>/docker.yaml`
> for the five reference playgrounds referenced at the bottom of this page
> (`minimal`, `minimal_kaggle`, `minimal_multi_agent`,
> `minimal_multi_agent_parallel`, `minimal_skill_task`). Copy one of those when
> you start.

---

## TL;DR — three steps

1. Make sure Docker is installed and the daemon is reachable (`docker info`).
2. In your playground's config, set:
   ```yaml
   session:
     type: "docker"
     docker:
       image: "python:3.11-slim"        # any image that ships bash
       working_dir: "/workspace"
       memory_limit: "8g"
       cpu_limit: 4.0
       auto_remove: true
   ```
3. Run as usual:
   ```bash
   python run.py --agent minimal --config configs/minimal/docker.yaml \
       --task "Your task description"
   ```

That's it — EvoMaster pulls the image on first use (policy `missing`), spins up
a container, mounts the run workspace at `working_dir`, and tears everything
down when the run finishes. Ctrl-C is also safe: the container is removed as
long as `auto_remove: true`.

---

## How it works under the hood

When `session.type == "docker"`, the playground builds a
[`DockerSession`](../evomaster/agent/session/docker.py) backed by a
[`DockerEnv`](../evomaster/env/docker.py). On `open()` it:

1. Calls `docker version` to fail fast if the daemon is not reachable.
2. Creates symlinks on the host so that bind mounts nested inside
   `working_dir` (e.g. `/workspace/input`) also appear under
   `runs/<run_dir>/workspace/input` on the host.
3. Pulls the image if necessary (`pull_image` policy).
4. Starts the container detached with `tail -f /dev/null` as PID 1, so it
   stays alive between tool calls.
5. Initializes a per-thread state directory inside the container
   (`/tmp/evomaster_session_<uid>/thread_<tid>`) used to persist the current
   working directory across subsequent `exec_bash` calls.

Every `agent.execute_bash("…")` then becomes exactly one `docker exec …` call.
The wrapper script `cd`s into the saved `cwd`, runs the user command, writes
the new `pwd` back to the state file, and prints a marker so the session can
parse the real exit code. **Consequences worth knowing:**

* `cd <dir>` in one tool call is visible to the next tool call (same thread).
* Exported env vars and background jobs **do not** survive between calls —
  chain them with `&&` if you need them in the same step.
* Parallel exps running in the same container each get their own cwd state
  (per-thread), so they do not stomp on each other.
* `is_input` (interactive stdin) is not supported and returns a clear error;
  use heredocs or chained commands instead.

File I/O is accelerated when the path sits inside one of your bind mounts:
`upload`, `download`, `read_file`, `write_file` operate on the host path
directly and only fall back to `docker cp` for paths that are **not** mounted.

---

## Configuration reference

Add this block under your playground config. Unless noted, fields map 1:1 to
`docker run` flags.

```yaml
session:
  type: "docker"

  docker:
    # --- image / container identity ---
    image: "python:3.11-slim"
    container_name: null          # null -> auto-generated per run
    use_existing_container: null  # attach to a container you already manage

    # --- paths ---
    working_dir: "/workspace"     # EvoMaster mounts the run workspace here

    # --- resources ---
    memory_limit: "8g"            # --memory (and --memory-swap, to disable swap)
    cpu_limit: 4.0                # --cpus  (CPU *time* quota)
    cpu_devices: null             # --cpuset-cpus: "0-15" | "0,2,4" | [0,1,2] | null
    gpu_devices: null             # "all" | "0" | "0,1" | ["0","1"] | null
    network_mode: "bridge"        # bridge | host | none | <custom>

    # --- mounts / env ---
    volumes: {}                   # {host_path: container_path}
    env_vars: {}                  # {KEY: VALUE}

    # --- lifecycle ---
    auto_remove: true             # true -> docker run --rm; false -> leave running
    pull_image: "missing"         # missing | always | never
    timeout: 300                  # default per-command timeout (seconds)
```

### Field-by-field notes

| Field | Notes |
|-------|-------|
| `image` | Must include `bash`. `python:3.11-slim` works out of the box; for ML workloads use a CUDA-ready image (see `evomaster/env/docker/base.dockerfile`). |
| `container_name` | Fixed name if you want to reuse the same container across runs. Combine with `auto_remove: false`. |
| `use_existing_container` | Attach to an already-running container by name or id. **EvoMaster never stops or removes a container it attached to.** |
| `working_dir` | Session cwd **and** workspace mount point. The run workspace (`runs/<run_dir>/workspace/`) is auto-mounted here — you do not have to list it under `volumes`. |
| `memory_limit` | Kernel cgroup cap. `htop`/`free` inside the container still show the host's total memory because `/proc/meminfo` is not namespaced — this is normal. |
| `cpu_limit` | Throttles CPU *time*, not core visibility. Use `cpu_devices` if you also want the container to *see* fewer cores. |
| `cpu_devices` | CPU pinning (`--cpuset-cpus`). Accepts `"0-15"`, `"0,2,4"`, a Python list, or `null`. |
| `gpu_devices` | `"all"`, a single id, comma string, or list. `null` / `"none"` disables GPUs. |
| `volumes` | `{host_path: container_path}`. Relative host paths (e.g. `./assets`) are resolved against the **project root** (the directory containing `evomaster/`), not the shell cwd. |
| `env_vars` | Extra env vars. You can reuse `${VAR}` substitution if your loader supports it. |
| `auto_remove` | `true` passes `--rm` and removes the container at session close. `false` keeps it around so the next run can attach to it. |
| `pull_image` | `missing` (default) pulls only if absent; `always` pulls every run; `never` fails if the image is not already local. |
| `timeout` | Per-command default. Individual tool calls may override it. |

### Special behavior: host-side symlinks for nested mounts

If a bind-mount target **nests inside `working_dir`** — e.g.
`./data/public → /workspace/input` — EvoMaster additionally creates a
host-side symlink:

```
runs/<run_dir>/workspace/input -> <abs path to ./data/public>
```

so that browsing `runs/<run_dir>/workspace/` from outside the container shows
the same tree the agent sees. Without this workaround the host view of
`/workspace/input` would be empty, because Docker layers the nested mount on
top of the workspace mount inside the container only. This is automatic — no
config needed.

### Path translation for tools and skills

Tools that accept **host** paths (e.g. `execute_bash` invoking a skill
script on disk) go through `DockerSession.to_session_path()` before the command
leaves the host. The session walks your `volumes` and rewrites any host path
covered by a bind mount to its in-container counterpart. This is why the
`minimal_skill_task` Docker config mounts `./evomaster` at
`/workspace/evomaster` — the `use_skill` tool needs the skill scripts to be
visible inside the container. If you see `file not found` errors from
`use_skill` or a custom tool, check that the file's host path lives under one
of your `volumes` entries.

---

## Parallel mode (`minimal_multi_agent_parallel`)

For playgrounds that run multiple experiments in parallel, add these two
extra keys under `session.docker`:

```yaml
session:
  type: "docker"
  docker:
    # ...standard fields as above...

    # Total budget across all parallel exps; divided per container.
    memory_limit: "48g"      # 48g / 3 exps = 16g per container
    cpu_limit: 36.0          # 36  / 3 exps = 12 per container
    cpu_devices: "0-35"      # split into 0-11 / 12-23 / 24-35
    gpu_devices: ["0","1","2"]  # one GPU per exp

    parallel:
      max_parallel: 3        # how many exps run concurrently

    # true  -> each exp spins up its own container (auto-removed),
    #          named `<container_name|auto>-exp-<i>`.
    # false -> all exps share the playground's single container and
    #          rely on per-thread cwd state to stay isolated.
    fresh_container_per_exp: true
```

Semantics:

* `fresh_container_per_exp: true` is the recommended default. Each exp gets
  an isolated container, a dedicated host workspace at
  `runs/<run_dir>/workspaces/exp_<i>`, and is torn down immediately when the
  exp finishes. These per-exp containers **always** use `--rm`, regardless of
  `auto_remove`.
* `fresh_container_per_exp: false` is cheaper if the image is heavy and the
  exps are short: a single container is reused across all exps, and isolation
  relies on each thread having its own cwd state inside the container.

---

## Reference examples

All five minimal playgrounds ship a ready-to-run `docker.yaml`. Copy the one
closest to your use case:

| Playground | Config | What it demonstrates |
|------------|--------|----------------------|
| `minimal` | [`configs/minimal/docker.yaml`](../configs/minimal/docker.yaml) | Single agent, no extra volumes — the bare minimum |
| `minimal_kaggle` | [`configs/minimal_kaggle/docker.yaml`](../configs/minimal_kaggle/docker.yaml) | Kaggle data mounted at `/workspace/input` via a nested volume |
| `minimal_multi_agent` | [`configs/minimal_multi_agent/docker.yaml`](../configs/minimal_multi_agent/docker.yaml) | Planning + Coding agents sharing one container |
| `minimal_multi_agent_parallel` | [`configs/minimal_multi_agent_parallel/docker.yaml`](../configs/minimal_multi_agent_parallel/docker.yaml) | `fresh_container_per_exp` + per-exp GPU / CPU pinning |
| `minimal_skill_task` | [`configs/minimal_skill_task/docker.yaml`](../configs/minimal_skill_task/docker.yaml) | Mounts `./evomaster` so skill scripts are callable from inside the container |

Run any of them with:

```bash
python run.py --agent <playground_name> \
    --config configs/<playground_name>/docker.yaml \
    --task "<your task>"
```

---

## Troubleshooting

* **`docker: command not found` / daemon unreachable** — install Docker and
  confirm `docker info` works before running EvoMaster. The session fails
  fast with a clear error here.
* **`use_existing_container='...' does not exist`** — the name/id you passed
  isn't known to Docker; `docker ps -a` will show what is.
* **Bind mount created an empty dir instead of mounting your data** —
  relative paths are resolved against the project root (directory containing
  `evomaster/`), not your shell cwd. Use an absolute path if in doubt.
* **`file not found` when a skill or tool calls a local script** — the path
  isn't covered by any of your `volumes`; add a mount so the file is visible
  inside the container.
* **Agent sees the host's total memory** — expected. `memory_limit` is a
  kernel cap; `/proc/meminfo` is not namespaced. Memory pressure *is* enforced
  correctly (the kernel will OOM-kill the container at the limit).
* **Container stayed behind after Ctrl-C** — only happens with
  `auto_remove: false` or when you were attached to an existing container. A
  normal `auto_remove: true` run will pass `--rm` and clean up on exit.
