# 在 Docker Session 中运行 Playground

EvoMaster 的智能体通过 **Session** 驱动各种工具（bash、editor、skill 脚本…）。
默认 Session 为 `local`，也就是所有命令直接在宿主机上执行。如果改为 `docker`
Session，所有工具调用都会进入一个隔离的容器运行，带来以下好处：

* 干净、可复现的运行环境，不会污染宿主机（非常适合运行不受信任的智能体输出、
  长跑型科学计算任务，或 CI 流水线）。
* 每次运行可独立配置资源上限（内存、CPU 核集、GPU、网络模式等）。
* 运行产物所在的工作区自动挂载到宿主机的
  `runs/<run_dir>/workspace/`，便于在容器外用与智能体完全一致的视角查看结果。

实现这一切**无需修改任何智能体或工具代码**，只需要改一段 YAML。

> 以下五个参考 playground 已经附带开箱即用的 `docker.yaml`，推荐直接复制一个作为起点：
> `minimal`、`minimal_kaggle`、`minimal_multi_agent`、
> `minimal_multi_agent_parallel`、`minimal_skill_task`。

---

## 三步快速上手

1. 确认本机已安装 Docker 且守护进程可用（`docker info`）。
2. 在 playground 的配置文件中加入：
   ```yaml
   session:
     type: "docker"
     docker:
       image: "python:3.11-slim"        # 任意自带 bash 的镜像
       working_dir: "/workspace"
       memory_limit: "8g"
       cpu_limit: 4.0
       auto_remove: true
   ```
3. 照常运行：
   ```bash
   python run.py --agent minimal --config configs/minimal/docker.yaml \
       --task "你的任务描述"
   ```

就这么简单。EvoMaster 在首次使用时会拉取镜像（默认策略 `missing`），启动容器、
把本次运行的工作区挂载到 `working_dir`，运行结束后自动清理。即使 Ctrl-C 中断，
只要 `auto_remove: true`，容器也会被删除。

---

## 工作原理

当 `session.type == "docker"` 时，playground 会构造一个
[`DockerSession`](../../evomaster/agent/session/docker.py)，底层由
[`DockerEnv`](../../evomaster/env/docker.py) 支持。`open()` 时的流程如下：

1. 先调用 `docker version` 快速检测守护进程是否可达。
2. 在宿主机上为嵌套在 `working_dir` 内的 bind mount（例如 `/workspace/input`）
   创建同名符号链接，使宿主机上的
   `runs/<run_dir>/workspace/input` 也能看到同样的内容。
3. 按 `pull_image` 策略拉取镜像。
4. 用 `tail -f /dev/null` 作为 PID 1 启动容器（detached），保证容器在多次
   工具调用之间一直存活。
5. 在容器内初始化一个**按线程隔离**的状态目录
   （`/tmp/evomaster_session_<uid>/thread_<tid>`），用来在多次 `exec_bash`
   之间保存当前工作目录。

每次 `agent.execute_bash("…")` 对应**一次** `docker exec …`。包装脚本会先
`cd` 到上一次保存的 cwd，执行用户命令，把新的 `pwd` 写回状态文件，最后输出
一个标记行用于解析真实的退出码。**有几个需要了解的语义：**

* 同一线程中，前一次调用的 `cd <dir>` 对下一次调用可见。
* **导出的环境变量、后台任务不会跨调用保留**——如果需要在同一步中生效，请用
  `&&` 或 `;` 串联。
* 如果多个 Exp 在同一个容器中并行执行，每个 Exp（线程）会有自己独立的 cwd
  状态，不会互相干扰。
* 不支持 `is_input`（交互式 stdin），会返回一条清晰的错误信息；请改用
  heredoc 或串联命令。

文件 I/O 做了优化：当路径位于某条 bind mount 下时，`upload` / `download` /
`read_file` / `write_file` 会直接在宿主机侧读写；只有命中不在挂载范围的路径
才会退化到 `docker cp`。

---

## 配置参考

在 playground 的配置中加入如下区块。除特殊说明外，字段基本与 `docker run`
参数一一对应。

```yaml
session:
  type: "docker"

  docker:
    # --- 镜像与容器身份 ---
    image: "python:3.11-slim"
    container_name: null          # null 表示每次运行自动生成
    use_existing_container: null  # 附着到你自己已经管理的容器

    # --- 路径 ---
    working_dir: "/workspace"     # 本次运行的工作区会被自动挂载到此处

    # --- 资源 ---
    memory_limit: "8g"            # --memory（同时设置 --memory-swap 防止落盘交换）
    cpu_limit: 4.0                # --cpus  (CPU *时间* 配额)
    cpu_devices: null             # --cpuset-cpus: "0-15" | "0,2,4" | [0,1,2] | null
    gpu_devices: null             # "all" | "0" | "0,1" | ["0","1"] | null
    network_mode: "bridge"        # bridge | host | none | <自定义>

    # --- 挂载 / 环境变量 ---
    volumes: {}                   # {host_path: container_path}
    env_vars: {}                  # {KEY: VALUE}

    # --- 生命周期 ---
    auto_remove: true             # true -> docker run --rm；false -> 不自动删除
    pull_image: "missing"         # missing | always | never
    timeout: 300                  # 单条命令默认超时（秒）
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `image` | 必须包含 `bash`。`python:3.11-slim` 即可开箱即用；ML 场景推荐使用 CUDA 镜像（可参考 `evomaster/env/docker/base.dockerfile`）。 |
| `container_name` | 固定容器名，便于跨运行复用。需配合 `auto_remove: false`。 |
| `use_existing_container` | 通过名称或 id 附着到已存在且正在运行的容器。**EvoMaster 永远不会停止或删除附着的容器**。 |
| `working_dir` | 容器内工作目录，**同时**也是工作区挂载点。本次运行的工作区（`runs/<run_dir>/workspace/`）会自动挂到此处，无需在 `volumes` 中手动声明。 |
| `memory_limit` | 内核 cgroup 上限。容器内 `htop` / `free` 显示的仍是宿主机的总内存（因为 `/proc/meminfo` 不受命名空间隔离）——这是正常现象，实际内存限制依然由内核强制执行。 |
| `cpu_limit` | 只限制 CPU *时间* 配额，不限制可见核数。若要同时限制可见核，请使用 `cpu_devices`。 |
| `cpu_devices` | CPU 亲和性（`--cpuset-cpus`）。支持 `"0-15"`、`"0,2,4"`、Python 列表、`null`。 |
| `gpu_devices` | `"all"`、单个 id、逗号串或列表。`null` / `"none"` 表示不分配 GPU。 |
| `volumes` | `{host_path: container_path}`。**相对路径**（如 `./assets`）相对于**项目根目录**（包含 `evomaster/` 的目录）解析，而不是 shell 的当前目录。 |
| `env_vars` | 额外环境变量。若你的配置加载器支持 `${VAR}` 替换，可以复用它。 |
| `auto_remove` | `true` 会传入 `--rm`，session 关闭时销毁容器；`false` 会保留容器以便下次 `open()` 复用。 |
| `pull_image` | `missing`（默认，只在本地无镜像时拉取）、`always`（每次运行都拉取）、`never`（要求镜像已存在，否则报错）。 |
| `timeout` | 每条命令的默认超时，单次工具调用可以覆盖。 |

### 特殊行为：嵌套挂载的宿主机符号链接

如果一条 bind mount 的目标**嵌套在 `working_dir` 之内**（例如
`./data/public → /workspace/input`），EvoMaster 会在宿主机额外创建一个同名
符号链接：

```
runs/<run_dir>/workspace/input -> <./data/public 的绝对路径>
```

这样即使在容器外看，`runs/<run_dir>/workspace/` 也与容器内视角一致。否则
Docker 的嵌套挂载只在容器内叠加，宿主机上的 `/workspace/input` 会是空目录。
这个行为**完全自动**，无需额外配置。

### 工具与 Skill 的路径转换

一些工具（例如通过 `execute_bash` 调用磁盘上的 skill 脚本）接受的是**宿主机
路径**。这类调用会先经过 `DockerSession.to_session_path()`：session 会遍历
`volumes`，把命中挂载前缀的宿主路径改写为容器内对应路径。这也是为什么
`minimal_skill_task` 的 Docker 配置会把 `./evomaster` 挂到
`/workspace/evomaster`——`use_skill` 工具需要在容器内也能读到 skill 脚本。
如果你在使用 `use_skill` 或自定义工具时看到 `file not found`，请先检查涉及
的文件是否位于任一 `volumes` 项下。

---

## 并行模式（`minimal_multi_agent_parallel`）

对于多实验并行运行的 playground，除常规字段外，再加入两个扩展字段：

```yaml
session:
  type: "docker"
  docker:
    # ...与上文相同的常规字段...

    # 所有并行 Exp 共享的资源总量；会按并行度拆分到每个容器
    memory_limit: "48g"      # 48g / 3 exps = 每容器 16g
    cpu_limit: 36.0          # 36  / 3 exps = 每容器 12
    cpu_devices: "0-35"      # 分片为 0-11 / 12-23 / 24-35
    gpu_devices: ["0","1","2"]  # 每个 Exp 分配一张 GPU

    parallel:
      max_parallel: 3        # 同时运行的 Exp 数

    # true  -> 每个 Exp 启动独立容器（自动销毁），名为 `<container_name|auto>-exp-<i>`
    # false -> 所有 Exp 共用 playground 的同一个容器，靠每线程的 cwd 状态互不干扰
    fresh_container_per_exp: true
```

语义：

* `fresh_container_per_exp: true`（推荐默认）——每个 Exp 有独立容器，专属的
  宿主机工作区位于 `runs/<run_dir>/workspaces/exp_<i>`，Exp 结束即刻销毁。
  **这些 per-exp 容器永远使用 `--rm`**，与外层 `auto_remove` 无关。
* `fresh_container_per_exp: false`——适合镜像较大而 Exp 较短的场景：所有 Exp
  共用同一个容器，隔离依赖容器内的每线程独立 cwd 状态。

---

## 参考示例

这五个 minimal 系列 playground 都附带了现成的 `docker.yaml`，选一个最接近你
场景的复制改写即可：

| Playground | 配置文件 | 演示要点 |
|------------|---------|----------|
| `minimal` | [`configs/minimal/docker.yaml`](../../configs/minimal/docker.yaml) | 单智能体、最小可运行样例 |
| `minimal_kaggle` | [`configs/minimal_kaggle/docker.yaml`](../../configs/minimal_kaggle/docker.yaml) | 将 Kaggle 数据通过嵌套挂载放到 `/workspace/input` |
| `minimal_multi_agent` | [`configs/minimal_multi_agent/docker.yaml`](../../configs/minimal_multi_agent/docker.yaml) | Planning + Coding 两个智能体共享一个容器 |
| `minimal_multi_agent_parallel` | [`configs/minimal_multi_agent_parallel/docker.yaml`](../../configs/minimal_multi_agent_parallel/docker.yaml) | `fresh_container_per_exp` + 每 Exp 独立 GPU / CPU 绑核 |
| `minimal_skill_task` | [`configs/minimal_skill_task/docker.yaml`](../../configs/minimal_skill_task/docker.yaml) | 挂载 `./evomaster`，让 skill 脚本可在容器内被调用 |

运行方式：

```bash
python run.py --agent <playground_name> \
    --config configs/<playground_name>/docker.yaml \
    --task "<你的任务>"
```

---

## 常见问题排查

* **`docker: command not found` 或守护进程不可达** —— 请先安装 Docker 并确认
  `docker info` 可用；EvoMaster 在 `open()` 时会快速报错。
* **`use_existing_container='...' does not exist`** —— 传入的名称或 id 在
  Docker 中不存在；`docker ps -a` 可以列出所有容器。
* **挂载生成了空目录** —— 相对路径是相对于项目根目录解析的，不是 shell 的
  当前目录；若不确定请直接使用绝对路径。
* **Skill 或工具报 `file not found`** —— 对应文件没有被任何 `volumes` 覆盖；
  加一条挂载把它暴露到容器内即可。
* **容器内看到的是宿主机总内存** —— 这是预期的。`memory_limit` 是内核层面的
  限制，`/proc/meminfo` 不受命名空间隔离。当实际占用达到上限时，内核会
  OOM-kill 容器。
* **Ctrl-C 之后容器没被销毁** —— 只会发生在 `auto_remove: false` 或附着到
  已有容器的场景。正常 `auto_remove: true` 运行会带上 `--rm`，退出时自动清理。
