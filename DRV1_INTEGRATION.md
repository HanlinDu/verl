# drv1 / v1 协作与验收规则

最后更新：2026-11-09（Asia/Shanghai）。适用仓库：`HanlinDu/verl`。

## 主分支与初始状态

- **`v1` 是整个 drv1 开发过程的长期主分支，也是 GitHub 默认分支。** 所有研究 PR 的 base 都选 `HanlinDu/verl:v1`。
- 初始运行代码来自 HanlinDu 的 `a91542dc59686d243843040dac4390c29a68c873`。保留其完整祖先历史；“个人基线”指没有纳入该提交之后两位协作者的 11 个提交，并非重写上游历史。
- 本次规则 PR 仅增加协作制度、检验工具和 CI。`verl/` 运行代码保持个人基线；这次规则本身也通过 PR 提交。
- `main` / `upstream/main` 是独立的上游来源，升级上游同样需开 PR 到 `v1`。
- 旧共享分支 `dr-v1/multi-node-direct-reshard` 和备份标签 `archive/dr-v1-20261105` 保留。现有 PR #1、#2 保持开放草稿，是否选择其中代码由 HanlinDu 日后决定。

## 谁提交、谁审批、谁合并

维护者：HanlinDu、Jetteyc、Knight-of-Thunder、LeonardW-sl。四人均使用 PR；HanlinDu 也不直接向 `v1` 推代码。

1. 从最新 `origin/v1` 建立自己的功能分支，或在个人 fork 建分支；一个 PR 对应可独立解释、测试和回退的功能。提交时保留作者及 AI 协助信息。
2. 新建 PR 的 base 选 `HanlinDu/verl:v1`；fork 的 head 选 `个人账号:分支名`。检索本仓库及上游已有 PR，说明差异和依赖。按现有 PR 模板填信息。
3. 有依赖时，在正文列出父 PR 和原始 SHA；所有 PR 的最终目标仍为 `v1`。未满足依赖、未测试或存在冲突时保持 Draft。
4. 作者逐行审查并运行相关测试，记录真实结果；Codex 不代替人签署这项声明。至少一名**其他**有写权限的维护者批准最新修改；HanlinDu 的 PR 由三位协作者之一审查。
5. 协作者的 PR 由 HanlinDu 检查冲突、测试证据并决定是否合并；只有 HanlinDu 发布 `drv1/acceptance` 验收状态。该状态不是批准或合并操作。
6. 使用 GitHub **Create a merge commit**，保留来源与回退边界。合并期间串行操作；先合入一个，再刷新并复验下一个。

GitHub `v1` 分支保护：要求 PR、至少 1 个批准、过期批准失效、最后一次推送由其他人批准、所有讨论解决、分支更新至最新 base，且 `drv1-policy`、`drv1-cpu`、`drv1/acceptance` 全部通过；管理员同样受约束，禁用 force-push 和分支删除。原有上游 CI 的绿灯不能代替这三个门槛。

可审查的保护配置保存在 `scripts/drv1/branch-protection.json`。仓库保留 merge commit，关闭 squash/rebase merge、自动合并及自动删除已合并分支。以后修改保护制度也应先提交可审查的 PR，再由管理员同步 GitHub 设置。

个人仓库的经典保护 API 不支持按用户设置 push 白名单，故 `restrictions=null`；维护者依照本规则由 HanlinDu 决定合入。不要把“CODEOWNERS 有四个人”解释成“必须四人都批准”，它要求其中一名合资格的其他人。

## 验证范围与硬件矩阵

| 改动 | 合入前最低要求 |
| --- | --- |
| 仅说明文档、指定协作工具/CI | CPU、工具负例测试、人工审查；不声明 GPU 已通过 |
| 运行代码、配置、依赖、训练脚本或其他未豁免文件 | CPU + 下表全部四种 GPU profile + 人工核对日志 |

豁免由 `verify.py` 根据完整候选 diff 保守判断，不能通过修改 PR 正文自选。所有 GPU 结果必须对应同一个 base/head/merge tree；任一提交或冲突解决改动都重新准备候选。严格状态检查要求 base 更新后重新验证。

| profile | 集群 | 使用规模 |
| --- | --- | --- |
| `h20-8` | H20 | 单机 8 卡 |
| `h20-16` | H20 | 双机，每机 8 卡 |
| `5090-8` | RTX 5090 | 单机 8 卡 |
| `5090-16` | RTX 5090 | 双机，每机 8 卡 |

两套集群互不连通，**分别运行、搬运证据、集中判定**。从 H20 开始建立基线，再在独立 5090 环境复验。只有 H20 通过时，完整 runtime 验收仍是 pending。

每个 profile 都需要：

- NCCL 实际载荷检查：all-reduce、broadcast、all-gather、环形点对点通信；记录每个 rank 的主机、GPU、CUDA 能力和代码身份。
- `disabled`：关闭动态 resize，完成至少 6 个训练步骤，确认 actor 更新、rollout 同步、有限 loss、正常结束及原有行为。
- `resize`：从已能运行的配置出发，进行至少两次拓扑切换，最后一次后至少继续 2 步；核对同步、权重一致性、无死锁/OOM、正常结束。跨节点场景必须实际使用两个节点。
- `checkpoint-resume`：保存真实 checkpoint，在新的训练进程恢复，并继续至少 2 步；核对 step、模型状态和声明的 optimizer/scheduler/RNG 语义。weights-only 恢复应清楚标注，不得声明等价连续训练。
- 记录模型/数据 revision、有效配置、镜像或环境版本、驱动、CUDA/NCCL、随机种子、完整命令和日志。比较固定小样本基线的 reward/loss、耗时、峰值显存；性能变化需解释。

当前个人基线**尚未在这两套集群认证为可运行**。新制度不会修复基线代码，也不会凭 CPU 检查保证训练正确。首次迁移如发现问题，记录失败并新建修复 PR；保留 pending，直至真实验收通过。

## 容器 SSH 环境的首次准备

仅在你有权限的两个 GPU 容器内工作。两节点容器应有独立 hostname、各可见 8 卡，并能在本集群内连通 rendezvous、NCCL、Ray 所需端口；`localhost` 或单个 SSH 端口转发不足以证明多节点训练网络正常。遇到网络隔离时先记录 blocker，再由集群维护方配置容器网络；无需预先登录宿主机。

1. 两个容器检出相同候选 SHA，准备相同模型/数据 revision。共享 checkpoint 路径，或按现有训练方案明确每节点分片的可见性。
2. 在现有已适配 H20 / 5090 的镜像中，用 `uv` 管理 Python 环境。**CPU CI 的 torch wheel 只用于独立 CPU 环境**；GPU 环境沿用已验证 CUDA/PyTorch/vLLM 组合，记录 `nvidia-smi`、`uv pip freeze`、`torch.cuda.get_device_capability()` 和 `torch.cuda.get_arch_list()`。5090 单独确认架构与算子支持。
3. 运行 GPU probe，再配置训练命令。probe 通过仅说明通信基础可用，不表示训练或 resize 已通过。
4. 训练需要 Ray 时，在本集群容器内按既有部署方案启动 head/worker，核对 `ray status` 和实际资源。设置 `ray_kwargs.ray_init.address=auto` 接入它；不要在共享集群执行全局 `ray stop`。超时后由本次任务的操作员清理对应远端任务。

## Codex / 人工操作入口

使用最新可信 `v1` 的检验工具；先审查候选对工具/CI 的修改。特权凭据不交给候选训练进程。`pull_request` 在 GitHub 托管 CPU runner 上运行、只有只读权限；`pull_request_target` 仅重置 metadata 状态，不 checkout 或执行 PR 代码。GPU 不自动接收未审查的公网 PR。

### A. 准备独立候选，不改动主工作区

```bash
# 在可信控制 checkout 内；第一次引入规则时使用本规则 PR 的分支。
export DRV1_TOOLS="$PWD/scripts/drv1"
python "$DRV1_TOOLS/verify.py" prepare --pr 123 --output /tmp/drv1-pr123
cd /tmp/drv1-pr123/checkout
python "$DRV1_TOOLS/verify.py" cpu --output /tmp/drv1-pr123-cpu
```

`prepare` 自动 fetch GitHub 的 PR merge ref，检查其两个 parent 正是最新 `v1` 和 PR head；fork PR 同样适用。输出 `candidate.json`。有冲突时在作者/协商后的分支解决并推送该分支，待 merge ref 刷新后重新运行；不要为了消除冲突改写共享旧分支或直接推 `v1`。

独立 CPU 环境可用 Python 3.12、`uv venv`，先安装 CPU `torch==2.8.0`，再 `uv pip install -r scripts/drv1/requirements-cpu.txt`、`uv pip install --no-deps -e .`。精确命令亦见 `.github/workflows/drv1.yml`。输出目录必须在 checkout 外且尚不存在，避免混入旧结果。唯一预先允许的 CPU skip 是无加速器时的 pin-memory 能力测试，其余 skip、失败、零测试、被改过的日志均阻止验收。

个人基线已有部分文件缺少 license header，故全仓 pre-commit 的 license hook 有遗留失败。本次不修改这些运行文件；`drv1-policy` 对新增/修改的 Python 文件检查 license。PR 应分别记录全仓结果、已有问题和变更范围内的检查，不能把豁免写成全部通过。

### B. 在各 GPU 容器运行 probe

两节点先分别检出 `candidate.json` 的 `merge_sha`，并保持干净。以下在**同一集群的两个容器同时执行**，分别设置 `NODE_RANK=0/1`，`RDZV_ENDPOINT` 为可互访的容器地址和端口；同一次运行共用新的 `RDZV_ID`。H20 和 5090 使用不同运行编号，不互相 rendezvous。

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --rdzv-backend=c10d --rdzv-endpoint="$RDZV_ENDPOINT" --rdzv-id="$RDZV_ID" \
  "$DRV1_TOOLS/nccl_probe.py" --profile h20-16 --output /tmp/h20-16-probe.json
# 单机：在一个容器内执行
torchrun --standalone --nproc-per-node=8 \
  "$DRV1_TOOLS/nccl_probe.py" --profile h20-8 --output /tmp/h20-8-probe.json
```

5090 改用 `5090-16` / `5090-8`。rank 0 写 JSON，另存 torchrun stdout/stderr。准备两节点时核对有效配置中的 nnodes、每节点 GPU 和角色分配；旧 `run_dynamc_resize.sh` 固定 `nnodes=1`，须提供真实双节点训练配置。

### C. 训练与人工观察

复制 `plan.example.json` 到仓库外，把三个 argv 数组换为本集群真实训练命令（通常为你自己的三个 shell adapter）；模型/数据路径、checkpoint 和日志都使用本次运行的新目录。单机/双机分别一份计划。示例中占位路径有意无法执行，未配置时不能得到通过结果。

三个 adapter 从候选 checkout 启动训练：使用现有 `python -m verl.experimental.one_step_off_policy.main_ppo` 入口，传入模型、parquet、实际 trainer/rollout nnodes、GPU 数、步数及各 case 配置。`disabled` 明确设置 `trainer.dynamic_resize.enable=false`；`resize` 使用经过人工确认的两次切换 schedule；`checkpoint-resume` 应包含保存和新进程恢复两个阶段，并传播任一阶段的非零返回值。仅在 driver 容器启动训练，Ray workers 使用另一节点的同一候选代码。

先运行下列命令收集日志，阅读日志后复制 `observations.example.json` 到私有目录，填写操作者、证据位置和真实观察，再用新输出目录重跑或将已审查的 observations 填入该次 `report.json`。人工观察字段明确属于操作者声明，工具校验日志 hash 并不构成对训练正确性的独立证明。

```bash
python "$DRV1_TOOLS/verify.py" gpu --candidate /tmp/drv1-pr123/candidate.json \
  --profile h20-16 --probe /tmp/h20-16-probe.json --plan /private/h20-16-plan.json \
  --observations /private/h20-16-observations.json --output /tmp/pr123-h20-16
```

每个 case 默认超时 1800 秒，可显式调整 `--timeout`；超时/失败会留日志并返回非零。工具终止本地子进程组，不自动终止其他容器或共享 Ray 服务。`gpu` 的零返回值只说明三个命令退出成功；最后的 `gate` 还要求完整拓扑、源代码身份、日志与人工观察。

### D. 集中判定、审批、合并

从另一集群拷贝完整 evidence 目录（包括日志、JSON、JUnit）；保持文件内容不变。在**原候选 checkout** 中运行：

```bash
python "$DRV1_TOOLS/verify.py" gate --candidate /tmp/drv1-pr123/candidate.json \
  --cpu /tmp/drv1-pr123-cpu/report.json \
  --gpu /evidence/h20-8/report.json /evidence/h20-16/report.json \
        /evidence/5090-8/report.json /evidence/5090-16/report.json
```

仅文档/治理改动可省略 `--gpu`。将完整证据保存在团队可访问且保留期足够的存储，将 URL 加入 PR；GPU 目录不提交 Git。HanlinDu 审查证据后，在同一命令后加 `--publish --evidence-url https://...` 发布 `drv1/acceptance`。使用本机 Git 认证或本机环境凭据；凭据不写入 JSON、日志、PR 正文或聊天。发布前会再次核对远端 base/head；没有凭据时只做本地判定。

GitHub 托管 CPU artifact 默认保留 30 天；长期实验需要另存。状态检查、JSON 和操作者声明均不是防伪证明，具有写权限的人仍有发布 commit status 的能力；信任边界是经审查的工具、HanlinDu 对证据的核验，以及另一位维护者对最新修改的 GitHub 批准。

验收通过后，作者完成逐行审查和相关测试声明，其他维护者批准，HanlinDu 再决定合并。工具从不自动批准、合并、推送或删除分支。

### E. 合入后的完整性与回退

```bash
git fetch origin v1
git switch v1
git pull --ff-only origin v1
python "$DRV1_TOOLS/verify.py" post-merge --candidate /evidence/pr123/candidate.json
```

核对实际 merge commit 的父提交与 tree 等于候选；`push:v1` 会再跑 CPU。实际合入 SHA 再做一次 H20 的短训练 smoke；涉及硬件兼容/通信的改动同时复核相应 5090 场景。只有这些完成后，才记录该 SHA 为 known-good，并可由维护者创建带说明的 `drv1-verified/...` 标签。已有 archive 标签表示备份，不代表通过 GPU 验收。

如合入后失败：暂停下一次合入，保留日志，在从最新 `v1` 建立的回退分支执行 `git revert -m 1 <merge_sha>`，通过回退 PR 再次执行必要检查。禁止 reset/force-push 主分支。单次 CPU/GPU 测试无法保证未来所有工作负载；新增功能应补充对应回归用例。

## 给下一次 Codex 的任务模板

> 阅读 AGENTS.md 与 DRV1_INTEGRATION.md。审计 PR #N 的重复实现、依赖、完整 diff、冲突与测试。使用可信 v1 工具准备最新 merge candidate；在当前容器可运行的范围内执行 CPU/GPU 验证并保留原始日志。缺少另一集群、模型/数据、人工观察或人工审查时，明确列为 pending，不发布通过状态。不得擅自合并 PR、修改共享旧分支或向 main 集成。先交付可审查结果，再由人决定合入。

## GitHub 行为依据

- [受保护分支与最新批准/状态检查](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [Branch protection API：管理员约束及个人仓库限制](https://docs.github.com/en/rest/branches/branch-protection)
- [pull_request / pull_request_target 的执行上下文](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)
