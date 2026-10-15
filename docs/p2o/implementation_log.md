# P2O 实施记录

计划入口：[p2o_agent_implementation_plan.md](../p2o_agent_implementation_plan.md)。

最新状态：TODO-00已实现并通过两容器验收；旧挂载问题已按用户授权通过重建和初始化解除。当前等待用户反馈后进入TODO-01。先读[TODO-00实现说明](modules/TODO-00.md)，复现证据见[验收记录](test_records/TODO-00.md)，环境见[环境文档](environment.md)。主计划已精简为导航，以下保留定位及决策历史。

## 2026-10-15：TODO-00预检停点

用户已授权开始实现，并允许使用本机8张GPU调试。每完成一个大TODO或遇到需要讨论的问题，暂停等待反馈；正式长训练不在授权范围内。

本轮完成仓库、模型/数据路径、GPU、Python依赖预检；读取两个环境worker/manager的reset、step、reward及success输出，复核Math判难/保存/恢复代码。未创建业务代码、未运行环境或加载模型。TODO-00未实现、未验收，未进入TODO-01。

- 起始代码版本：目标仓库 `a4f930c`，只读参考仓库 `92997e2`。
- 起始工作区：仅未跟踪的 `docs/p2o_agent_implementation_plan.md`；原始计划已通过 `0d4999d` 保存。
- 预检证据、完整命令和依赖对比：[TODO-00-preflight.md](test_records/TODO-00-preflight.md)。

### P2O-ENV-001：默认user-site覆盖导致CUDA不可用

**预期：**找到后续实现和有界GPU功能验证可使用的Python环境，不修改原GRPO路径或共享系统环境。

**实际：**shell无默认 `python`；系统 `python3` 缺少训练依赖。已有conda `verl` 和 `verl_newer_vllm` 默认导入用户目录的PyTorch 2.14.0，提示驱动过旧、CUDA不可用。仅在诊断命令中设置 `PYTHONNOUSERSITE=1` 后，两环境分别恢复原有PyTorch 2.6.0/cu124与2.8.0/cu128，均可识别8卡。

**影响：**TODO-00及其后模块需要固定测试解释器；后续GPU验收不能使用当前默认的混合依赖。禁用user-site也改变可见包集，不能把默认混合环境的版本记录当作实际运行版本。尚未验证全部训练/环境依赖导入或执行GPU计算。

**此前候选处理（历史记录，已由用户指定容器替代）：**

1. 当时建议：对P2O实现与测试命令使用 `PYTHONNOUSERSITE=1 /ceph_home/zhangkaiqi2024/miniconda3/envs/verl/bin/python`，保留已有vLLM 0.8.5.post1；不修改用户目录包、驱动或原baseline脚本。
2. 若用户指定较新后端：同样禁用user-site，使用 `verl_newer_vllm`（vLLM 0.11.0），兼容性需后续对应模块验证。
3. 用户提供另外的训练容器/解释器，再对指定环境预检。

**用户决定（2026-10-15）：**ALFWorld使用容器 `verl_zhangkaiqi2024_alfworld`，WebShop使用容器 `verl_zhangkaiqi2024_webshop`；两个容器都在当前这台机器上。环境选择已明确，宿主机conda候选方案不再作为默认执行方案。

**状态：**环境选择问题已解决。后续容器预检发现独立挂载问题P2O-ENV-002，现已按用户授权重建并初始化通过；宿主机覆盖问题没有在新容器中复现，验收标准不变。

## 2026-10-15：补充用户指定的本机容器

本轮按用户要求只更新Markdown，登记任务与容器对应关系，并替换“等待选择宿主机Python环境”的过时状态。未进入、启动、重启或修改容器，未修改业务代码，未继续TODO-00实现。容器名称来自用户确认，容器内路径、包版本与功能测试结果仍待实测。

## 2026-10-15 12:50–13:00：继续TODO-00，容器预检

用户要求继续。两指定容器均运行中，Docker daemon与宿主机名称一致（`CIP-GPUSERVER-144`）。在容器 `/tmp` 下检查Python、依赖metadata和CUDA设备发现；两者均CUDA可用、可见8设备。实际版本见[预检记录](test_records/TODO-00-preflight.md)。未运行GPU计算、真实环境或模型加载。

### P2O-ENV-002：两容器无法访问绑定的 /ceph_home

**预期：**容器通过已有读写bind mount访问宿主机目标仓库及模型、数据。

**实际证据：**两容器的 `docker inspect` 都显示 `/ceph_home:/ceph_home`、`RW=true`；容器默认root运行 `ls -ld /ceph_home` 返回 `Permission denied`。用 `--user 1083:1083`（宿主机仓库owner）在 `/tmp` 执行 `cd /ceph_home/zhangkaiqi2024/verl-agent` 仍被拒绝。未启用UID映射，`/proc/self/uid_map` 为 `0 0 4294967295`。宿主机正常读取仓库。原因未确认，不能将其直接归因为root squash或UID不匹配。

**影响：**容器直接运行当前工作区与读取模型/数据暂不可用；TODO-00不需要加载模型，可选择暂存必要代码测试，但此执行路径尚未获确认。TODO-01及后续真实环境/GPU验收需要可用的仓库/模型/数据路径。

**选项：**

1. 推荐：TODO-00代码仍写宿主机目标仓库，只将合同模块、测试及样例临时复制到两容器 `/tmp`，不修改挂载/权限/原GRPO路径；按暂存源码hash核对测试版本。TODO-00仅做合同验收，真实环境/模型测试前再解决挂载。
2. 先定位并处理容器挂载访问问题，再按原路径继续；任何重启、重建或权限修改的具体操作需要另行明确，不根据猜测执行。

**用户决定：**用户明确授权删除这两个旧容器，参照 `/ceph_home/zhangkaiqi2024/start_docker.sh` 重建；ALFWorld使用 `verlai/verl:vllm011.latest`，WebShop使用 `whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3`。重建后分别执行 `my_scripts/init_alfworld.sh` 和 `my_scripts/init_webshop.sh`，验证环境后继续TODO-00。无需再次确认删除这两个容器。

**状态：**已按授权完成重建及初始化，仓库/模型config/数据挂载可读，问题解除。未采用临时复制源码方案。重建前确认两容器内仅有sleep进程，无训练进程。工作区另有用户的 `my_scripts/run_webshop.sh` 注释修改，保留且不纳入本次提交。合同测试使用标准库unittest，无需安装pytest。

## 2026-10-15约13:00–13:15：环境恢复与TODO-00完成

复用用户启动脚本的docker run参数、指定镜像及原任务容器名重建两容器，分别执行对应init脚本，最终exit code均为0。WebShop镜像附加pip源反复TLS重试，终止本次安装进程后仅以临时环境变量忽略pip配置重跑；保留初始化脚本及包版本约束。两容器挂载恢复，关键任务依赖/训练入口导入、GPU 0–7逐卡基础CUDA运算通过；未加载模型、运行真实环境episode或启动训练。

新增文件：

- `verl/trainer/p2o/{__init__,contracts}.py`：schema v1记录、异常、能力声明、任务/实例/episode/turn身份、subset请求/结果、终止结果、上下文/原始生成token/评分、评价面板与请求结果、历史证据和P2OState/组件state边界。
- `agent_system/p2o/{__init__,base}.py`：Stateful、EnvAdapter、ContextBuilder、PolicyGenerator/Scorer与EpisodeEvaluator Protocol。
- `tests/trainer/p2o/{__init__,test_contracts,export_examples}.py` 及 `fixtures/traces.json`：共用fake adapter/evaluator、两环境模拟trace、错误/序列化/历史/对齐验收与可复现样例导出。
- `docs/p2o/test_records/TODO-00.md`、`TODO-00-examples.json`：验收与环境恢复记录、完整模拟输入/输出。

最终测试：初始化完成后，两容器各16项unittest均通过；ALFWorld为0.059s、WebShop为0.071s（测试体耗时）。两容器导出的JSON样例逐字节一致。源码指纹：contracts.py为 `f4f3d6c9f3130335d6bb1f19b59b183d4ba163a6f7993d5ef33c91c21c22209f`，base.py为 `08caba7c738df571a062332a1a57793b2f0cee426c70f95f86e16ea325d4bbce`，test_contracts.py为 `d4cf639940fa97bb411190fec856d669b2fa2de06abddca1e264ff09c94b8ab0`。完整命令、依赖、资源、输出和原始日志位置见[验收记录](test_records/TODO-00.md)。

计划正文已更新为当前接口与已验收状态，TODO-00勾选。原入口/trainer/环境/collector/reward/actor及初始化脚本未改。保留待对应模块决定的随机化与退化模板采样边界，不先行实现。当前按逐大TODO停点要求暂停；下一步为TODO-01，等待用户反馈再开始。

## 2026-10-15：补充TODO-00说明并拆分文档（本轮仅文档）

用户要求每次停点不仅写Markdown，还要在对话中解释当前进度、做了什么、核心代码路径与逻辑、测试结果及问题；同时拆分过长的主计划。本轮没有进入TODO-01，没有修改业务代码、执行新环境操作或重新运行合同测试。

- 主计划从1049行精简为52行，只维护当前状态、导航、约束与执行纪律。
- 原§0/§1–4/§5–6/§7–8/§9–10/§11–14分别移至 `environment.md`、`method.md`、`interfaces_and_search.md`、`runtime_and_recovery.md`、`todo_list.md`、`decisions.md`；保留原编号与全部后续TODO验收门槛，补显式锚点和导航。修正少量“尚无模块/未做GPU计算”等过时状态文字。
- 新增 `modules/TODO-00.md`：按任务/实例→subset请求→轨迹结果→token评分对齐→历史证据→序列化说明核心代码；区分可执行校验、Protocol声明、模拟测试与后续实机工作。
- 环境恢复命令/镜像/版本从TODO-00验收页移至环境文档，验收页保留测试命令、结果、样例与证据链接，减少重复。
- 将“Markdown与对话同时汇报”的固定内容写入主入口、§8.4记录规范与§12停点模板；后续模块分别提供实现说明及验收记录。

检查结果：11份Markdown的本地链接、显式锚点和代码围栏通过；原64个编号标题均保留，15项TODO仅00勾选，后续验收条目完整；`git diff --check`通过。TODO-00测试结果仍引用实现提交 `7c297e8` 时两容器各16项通过的记录；本轮没有新增代码测试结果。用户已有 `my_scripts/run_webshop.sh` 注释修改保留且不纳入文档提交。

当前仍在TODO-00完成停点，待本轮向用户解释实现与证据后，等待反馈再开始TODO-01。
