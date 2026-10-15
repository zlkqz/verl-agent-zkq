# P2O 实施记录

计划入口：[p2o_agent_implementation_plan.md](../p2o_agent_implementation_plan.md)。

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

**状态：**已按反馈同步计划§0.2.1及本记录；环境选择问题已解决，容器内预检待执行。宿主机覆盖问题不能推断为容器内同样存在。下一步核对对应容器状态、仓库/模型/数据/输出挂载、实际Python/依赖及GPU可见性，然后继续TODO-00；验收标准不变。

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

**用户决定：**已发出异步问题，待回复。

**状态：**按用户的暂停要求和计划§0.1/§13.10，保留现场并暂停合同实现；只完成只读定位和记录。未安装依赖、改权限、复制源码、修改挂载或重启容器。ALFWorld容器未安装pytest，后续合同测试可采用标准库unittest，尚未选择或运行测试实现。
