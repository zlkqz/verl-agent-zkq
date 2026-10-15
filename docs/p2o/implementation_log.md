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

**候选处理：**

1. 推荐：对P2O实现与测试命令使用 `PYTHONNOUSERSITE=1 /ceph_home/zhangkaiqi2024/miniconda3/envs/verl/bin/python`，保留已有vLLM 0.8.5.post1；不修改用户目录包、驱动或原baseline脚本。
2. 若用户指定较新后端：同样禁用user-site，使用 `verl_newer_vllm`（vLLM 0.11.0），兼容性需后续对应模块验证。
3. 用户提供另外的训练容器/解释器，再对指定环境预检。

**用户决定：**待回复。诊断中的临时环境变量不代表已决定后续执行方案。

**停点：**依用户暂停要求及计划§0.1/§13.10，仅记录和只读定位；未安装/卸载依赖、未改驱动、未实施合同代码。回复后先同步本记录与计划，再继续TODO-00。验收标准不变。
