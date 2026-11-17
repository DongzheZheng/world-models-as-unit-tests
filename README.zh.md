# World Models as Unit Tests

**[English README](README.md)**
&ensp;|&ensp;
**[Read the full write-up →](https://dongzhezheng.github.io/world-models-as-unit-tests/)**

Code as World Models：智能体主动观察任务环境，把世界模型写成代码，并用它帮助编写和验证策略，包括 Code as Policies 写出的程序

Denzel Zheng · 普林斯顿大学机械与航空航天工程系 (MAE)

## 核心发现

前沿语言模型已开始编写机器人控制器。在 [RoboDojo Sim](https://www.anthropic.com/claude-fable-and-mythos-5-2) 上，Claude Fable 5.2 得分 64.9%，GPT-6 Astra 29.8%。在 Code as Policies (CaP) 中，LLM 把控制器写成代码，在真实系统上运行，然后修改（[Liang et al., 2022](https://arxiv.org/abs/2209.07753)）。ASPIRE（[Lu et al., 2026](https://arxiv.org/abs/2607.00272)）将此扩展为自主技能发现。模型改程序越频繁，真实世界评估越成为瓶颈。

<details>
<summary>其他 RoboDojo 分数和 Chen et al. (2026) 结果</summary>

Claude Opus 5.5 在 RoboDojo Sim 上得分 39.4%。[Chen et al. (2026)](https://arxiv.org/abs/2609.38537) 在六个领域评估 GPT-6 Astra：RxR 指令跟随 92%，HM3D 物体搜索 82%，RoboDojo 子集结合 π0.5 为 48%，运动控制无成功，每次模型调用约 40 s。Anthropic 在同一 harness 上运行了所有 RoboDojo 模型。
</details>

真实世界评估是瓶颈：agent 无法分辨自己的策略代码的好坏。LLM 几秒钟就能写出新控制器，在真实系统上评估却需要数分钟、硬件和风险。操控类策略的真实世界评估难以规模化、难以重复（[Li et al., 2024](https://arxiv.org/abs/2405.05941)；[Kress-Gazit et al., 2024](https://arxiv.org/abs/2409.09491)）。五次试验全部成功，在 95% 置信度下只能说明真实成功率高于 55%（Clopper–Pearson 界）。LLM 产生新版本的速度远快于机器人能测的速度，五次试验分散在前后版本上，每个版本大约只被测了一次。一次成功在同等置信度下只能说明成功率高于 5%。LLM 在代码和数学上变得好用，是因为输出可以被自动验证（单元测试、证明检查器）。机器人控制代码缺少这样的验证器。

我们探索的问题是：当通用编程智能体面对一个未知的物理系统、手里没有现成的测试环境时，能否只用少量真实试验，在自己的 harness 中搭出测试？这里的验证指部署前在世界模型里运行程序，对其成功率做出的经验估计。

我们在八个任务上比较两种智能体。CaP 智能体在每次真实试验后修改控制程序。世界模型智能体先写一个世界模型，一个几百行的 Python 文件，模拟这个任务，然后在其中测试程序再部署。八个任务按测试帮助从小到大排列：

1. 无人机沿两条固定路线穿越峡谷，伴有受损旋翼和隐藏横风
2. 线缆布线，伴有隐藏刚度、摩擦和实时力读数
3. Doom 在随机敌火下生存
4. 先推 T 形方块、再推未见过的 L 形方块，各有秘密偏移的质心和随机初始位姿
5. 同一无人机峡谷，每个 episode 加入 7–8 架随机其他飞行器
6. 绳索绕柱抛入目标：可变形、一次抛掷、对细微变化极敏感

在无人机任务的一个变体中，每个 episode 有 7–8 架其他飞行器沿随机路径穿过峡谷。世界模型智能体用第一次真实试验悬停 19.4 秒，全程只记录其他飞行器的运动。从这份记录中，它写入了四种飞行器行为到世界模型，在模型中测试控制器，真实成功率从 50% 升到 97%。

世界模型扮演的角色类似于单元测试对普通代码的作用：每一个候选程序在部署之前先在世界模型中运行。写成代码的世界模型是在智能体 harness 中提供此类测试的一种方式（[Weng, 2026](https://lilianweng.github.io/posts/2026-07-04-harness/)）。多个团队已探索将世界模型写成代码（[Dainese et al., 2024](https://arxiv.org/abs/2405.15383)；[Tang, Key and Ellis, 2024](https://arxiv.org/abs/2402.12275)；[Meta FAIR, 2025](https://arxiv.org/abs/2510.02387)；[Wang et al., 2026](https://arxiv.org/abs/2608.27549)；[Chen, Lin and Zhang, 2026](https://arxiv.org/abs/2608.25927)）；这里的重点是智能体从试验记录中估计模型参数，并用模型在部署前验证控制器。

这些试验提示，世界模型也许不必是在上万局随机模拟（rollouts）上训练出的神经网络。一个几百行的 Python 文件，从智能体自己的实验中就地拟合出来，或许就可以充当代码生成与物理部署之间的单元测试层。在绳索抛掷上接近 oracle planner（78% vs 84%），在未见过的 L 形方块上同样接近（留出 84% vs oracle 80%），而在宣布新航线且不给额外试验时，智能体复用已有模型、19.5 秒搜索即找到 100% 控制器。在第一条航线上，同等五次试验的 PPO 和神经动力学模型均为 0/50。

> **主动观察。** 世界模型智能体选择先看什么再行动。在推块任务上，两次脚本推动几乎精确定位了隐藏的质心。**隐藏动力学。** 物理包含手册未提及的变化：受损旋翼、横风、偏移的质心、一根末端加重的绳索，着陆点随微米级位移变化数厘米。世界模型将受损旋翼、横风和偏移的质心写成估计参数。对于绳索，模型在短时间尺度上再现了运动轨迹，足以测试抛掷，但未识别出末端配重。**随机变化。** 当 episode 随机变化时，少量真实试验说明不了什么。测试在多个参数略有不同的模型副本上运行大量模拟 episode，使程序在模型不完全准确时也必须成功。在加入随机飞行器的无人机任务上，世界模型智能体在未见过的 episode 上达到 98%，CaP 智能体同等五次试验仅 49%。在绳索抛掷上，控制器在每步执行前测试多个候选动作。**测试时计算。** 验证从真实试验移到计算：数千次模拟 episode 可在数秒内排序程序。以世界模型为测试环境，CaP 智能体在无人机任务上从 49% 升到 95%，零额外真实试验。在绳索抛掷上，每步在模型中测试候选动作，达到 78%，仅从真实试验编辑程序为 0–2%。

当随机性很高、隐藏物理不可知、且没有实时失败信号可以用来纠正时，这些测试对 Code-as-Policies 类方法可能帮助最大。

我们在四个仿真器的八个任务上尝试了这种做法，CaP 智能体按 ASPIRE 范式在五次试验上限下运行，每个世界模型为 14–26 KB 的单个 Python 文件，包含从智能体观测日志拟合的动力学方程、碰撞规则和摩擦参数。这些测试可以是错的：它们能有效排序程序，但会高估绝对成功率。**34 个程序的离线评估 MAE 5.2 pts, r = 0.974。**

我们给两种 LLM 智能体，世界模型智能体和 CaP 智能体，各 5 次真实试验，横跨八个任务：VizDoom 生存、四旋翼峡谷三个任务（确定性扰动 A/B + 随机飞行器 C）、Push-T 和 Push-L 两个推块任务、MuJoCo 线缆布线、以及 MuJoCo 绳索抛掷。在这些任务上，两个智能体的表现均超过了同等预算下的 PPO、神经动力学模型和出厂自带控制器。

**逐版本回测所见**: 在 智能体完成后，我们回溯评估了它们留下的每一个程序版本（seeds 6000+）。CaP 智能体在 Push-T 上的学习曲线是：72% → 90% → 92% → **96%** → **16%** → 16% → 96% → 96%。峰值出现在 3 次真实试验处，智能体在第 4 次真实试验后走掉了。这个智能体写出过 96% 的版本，但选出最佳版本需要比五次试验更多的证据。五次试验全部成功，在 95% 置信度下只能说明真实成功率高于 55%（单侧 Clopper–Pearson 界：0.05^(1/5) ≈ 0.549）。在 Push-T 的运行中，五次试验分散在前后版本上，每个版本大约只被测了一次。一次成功在同等置信度下只能说明成功率高于 5%（0.05^(1/1) = 0.05）。当 LLM 写程序时，新版本产生的速度远快于机器人上能测试的速度，测试成为瓶颈。机器人团队已经在仿真中测试控制器，但那些仿真器是为标称系统手写的，不包含特定机器人的故障。缺少的是一个闭环：每个新版本在上机前先在实际系统的模型中验证。用机器人自身试验拟合的世界模型是实现这一闭环的一种方式。在 Push-L 上同样：50% → 8% → 10% → 46% → 46% → **78%** → **36%**。

**离线评估**: 我们把 CaP 智能体留下的全部 34 个控制器程序（五个任务），交给世界模型评分。世界模型从未见过这些程序或它们的表现（有一个例外：推块 智能体在 Push-T v5 的 docstring 中看到了该版本的官方成绩 10/50）。结果：均绝对误差 (MAE) 5.2 个百分点，Pearson r = 0.974。

**CaP 以世界模型为测试环境**: 将世界模型交给 CaP agent 作为测试台，零额外真实 episode。四旋翼 Mission C: 单独 49% → CaP-in-world-model 95%（复测 95%，均时 9.68 s）；Push-L: 单独 16% → CaP-in-world-model 84%（复测 80%）。CaP 写程序、世界模型验证哪个版本该发布。

世界模型智能体在 Doom 上领先（存活 19162 vs 2385 游戏帧），在 Mission C 上领先（复测 95% vs 48%，CaP堆到 20 次真实试验仍只有 77%），在推块任务上等命数下领先（Push-T 复测 76% vs 22%；Push-L 72% vs 16%），而在确定性扰动下打平（Mission A/B 均 ≥ 98%）。但 CaP 智能体在推块任务的最佳版本上分数更高：Push-T 96% vs 世界模型智能体 82–90%；Push-L 78% vs 66%。在线缆任务上，CaP拥有一个现有程序 10 个 episode 的完整日志，在首次尝试前就达到 100%；没有这些日志的对照智能体也达到 94%，实时力反馈让隐藏物理变得无关紧要。

> **说明：** CaP 智能体按 NVIDIA GEAR 团队 [ASPIRE](https://research.nvidia.com/labs/gear/aspire/)（"ASPIRE: Agentic Skills Discovery for Robotics"，arXiv 2607.00272）范式的循环运行，每个任务上限 5 次真实试验。ASPIRE 本身使用远多于此的 rollout。

### 结果总表

> *†留出: 协议固定 seeds 上的评估 (Doom 及四旋翼 C: 1000–1099, 其余: 1000–1049); ‡Re-test: 独立复测 seeds, 无智能体曾见过 (四旋翼 C: 5000–5099, 其余: 5000–5049)*

#### Doom Take Cover（mean survival game ticks, held-out n = 100, re-test n = 50）

| 控制器 | 真实试验 | 留出† | Re-test‡ |
|--------|-----|-----------|--------|
| **世界模型** | 4 | **19,162** | 16,705 |
| CaP | 5 | 2,385 | 2,519 |
| PPO | 5 | 206 | — |
| PPO | 4,672 | 1,716 | — |
| Random | 0 | 287 | — |

#### 四旋翼 Mission A（success rate / mean time, n = 50）

| 控制器 | 真实试验 | 留出† | Re-test‡ |
|--------|-----|-----------|--------|
| **世界模型** | 4 | 98%, 8.03 s | 100%, 8.06 s |
| **CaP** | 5 | 100%, 9.50 s | 100%, 9.53 s |
| PPO | 5 | 0% | — |
| Neural dynamics model | 5 | 0% | — |
| 出厂自动驾驶 | 0 | 0% | — |
| PPO | 14,811 | 100%, 3.0 s | — |

#### 四旋翼 Mission B（新路线，零额外真实试验, n = 50）

| 控制器 | 留出† | Re-test‡ |
|--------|-----------|--------|
| **世界模型** | 100%, 11.28 s | 100%, 11.20 s |
| **CaP** | 100%, 13.98 s | 100%, 13.96 s |
| 出厂自动驾驶 | 0% | — |

#### 四旋翼 Mission C（随机其他飞行器，success rate / mean time, held-out n = 100, re-test n = 100）

| 控制器 | 真实试验 (Mission C) | 留出† | Re-test‡ |
|--------|----------------|-----------|--------|
| **世界模型** | 5 | 98%, 10.34 s | 95%, 10.43 s |
| **CaP in world model** | 5+10† | **95%**, 9.86 s | **95%**, 9.68 s |
| CaP | 5 | 49%, 10.69 s | 48%, 10.77 s |
| CaP (extended) | 20 | 76%, 10.06 s | 77%, 10.06 s |
| Lab oracle avoider | — | 75%, 12.94 s | — |
| Lab mission-A controller (忽略飞行器) | — | 31% | — |
| 出厂自动驾驶 | 0 | 0% | — |

*† CaP in world model: CaP 智能体的 5 次真实试验 + 建造世界模型的 10 次真实试验（跨 Missions A 和 C）；零额外真实 episode*

#### Push-T（success rate, coverage > 0.95, n = 50）

| 控制器 | 真实试验 | 留出† | Re-test‡ |
|--------|-----|-----------|--------|
| **世界模型** | 4 | 90% | 76% |
| CaP | 5 | 20% | 22% |
| CaP (extended) | 13 | 94% | 94% |
| oracle planner (真实仿真器) | — | 84% | — |
| Datasheet-model planner | — | 0/20 | — |
| Random | 0 | 0% | — |

#### Push-L（新方块，success rate, coverage > 0.95, n = 50）

| 控制器 | 真实试验 | 留出† | Re-test‡ |
|--------|-----|-----------|--------|
| **CaP in world model** | 2+2† | **84%** | **80%** |
| **世界模型** | 2 | 72% | 62% |
| CaP | 2 | 16% | 14% |
| CaP (extended) | 8 | 46% | 44% |
| oracle planner (真实仿真器) | — | 80% (n=30) | — |
| Datasheet-model planner | — | 0/20 | — |
| Random | 0 | 0% | — |

*† CaP in world model: CaP 智能体的 2 次真实试验 + 建造世界模型的 2 次真实试验（世界模型从 Push-T 的 4 次真实试验继承并用 2 次 Push-L 真实试验重新拟合）；零额外真实 episode*

#### Cable 线缆布线（MuJoCo, success rate, n = 50）

| 控制器 | 真实试验 | 留出† | Re-test‡ |
|--------|-----|-----------|--------|
| **CaP** | 5 | **100%** | 100% |
| **世界模型** | 3 | **100%** | 98% |
| Scripted (oracle) | — | 78% | — |
| Constant-speed program | — | 28% | — |
| Random | 0 | 0% | — |

---

## 演示

<p align="center">
  <img src="runs/doom/gifs/doom_showdown.gif" width="560" alt="Doom showdown">
</p>
<p align="center"><em>四个控制器在同一真实 VizDoom episode (seed 1000) 上的表现</em></p>

<p align="center">
  <img src="runs/drone/gifs/drone_showdown.gif" width="560" alt="Drone showdown">
</p>
<p align="center"><em>六个控制器飞同一 Mission A episode；图中可见隐藏的风区和受损旋翼（仅供观看者参考）</em></p>

<p align="center">
  <img src="runs/drone/gifs/drone_topdown_B.gif" width="560" alt="Drone Mission B top-down">
</p>
<p align="center"><em>Mission B 俯视图：复用 world model，零额外真实试验</em></p>

<p align="center">
  <img src="runs/drone/gifs/drone_C_showdown.gif" width="560" alt="Drone Mission C showdown">
</p>
<p align="center"><em>四个控制器飞同一 Mission C episode（随机其他飞行器和风区对观看者可见）</em></p>

<p align="center">
  <img src="runs/drone/gifs/drone_topdown_C.gif" width="560" alt="Drone Mission C top-down">
</p>
<p align="center"><em>Mission C 俯视图：同一场地加入随机其他飞行器</em></p>

<p align="center">
  <img src="runs/pusht/gifs/pusht_showdown.gif" width="560" alt="Push-T showdown">
</p>
<p align="center"><em>四个控制器在同一真实 Push-T episode 上的表现</em></p>

---

## 想法与起源

多个团队探索了将世界模型写成代码。[Code World Models](https://arxiv.org/abs/2405.15383) (Dainese et al., 2024) 让 LLM 从环境描述和离线轨迹将世界模型写成 Python，用树搜索生成、改进和修复代码，并用模型做规划。[WorldCoder](https://arxiv.org/abs/2402.12275) (Tang, Key and Ellis, 2024) 是一个在行动中写世界模型的 LLM 智能体，在网格世界和任务规划域中乐观探索并用模型规划。[CWM](https://arxiv.org/abs/2510.02387) (Meta FAIR CodeGen Team, 2025) 是一个在 Python 执行轨迹和智能体交互上训练的 32B 开放权重语言模型，用于建模代码执行过程。[Code as Worlds](https://arxiv.org/abs/2608.27549) (Wang et al., 2026) 让智能体将物理场景写成可执行代码（物体、动力学、外观），提出、运行、渲染和验证假设，用验证后的场景监督视觉-语言模型做定量物理推理。[Code World Model: Coding Agent as World Brain](https://arxiv.org/abs/2608.25927) (Chen, Lin and Zhang, 2026) 让编码智能体写出保存世界状态和规则的代码，这段代码驱动视频模型渲染画面。我们在这里探索一个相关的设定：智能体用少量真实试验观察物理控制任务，将世界模型写成代码，并用它帮助编写和验证控制策略，包括 Code-as-Policies 智能体写出的程序。

修改机器人程序的智能体需要测试，而在机器人领域每一次测试通常意味着一次真实试验。写成代码的世界模型是在智能体 harness 中提供这类测试的一种方式 ([Weng, 2026](https://lilianweng.github.io/posts/2026-07-04-harness/))。世界模型扮演的角色类似于单元测试对普通代码的作用：每一个候选程序在部署之前先在世界模型中运行。与单元测试不同的是，这些测试可以是错的。它们能有效排序程序，但会高估绝对成功率。

本项目的直接灵感来自 Saltline。据流传的介绍文章，这是一个单文件浏览器 FPS 游戏（一个 index.html 约 1.43 MB，由 Anthropic 的 Felix Rieseberg 用 Claude Fable 5.2 一次性提示生成）。"几猫"是在中文群组中流传的换皮复制版（存档：https://github.com/Megasu/jimao-3a-game ）。

这启发了一个问题：

> 如果 LLM 能把一个虚构的游戏世界写成代码，它能不能把一个**真实的**物理环境也写成代码？

本文档新增的内容是：一个只有少量真实尝试预算的不透明协议，以及世界模型可以同时充当 code-as-policy agent 的测试台和盲评评估器的量化证据。

这就是 **Code as World Models** 的核心思想：

1. **观察**：给 LLM 少数几次与不透明环境交互的机会（看不到源码）（"真实试验"）
2. **建模**：LLM 把观察到的物理规律写成一份可执行的 Python 代码，即世界模型
3. **想象**：在世界模型中搜索或优化控制器
4. **部署**：把在想象中找到的控制器直接部署到真实环境

这与已有的范式形成对比：

| 范式 | 写什么 | 训练在哪里 | 样本效率 |
|------|--------|-----------|---------|
| **Model-free RL** (PPO) | 神经网络策略 | 真实环境 | 数百至数千次真实试验 |
| **Neural dynamics model** | 神经网络动力学 | 真实数据 → 想象 | 四旋翼 5 次真实试验: 0%（未进一步测试） |
| **Code-as-policy** (CaP 智能体) | 代码策略 | 真实环境 debug | Doom/四旋翼 A/B 5 次; 四旋翼 C 20 次; Push-T 13 次 |
| **Code-as-world-model** (本项目) | 世界模型 | 世界模型想象 | 3–5 次 (含 Mission C) |

代码即世界模型的潜在独特优势在于**可复用性与可扩展性**：一旦世界模型写好，换一个任务目标或改一条路线只需重新搜索控制器，不需要再消耗真实环境的"命"。四旋翼 Mission B 正是复用的展示：复用 Mission A 的世界模型，仅花 19.5 秒 wall time 即得到 100% 成功率的控制器。Mission C 则展示了扩展：世界模型从 15466 bytes 扩展到 25975 bytes，用 5 次新真实试验加入了其他飞行器模型后，在包含随机长尾危害的场景中达到 98% 成功率，而 CaP用同样的 5 次真实试验仅达 49%。

---

## 流水线 (Pipeline)

```
Orchestrator (Claude Fable 5.2)
  │
  ├── 世界模型智能体 (Claude Opus 5.5)
  │     ├── 观测试验: 主动实验，收集系统观测数据
  │     ├── 拟合世界模型: 逐机制参数拟合 (CEM / least-squares / 梯度下降)
  │     ├── 世界模型 (.py): 可执行的 Python 仿真器 (pure torch, batched)
  │     ├── 控制器搜索: 策略代码 / CEM 增益调优 / sampling MPC 在世界模型中运行
  │     └── 评估 (evaluation episodes): 控制器部署到真实环境（评估 ≠ 真实试验，不可用于学习）
  │
  ├── CaP 智能体 (Claude Opus 5.5)
  │     ├── 编写策略代码
  │     ├── 在真实环境执行
  │     ├── 读 trace → 找 bug → 修策略
  │     └── 循环至收敛或耗尽真实试验
  │
  └── Black-box Labs (Claude Sonnet 5.5; cable: Claude Opus 5.5)
        ├── VizDoom 环境
        ├── PyBullet/PyFlyt 无人机环境
        ├── pymunk Push-T 环境
        └── MuJoCo 线缆/站台环境
```

**预算计数器**: 硬性预算，由 `cawm/ledger.py` 强制执行。"真实试验"仅指 智能体可以从中学习的观测试验（种子 < 1000）。评估 episodes 使用协议固定的种子（1000+），智能体不可从中学习。CaP 智能体在 Push-T 上额外获得了 8 次真实试验用于延伸实验（共 13 条），在四旋翼 Mission C 上额外获得了 15 次真实试验（共 20 条），结果中明确标注。

---

## 智能体如何与仿真器隔离

在仿真中评估控制器时，一个常见陷阱是智能体可以接触到仿真器本身的信息（数值方法领域称之为 inverse crime）。为避免这种循环论证，实验实施了严格的信息隔离：

1. **Clean-room split**: 构建环境的 agent（唯一知道隐藏异常的角色）与智能体完全隔离。智能体只能阅读环境的仪表手册（instrument manual）和自己的试验记录。
2. **不透明环境**: 智能体通过环境 API 与环境交互。环境内部实现不可见。
3. **Instrument-manual docstrings**: 环境 API 文档仅包含传感器规格、动作空间、奖励定义，不含动力学细节。
4. **预算计数器硬上限**: 每个智能体最多 5 次观测试验。观测种子 < 1000；评估种子（1000+）永远不可用于观测。
5. **Import guard test**: 世界模型文件不得 import 任何环境模块。
6. **Hidden anomalies**: 每个任务包含智能体必须从观测中自行发现的隐藏异常。
7. **Unseen re-test seeds**: 独立复测使用 seeds 5000–5049，无智能体曾见过。
8. **审计**: 我们检查智能体的代码提交是否违反隔离规则。

### 真实试验消耗记录

| 智能体 | 任务 | 真实试验 (观测) | 步数 |
|-------|------|------------------|------|
| 世界模型 | Doom | 4 | 6,216 |
| CaP | Doom | 5 | 1,182 |
| 世界模型 | 四旋翼 A/B | 5 | 1,654 |
| CaP | 四旋翼 A/B | 5 | 1,354 |
| 世界模型 | 四旋翼 C | 5 | 1,810 |
| CaP | 四旋翼 C | 5 + 5 + 5 + 5 = 20 | 6,031 |
| 世界模型 | Push-T | 4 | 811 |
| CaP | Push-T | 5 + 5 + 3 = 13 | 2,094 |
| 世界模型 | Push-L | 2 | 600 |
| CaP | Push-L | 2 + 6 = 8 | 2,367 |
| 世界模型 | Cable | 3 | — |
| CaP | Cable | 5 | — |

*来源: `runs/ledger_summary.txt`, 仅 purpose = 观测 的条目*

---

## 各任务详解

### 一、VizDoom：测试时计算，4 次试验存活 19162 游戏帧

**环境**: 576 × 768 的封闭房间。玩家站在 x = −416 只能沿 y 轴平移闪避。Imps 站在 x = 64 投掷火球，每颗命中造成 21 × U{1,8} 伤害（一击约有一半概率致死）。最多 13 个 imps，spawn 周期 60 游戏帧。Reward = 存活 tick 数。

#### 世界模型 (16599 bytes, 4 次真实试验)

世界模型智能体从 4 次真实试验中诊断出 16 种机制，写出了一个 batched torch 仿真器。每个参数都附有来自真实观测的证据：

```python
# 玩家动力学 (逐 tic)
ACCEL = 0.75       # fit.py least squares on life 1 (95 steps)
FRICTION = 0.9065  # 1-step y RMSE 0.012 units
#                    → 终端速度 ACCEL*F/(1-F) = 7.27 u/tic, life 1 观测值 7.25

# 火球参数
BALL_SPEED = 10.0  # life 1: |v| = 10.000 for all 8 balls (std 3e-6)
HIT_HALF = 23.0    # 碰撞: |dx|<23 and |dy|<23
DAMAGE_UNIT, DAMAGE_MAX_K = 21.0, 8  # 伤害 = 21 × U{1..8}

# Imp 行为
SPAWN_PERIOD = 60   # 每 60 游戏帧 尝试新增一个 imp
FIRE_RATE = 0.0068  # life 3: 1492 throws / 218k imp-tics
```

#### 控制器

最终控制器是一个**手写的、向前看 24 个 tick 的规划器**。它的参数在世界模型中调优，并通过在世界模型中的评分从多个候选方案中被选出。世界模型总共消耗了约 389M 想象步，其中包括两轮落选的 PPO 训练（150M + 200M 步；PPO 在世界模型内得分约 2764，远低于 planner 的约 23000 capped 得分）以及 planner 的调优和选择。

#### 世界模型的乐观偏差

在 24000-tick 上限下做同口径比较：世界模型预测均值 22998、91% 的 episode 存活到上限；真实环境（同样封顶 24000 游戏帧）均值 14318、仅 26% 存活到上限。偏差约 **1.6 倍**。控制器在"乐观版"世界中训练后部署到更难的现实仍然有效，但这种偏差在更脆弱的任务上可能致命。

**保真度**: 玩家 1-step 位置误差 0.003–0.009 units；火球 40-tick 位置误差 < 0.001 units；视野判断准确率 99.7–100%。

#### 结果

CaP 智能体用 5 次真实试验迭代修复了 5 个 bug：视野外目标跟踪 → 墙壁安全距离 → swept-box 碰撞检测 → coast-don't-brake → 视野外命中（未修复）。

**上下文**: World Models (Ha & Schmidhuber 2018) 在同一 VizDoom Take Cover 任务上报告了 1092 ± 556（temperature 1.15, 10000 random rollouts 训练 V 和 M 模型，"solved" = 超过 750，其随机基线 210 ± 108）。注意这不是同口径比较：World Models 使用像素输入，本实验使用结构化状态观测。PPO 约需 454 次真实试验才首次超过 750 游戏帧。

---

### 二、四旋翼：系统辨识下确定性扰动打平，随机飞行器后世界模型领先

**环境**: PyBullet/PyFlyt Crazyflie 2.x 四旋翼在集装箱堆场峡谷中飞行。30 Hz 控制频率，body rate + collective throttle 指令。峡谷最窄处仅 3.2 m 宽。

**Mission A**: 穿过 3 个 检查点，17 秒时间窗。出厂位置自动驾驶 (出厂自动驾驶) 被限速 2 m/s，设计上无法在时限内完成。

**Mission B**: 进入北侧巷道的新路线（在真实试验全部用完后才公布），24.2 秒时间窗。

#### 隐藏异常

- **受损旋翼**: 前左电机 (motor 2) 推力和扭矩系数仅为正常值的 60%
- **侧风喷射**: 北侧巷道 (x ≈ 23.75) 有 40 m/s 南向喷流；南侧巷道 (x ≈ 39.75) 有 30 m/s 北向喷流

#### 诊断对比

两位智能体都成功检测到了全部隐藏异常。两位智能体都在第 1 次真实试验坠毁，从第 2 次真实试验起成功飞行。

| 异常 | 隐藏真相 | 世界模型诊断 | CaP诊断 |
|------|---------|-------------|-------------|
| Command delay | 1 control step | 1.04 steps | 2 steps |
| Rate bias | (受损旋翼的外在效应) | (−0.52, +0.52, −0.26) rad/s | (~−0.50, +0.54, −0.27) rad/s |
| Thrust deficit | Motor 2 仅 60% | 整体增益 0.875 | hover throttle 0.33–0.35 vs datasheet 0.30 |
| North jet (→ −y) | 40 m/s 气流，核心加速度约 14.8 m/s²，x = 23.75 | −16.7 m/s² at x = 23.8, σ = 1.38 | −13 to −18 m/s² |
| South jet (→ +y) | 30 m/s 气流，核心加速度约 11 m/s²，x = 39.75 | +10.4 m/s² at x = 40.0, σ = 1.22 | +7.5 to +12.4 m/s² |

区别在于它们如何利用诊断：世界模型智能体把所有参数写入世界模型并在想象中优化控制器；CaP直接在真实环境中调试策略。

#### 真实试验的使用

世界模型智能体: 4 次真实试验（life 1–4）用于拟合世界模型和搜索 Mission A 控制器。第 5 次真实试验在 Mission A 评估完成后飞行，用于探索巷道入口，仅为 Mission B 提供信息。

#### 世界模型的精度

世界模型 (25975 bytes, 含 Mission C 追加的其他飞行器模型；Missions A/B 版本为 15466 bytes) 从零开始用 `DATASHEET` 参数定义所有物理机制，然后用观测数据拟合实际值（四旋翼的参数通过对世界模型的梯度下降拟合）：

```python
DATASHEET = dict(
    CMD_DELAY=0.0,      # 起始值：无延迟
    THRUST_GAIN=1.0,    # 起始值：公称推力
    JET_N_AY=0.0,       # 起始值：无侧风
    JET_S_AY=0.0,       # ...
    # 这些参数在拟合后被实际诊断值覆盖
)
```

在保留真实试验（held-out trials）上测试，世界模型的 1 秒开环位置误差中位数为 **0.16–0.25 m**，而仅依赖 datasheet 参数的模型误差为 1.2–2.1 m（**5–13 倍改善**）。对比神经世界模型（MLP, 5 次真实试验数据训练），1 秒位置误差约 0.82 m，世界模型在长时域上精度高出 3–5 倍。

**世界模型的自我预测**: 部署前，世界模型预测了自己的控制器表现。Mission A: 95.7% 成功率、8.0 s imagined vs 真实 98%、8.03 s；Mission B: 99.6%、10.78 s vs 真实 100%、11.28 s；Mission C: 94.3%、10.35 s vs 真实 98%、10.34 s。Missions A/B 略偏乐观，Mission C 略偏悲观（因为世界模型故意将其他飞行器的群体采样范围画得比观测到的更宽）。

#### 控制器与飞行特征

手写 cascade policy（8 个增益由 cross-entropy search 在世界模型中调优）。在复测中，世界模型智能体的飞行轨迹更贴近墙壁（最小距离 0.21–0.26 m vs CaP的 0.53–1.10 m）。

世界模型智能体在 Mission A 的 留出种子评测中有 1 次失败（错过一个检查点后悬停直到超时），该失败在世界模型中被复现并修复，Mission B 控制器不再有此问题，但 Mission A 的数字保持不变。

#### Mission B 的近乎免费复用

复用 Mission A 的世界模型，仅重新搜索控制器增益和路线：6.89M wm steps，19.5 s wall time（Mission A: 9.56M steps，584 s）。

#### 参考控制器

Lab 自身的真相知情控制器 (oracle, truth-informed): Mission A 86% 成功率、11.3 s；关闭其 wind feed-forward 后仅 13%。PPO 的 14811-life 策略以约 18 m/s 飞行，3.0 s 完成 Mission A，但需要 14811 次真实试验训练。

#### Mission C: 随机其他飞行器

**设计动机**: Missions A/B 的结论是确定性扰动下两者打平。那如果在同一场地加入随机的长尾危害呢？Mission C 正是对这一预测的检验：同一峡谷、同一受损旋翼、同一侧风，加上每个 episode 7–8 架随机其他飞行器（半径 0.35 m 的运动学球体）。Pilot 事先不知道它们的行为模式。时间窗 19.4 秒。每位智能体在 Mission C 上获得 5 次新真实试验；评估 100 episodes。

**其他飞行器的行为**（lab 隐藏真相，来自 `reallab/drone/_movers.py`）：

4 种状态机：沿峡谷巡逻 (along, 25%)、横穿峡谷或侧巷 (across, 25%)、悬停后冲刺 (dart, 20%)、反应式，巡逻但在智能体接近触发半径内会拦截/追逐/后退 (reactive, 30%，其中 block 40% / chase 30% / shy 30%)。每个飞行器每 episode 有 5% 概率出现一次急冲（3.5 倍速、0.8 s）和 5% 概率出现一次突然反向。观测仅列出 15 m 以内且视线内的飞行器。

**世界模型 (5 次真实试验, 留出 98% / 复测 95%)**:

第 1 次真实试验为纯观测：智能体让无人机在起点悬停 19.4 秒，收集飞行器轨迹。试验日志记录了每架飞行器的完整轨迹。从这些轨迹中，智能体写入了四种行为模型到世界模型：沿峡谷穿梭、横穿穿梭（含穿过侧巷的）、随机游荡者、以及靠近时追击的"守卫"（估计约四分之一的穿梭者是守卫）。在想象中发现除追击的守卫外所有飞行器都在 z 1.6–3.6 m 之间活动，因此合成了一条 z = 4.8 m 的高空航线，对每个检查点陡降。世界模型中的进展：phase-1 控制器 22–37%，高空航线 78–84%，航线+避让+调优 94%。约 34.6M 想象步，793 s 合成时间。第 2–5 次真实试验飞行了逐版控制器，全部成功。世界模型扩展到 25975 bytes（A/B 版本为 15466 bytes）。

世界模型预测：94.3% 成功率、10.35 s，这次略偏悲观（因为其他飞行器群体的采样范围被故意画得比观测到的更宽）。

**CaP (5 次真实试验, 留出 49% / 复测 48% → 20 次真实试验, 76% / 77%)**:

使用闭合形式的最近接近点 (constant-velocity CPA) 避让法，逐次真实试验修复。观测试验中 7/8 次真实试验成功，但 留出种子评测只有 49/100，观测试验成功率没有预测到真实表现。每次 crash 暴露一个场景并修复（seed 517 的 crash 在 v11→v12 后通过；另一个 seed 连续 crash v12、v13，在 v14 后通过）。CaP 智能体观察到其他飞行器会对自己做出反应，但常速预测无法预知这种反应。

一个重要的完整性注释：CaP 留出种子评测的程序 (v11) 是在 5 次真实试验用完后修改的，**从未在修改后被飞行过**，只通过回放日志验证，这一点由智能体自己标注。

**参考**: 环境的 oracle 避让控制器（对观测到的飞行器做常速前瞻）75%、12.94 s。环境的 Mission A 控制器（忽略飞行器）仅 31%。出厂自动驾驶 0%。

---

### 三、Push-T：策略评估缺失，96% 控制器在第 4 次试验后降为 16%

**环境**: pymunk 2D 物理引擎上的操控任务。圆盘推手（半径 15 单位）推动 T 形方块到目标位姿 (256, 256, π/4)。10 Hz 控制，成功标准: coverage > 0.95。

#### 隐藏异常

- **质心偏移**: T 形方块的质心从图纸质心 (0, 45) 偏移到 **(24.0, 30.0)**，沿 u 轴偏了 24 个单位，意味着"看起来应该直推的推力实际上会让方块转弯"
- **释放后滑行**: pymunk Space.damping = 0.01（名义值为 0），方块在推手离开后继续滑行

#### 质心诊断：关键对比

| | 图纸质心 | 世界模型诊断 | CaP估计 | 隐藏真相 |
|---|---------|-------------|-------------|---------|
| COM (u, v) | (0, 45) | **(24.1, 30.0)** | ~(3, 44) | **(24.0, 30.0)** |

世界模型智能体几乎精确命中了隐藏真相。关键证据来自第 1 次真实试验的一次轴向推力实验：

```python
# M3 centre of mass in the body frame [u].
# The drawing's centroid is (0, 45); fits give (24.3, 30.2) on life 1,
# (25.2, 29.1) on life 2, (24.1, 30.0) on lives 1+2.
# Evidence: a push along the symmetry axis through the stem tip
# (life 1, steps 68-81) turned the block by -18 deg in 13 steps.
# Forcing the COM to the centroid raises the 10-step replay loss
# from 0.41 to 3.06 (life 1).
com_u=24.1, com_v=30.0,
```

独立拟合第 1 和第 2 次真实试验的结果分别为 (24.3, 30.2) 和 (25.2, 29.1)，两次真实试验之间一致性极好。将 COM 强制回图纸质心后，replay loss 暴增 7 倍。

CaP 智能体通过回归估计 COM 收敛到 ~(3, 44)，距真相偏差约 25 个单位。但它通过反复 debug 策略代码、在 13 次真实试验后达到了 94% 复测成功率。事实上，逐版本回测揭示了更戏剧性的故事：CaP 智能体在第 3 次真实试验时就达到了 **96% 的真实成功率**（全场最高），但在第 4 次真实试验时跌落到 **16%** 并提交该版本用于留出种子评测。五次测试飞行全部成功，但那是不同版本积累的成功，最终提交版本自身只有 16%。

**控制器**: Sampling MPC（2048 samples），在世界模型中运行。

**世界模型的乐观偏差**: 世界模型在 32 个想象 episode 上预测 100% 成功率（参数抖动后 91%）。两组真实评估（留出 90%、复测 76%，合计 83/100）都低于此预测。两组评估的差异来自同一控制器在不同种子集上的随机波动（MPC 规划器内部有采样随机性）。

#### ⚠️ 重要 caveat: 模型家族巧合

Push-T 的世界模型采用了 velocity-level impulse 接触模型。真实环境使用的 pymunk 引擎属于**同一物理模型家族**（刚体冲量法）。在真实的机器人场景中，智能体不可能事先知道目标环境用的是什么物理引擎。这一结构上的巧合使得 Push-T 的结果可能高估了 世界模型智能体在接触密集任务上的泛化能力。

---

### 四、Push-L：real-to-sim-to-real，CaP 以世界模型为测试环境达 84%

**环境**: 与 Push-T 共享同一 pymunk 物理引擎，但使用 L 形方块。每个智能体获得 2 次真实试验。

#### 隐藏异常

- **质心偏移**: L 形方块的质心从图纸质心约 (45, 37.5) 偏移到 **(23.0, 55.5)**
- **接触摩擦**: 推手与方块之间的接触摩擦 pair coefficient **0.36**（Push-T 中几乎为零）
- **释放后滑行**: 与 Push-T 相同（damping = 0.01）

#### 世界模型 (2 次真实试验, 72%)

世界模型智能体直接复用了 Push-T 的世界模型（该世界在 Push-T 阶段用了 4 次真实试验），替换了 L 形几何和图纸参数，然后用 2 次 Push-L 真实试验重新拟合：质心估计 **(23.4, 55.6)**（真相 (23.0, 55.5)，偏差不到 1 个单位），接触摩擦估计 **0.4**（真相 pair coefficient 0.36）。世界模型在想象中预测 94% 成功率；三组评估合计 99/150 = 66%（留出 72%、复测 62%、第三组 64%），偏差 28 个百分点，9/14 失败案例撞墙。

#### CaP (2 次真实试验, 16% → 8 次真实试验, 46%)

CaP 智能体用 2 次真实试验达到 16% 成功率。延伸至 8 次真实试验（2 次初始 + 6 次额外）后达到 46%。

#### 逐版本回测曲线

CaP在 Push-L 上的学习曲线（seeds 6000+）：50% → 8% → 10% → 46% → 46% → **78%** → **36%**。与 Push-T 相同的模式，峰值（78% 在 6 次真实试验）出现后又跌落。

#### 参考

oracle planner（知道真实仿真器）80% (n=30)。Datasheet-model planner 0/20。

---

### 五、Cable：闭环反馈消除隐藏物理，验证器仍捕获 bug

**环境**: MuJoCo 仿真的约 1 m 弹性线缆链。一端固定，另一端由工具夹持。工具在平面内以 2-d 位置指令移动（20 Hz），速度限 0.30 m/s。三根柱子，线缆必须按绕行角规则通过每根柱子的指定侧，工具必须停在距目标 2 cm 以内。力读数（"load"）每步报告；超过 10 N 即刻失败。时间限 30 s。布局每 episode 不同。

#### 隐藏异常

- **弯曲刚度**: 3× 标称值
- **线缆-柱摩擦**: 1.0（标称 0.3）
- **粘性阻力**: 90 N·s/m per metre of cable（标称 0）

快拉就过载。智能体不被告知这是什么装置，仪表手册称之为"installation"，仅记录接口：20 个追踪点、一个工具、三根柱子、一个力读数、成功规则。无图片。一条场地备注：操作人员说它"你赶它就反抗"。

两位智能体在首次真实试验之前都提交了书面假设。两位都从数据中正确推断：这是一根约 0.99 m 的不可伸线缆，一端固定、另一端夹在工具上；这是一个线缆布线任务；力随速度和线缆绕柱量增长。证据：point 1 在 200 张快照中标准差为零（固定端）；point 20 与工具重合到微米量级；折线长度始终为 0.991–0.995 m；所有点高度 4 mm。

**参考**: random 0%；现有匀速程序 28%（关闭隐藏更改后 90%）；环境知情保守脚本控制器 78%（保守参考，非天花板）。

#### 为什么这个任务比呈现的更容易

事后的 forensic 分析揭示了三个因素：(1) 成功规则要求线缆从每根柱子的指定侧通过，但 14–29 cm 的松弛意味着线缆在成功 episode 结束时平均只承压约 1 根柱子（CaP 1.16, CW 1.02），实际上只有约 1 根柱子承压，多柱张力路径规划并未被测试到；(2) 力被实时送给控制器，一条反应规则（力大了就慢，太大就停，再大就后退）让隐藏物理变得无关紧要；(3) CaP 的平均速度 (0.084 m/s) 和路径长度 (0.72 m) 远低于速度限 (0.30 m/s) 和理论最短路径，它走得慢且绕得远，因此安全。Lab 的知情控制器 (78%) 是保守参考，它用 417 步，比 CaP (171 步) 慢得多，平均速度仅 0.045 m/s。

#### CaP (5 次真实试验, 100%)

首个程序在任何真实 episode 之前从观测日志写出，做力控调速。5 次真实试验全部成功。留出 50/50；复测 50/50；seeds 6000+ 所有三个版本均 50/50。平均峰值力 5.0 N（限 10 N），171 步（8.5 s）。

#### 控制实验：无日志的 blind 智能体 (5 次真实试验, 94%)

为检验免费日志是否驱动了结果，我们运行了一个控制实验：一个没有 10-episode 免费日志的 blind CaP 智能体。它同样在首次真实 episode 之前就从手册假设中推断出这是一根约 0.99 m 线缆的布线任务，写出了力控调速程序。3 个版本得分 44/50, 47/50, 50/50（v2 提交，留出 47/50 = 94%，复测 49/50 = 98%）；失败集中在 3 个种子上的路径 bug。这证明实时力读数是让任务容易的关键。

#### 世界模型 (3 次真实试验, 留出 100%, 复测 98%)

使用 3 次真实试验。世界模型 13,870 bytes，20 节点弹簧阻尼链。位置误差 2.8/8.7/20.5 mm (步后 5/20/60 步)，力误差 0.26/0.31/0.37 N。控制器较慢：350 步（17.5 s）。

#### 世界模型作为评估器

**20 个开环程序**：MAE 6.0 pts，r = 0.976，916/1000 个单独 episode 一致。

**控制 智能体的 3 个程序**：世界模型精确预测了 blind 智能体三个版本：v1 world = 44/50 (real 44/50), v2 world = 47/50 (real 47/50), v3 world = 50/50 (real 50/50)，在 150/150 个 episode 上结果一致，失败种子精确匹配。23 个程序的汇总 MAE 5.2 pts，r = 0.980。世界模型在这里的价值在于捕获 5 次成功飞行未暴露的 bug（力读数已让 CaP 轻松区分好坏程序）。

---

### 六、Rope：模型预测控制，0.16 s 重规划达 78%

**环境**: MuJoCo 仿真的 0.70 m 绳索，1 cm 粗，平躺在桌面上。一端由工具夹持（仅能在 0.25 × 0.40 m 区域内移动）。一根柱子（半径 3 cm）距区域中心 0.40–0.56 m；目标为柱子后方 0.12 m 处的一个半径 0.10 m 的圆盘。工具无法触及目标，只有绳索自由端可以。成功：绳索和工具完全静止（每个追踪点速度 ≤ 0.02 m/s），自由端在圆盘内，236 步（4.72 s）内完成，50 Hz。读数：20 个追踪点、工具位置和速度、柱子、目标。无力读数。工具限 4 m/s、60 m/s²。

#### 隐藏异常

- **弯曲刚度**: 2.0e-3 N·m²
- **桌面摩擦**: 0.10
- **柱摩擦**: 0.25
- **质量**: 56 g 绳 + 30 g 自由端配重
- **粘性阻力**: 有

两位智能体不被告知这是什么装置。手册原文场注："the people who run this installation say you get one go at it, and that doing it slowly gets you nowhere. Nobody has written down why. They also say that a small change in how you do it changes where it ends up."

与其他任务不同，没有任何运动中的录像给到任何人。两位智能体有手册（接口和精确成功规则）、200 张静态快照（复位后读数，无命令发出）、5 次真实试验。

#### 为什么是这个任务

线缆任务是准静态的，可以慢行、可以纠正、力被实时送到。绳索任务相反：没有慢路径，只有一次抛掷。Lab 的灵敏度测量：一个成功抛掷的参数变化 ±5%，着陆点移动 6.5 cm（中位；90th 百分位 17 cm），目标半径 10 cm。精确物理但从 20 点读数重建起始状态（非复制）：17% (4/24)，开环。移动每个顶点 1 微米使 2/7 个成功计划失败。参考（在真实仿真器内搜索抛掷，每 episode 至多 1,000 次试抛）：**42/50 (84%)**。

#### CaP (5 次真实试验, 2%)

首次真实试验前写出假设：这是一根"约 0.70 m 长、~1 cm 粗的柔性绳/缆"，需抛掷使自由端落入圆盘。5 次真实试验，1,180 步，全部失败；自由端距目标 0.62–1.16 m。提交 v6 (3,603 bytes)：全速冲向柱旁瞄准点、立即反向、保持。留出: **1/50 (2%)**，中位距离 0.99 m。

延伸至 19 次真实试验 (4,484 步)：v7–v19。10 次真实试验在同一布局上飞行（环境确定性）。该布局上 ≥ 0.23 m 行程的 8 次抛掷中 7 次将自由端送至目标 0.05–0.14 m 内，但从未停在目标内（绳索翻转反弹或擦柱塌下）。v19 (8,973 bytes): **0/50**，中位 1.00 m。未见种子 (5000–5049): v6 0/50, v19 0/50。全部 19 个版本在评估种子上 0/50–1/50。

#### 世界模型 (4 次真实试验, 32%; 延伸至 20 次真实试验, 78%)

首次真实试验前写出假设：不可伸绳索平躺于桌面，库仑摩擦，需抛过柱子。4 次初始试验 (944 步)：试验 1 为观测脚本（保持、慢拖、阶跃、蓄力、抛掷）；试验 2–4 飞行连续控制器；全部失败（距目标 0.98, 0.75, 0.13, 0.58 m）。

**第一版世界**: 14,313 bytes，20 个质点的平面链，含拉伸弹簧、弯曲刚度和阻尼、桌面摩擦和阻力、柱碰撞和工具伺服；11 个参数从其真实试验中拟合。世界预测 24/24 (100%)；智能体书面补充真实数字会远低于此。**留出: 16/50 (32%)**。未见种子复测: 24/50 (48%)（合并 40/100）。

**延伸 (20 次真实试验, 3,766 步)**: 4 次脚本行程试验（1–4 m/s + 长保持）发现绳索抛出后弹性摆动超 70 步，低速时摩擦低。**修订世界** (16,069 bytes)：两个新机制：速度依赖摩擦（拟合 0.28 高速, 0.09 低于 ~0.7 m/s）和接近不可伸的链节在绷紧时耗散能量。每个机制贡献约一半增益：两次保留探测试验上，自由端 60 步后误差 24.6 → 19.3 → 14.5 cm。

最终控制器：同结构在修订世界中规划，6 个 ±35% 扰动想象世界，每 8 步 (0.16 s) 从实时读数重新规划。实际飞行 12 次真实试验后评估：9 次成功（最后 5 条为冻结控制器试飞: 4/5）。评估前 13 s 写下预测：65%（45–80% 不意外），来自 9/12 次实际试验。其世界自身预测 10/10；它书面写道刻意不使用该数字。**Extended: 39/50 (78%)**。未见种子: **39/50 (78%)**。距真实仿真器内规划的参考 (42/50) 差 3 个 episode。

#### 保真度

世界永远不够精确到可以开环抛掷（精确物理从读数重建：17%）。三次未用于拟合的控制器试验上：第一版世界 3.0 / 13.7 / 40.4 cm (5/20/60 步)；修订世界 1.2 / 5.8 / 17.4 cm。控制器每 8 步重新规划。有效的是只准几分之一秒的模型，每 0.16 s 从实际状态重规划，并对多个扰动世界取最优。

智能体始终未识别自由端的 30 g 配重（约为绳索每节质量的 11 倍）。其拟合两次将自由端质量推至上界（2×, 然后 6×），仅记录"惯性分布仍不正确"。

#### 世界模型作为评估器（弱测试）

19 个 CaP程序在第一版世界中评分。几乎所有程序 0–2%，世界预测 0–6%（评估种子均误差 0.9 pts）、0–10%（未见种子均误差 2.7 pts）。一致性廉价：950 个评估 episode 中世界预测 7 次成功、真实 2 次，无重叠。单 episode 距离相关仅 0.27。

#### 协议披露

假设文件先于各智能体的第 1 次真实试验。留出评估日志未被任何智能体用于拟合或调优。智能体运行期间，线缆任务的程序、日志、世界模型和全部文档被移出项目并锁定。绳索在第 2 次真实试验中离开桌面一次（0.4 m 反向行程 at 4 m/s，升高 0.27 m），其余 19 次真实试验平面世界成立（3 m/s 限速下）。

---

### 逐版本回测：策略评估缺失，学习曲线上下游走

在所有 智能体完成后，我们回溯评估了每一个留下的程序版本（在 seeds 6000+ 上，智能体从未见过），按消耗的真实试验数索引。这揭示了 CaP 智能体的编辑策略的一个系统性问题：**当 episode 之间有差异时，编辑策略会游走 (wander)**。

| 任务 | CaP学习曲线 (seeds 6000+, 按真实试验数) |
|------|----------------------------------------|
| Push-T | 72% → 90% → 92% → **96%** → **16%** → 16% → 96% → 96% |
| Push-L | 50% → 8% → 10% → 46% → 46% → **78%** → **36%** |
| Drone C | 66% → 9% → 54% → 54% → 56% → 79% → 87% → 85% |
| Doom （游戏帧） | 596 → 1049 → 648 → 1889 → 2455 |

世界模型智能体在同一坐标轴上的曲线更稳定：四旋翼 A 98% → 100% → 100%；四旋翼 C 50%（沿用 A 的控制器）→ 97% → 99% → 95% → 96%。四旋翼 C 的跳跃最为鲜明：第一次真实试验纯观测后，一次真实试验就从 50% 跳到 97%。

**种子范围一致性检查**: 同一程序版本在不同种子范围上得分相似：CaP Push-T 提交版本 16%/20%/22%（seeds 0–99 / 1000+ / 5000+），四旋翼 C: 52%/49%/48%，世界模型智能体四旋翼 C: 92%/98%/95%。观测种子不比评估种子更容易。

---

## 解读 (Interpretation)

> 以下为对实验结果的初步解读，**不宜视为已被证明的结论**。

### 提出与验证：CaP 写程序，世界模型选出该发布的版本

我们没有预料到的发现是两种方法**互补**（代码即世界模型有效在意料之中）。CaP 智能体在推块任务的最佳版本上分数更高（Push-T 96% vs 82–90%；Push-L 78% vs 66%）；在线缆上，它甚至不需要真实 episode 就达到了 100%。它写好代码的能力足够；缺的是区分好代码和坏代码的能力。世界模型提供了这种区分：离线评估 34 个程序的 MAE 仅 5.2 pts。CaP 以世界模型为测试环境把两者串成一条流水线：一个写候选控制器，另一个排名。

### 按任务类型

**随机变化与长尾失败 (Doom, Mission C)** → **大胜**。Doom: 每条真实命只暴露一个致命 bug（视野外命中、墙角卡死等）；世界模型让 agent 在想象中发现和应对这些场景，而无需真实经历。手写 planner 在世界模型中击败了同样在世界模型中训练的 PPO（~23000 capped vs ~2764），说明可解释的策略结构在此类任务上有优势。四旋翼 Mission C 在**同一机体、同一场地**上提供了一个更强的 within-robot 对照：确定性扰动（Missions A/B）下两者打平，加入随机其他飞行器后在等命数下拉开到 95–98% vs 48–49%，CaP堆到 4 倍命数（20 条）仍只有 76–77%。这一对比比 Doom 更有说服力，因为两个条件共享同一物理平台和同一 智能体的先验知识，差异仅来自随机性的引入。

**策略评估与 real-to-sim-to-real (Push-T, Push-L)** → **等命数下领先，两种方法互补**。世界模型 复测 76% vs CaP 22%（Push-T, 等命数）；72% vs 16%（Push-L, 2 次真实试验）。但逐版本回测显示 CaP 智能体的最佳版本（Push-T 96%, Push-L 78%）都超过了世界模型智能体，它写好控制器的能力足够，缺的是知道哪个版本是好的。CaP 以世界模型为测试环境在 Push-L 上达 84%，超过两者单独的表现。值得注意的是，oracle planner（使用真实仿真器）在 Push-T 上也只有 84%，Push-L 80%，瓶颈在 sampling planner 的利用能力。

**系统辨识与确定性扰动 (Drone A/B)** → **成功率打平，世界模型增值在别处**。两者均 ≥ 98% 成功率。世界模型额外提供了：(1) 飞行前的自我预测（95.7% imagined vs 98% real），(2) 更快的飞行（8.03 s vs 9.50 s），(3) Mission B 的 19.5 s 近乎免费重新合成。

**闭环反馈 (Cable)** → **世界模型的边界**。实时力读数让 CaP 94–100%（有或没有免费日志均如此），隐藏物理变得无关紧要。成功规则加上松弛意味着线缆只承压约 1 根柱子，不测试多柱张力路径。世界模型在这里的价值是捕获程序 bug：blind 智能体的 3 个版本在 150/150 个 episode 上被精确预测，包括失败种子的匹配。我们尝试在同一装置上构造互补条件（移除力读数 + 缩短时限），但无法把窗口做得足够紧，按预设规则终止，未运行智能体。

**模型预测控制 (Rope)** → **世界模型作为在线规划器**。没有慢路径，没有力读数，一次抛掷决定结果。CaP 从真实尝试编辑程序，19 次真实试验后仍 0–2%。它学到了正确的东西（反向、行程阈值），但单布局上的收益没有迁移到其他布局。世界模型智能体从 20 次真实试验写出 16 KB 世界模型用于在线规划（每 0.16 s 重规划），达到 78%，距在真实仿真器内规划的参考 (84%) 差 3 个 episode。世界永远不够精确到可以开环抛掷（精确物理重建起始状态：17%），但只准几分之一秒就够。

### 范式的边界比预想更模糊

CaP 智能体也把诊断出的世界方程写入了自己的策略代码，区别仅在于这些方程能否在接触真实环境之前被前向运行。

---

## 局限性 (Limitations)

1. **CaP 智能体与 ASPIRE 的关系**: CaP 智能体按 [ASPIRE (arXiv 2607.00272)](https://research.nvidia.com/labs/gear/aspire/) 范式的循环运行，限制在 5 次真实试验内。这是我们的重新实现，不可与 NVIDIA GEAR 的原始系统或其论文数字比较。ASPIRE 本身使用大量真实 rollouts 进行技能发现，5 次真实试验远低于其设计预算。

2. **仿真器而非真实机器人**: 所有任务在仿真环境中进行（VizDoom, PyBullet/PyFlyt, pymunk, MuJoCo）。虽然 智能体看不到仿真器源码，但 sim-to-real gap 未被检验。

3. **结构化状态、无渲染**: 所有观测均为结构化状态向量（位置、速度等），世界模型不渲染任何画面。本文档中的所有 GIF 帧来自真实仿真器的渲染输出。

4. **Doom 不是机器人任务**: 它是一个视频游戏，物理规则由离散代码定义而非连续力学。世界模型在这种环境上的成功不直接推广到真实机器人。

5. **Doom 规则中途放宽**: 在 Doom 上，世界模型智能体的规则在实验中途被放宽，允许在世界模型中调优手写控制器（而非仅使用 PPO 等自动方法）。这后来成为了方法的正式定义。

6. **四旋翼场景经过重新设计**: 初始版本的时间限制和异常设置未能阻止出厂自动驾驶通过，最终版本被选为超出该原始控制器的能力范围。

7. **Push-T 模型家族巧合**: 世界模型和真实环境恰好使用同一类物理模型（刚体冲量法）。在真实场景中这种结构匹配需要在每个具体系统上建立。

8. **单次运行**: 每种 paradigm 在每个任务上只运行了一次。PPO 学习曲线也是单一随机种子。结果的方差未被充分评估。

9. **Mission C 继承效应**: 两位智能体在 Mission C 之前都已完成 Missions A/B，携带着已有的世界知识和策略。世界模型从 A/B 版（15466 bytes）扩展而非从零开始写。这与现实中"逐步加难"的工程流程一致，但意味着 Mission C 的绝对表现不能与从零开始做比较。

10. **CaP Mission C 的 留出 程序未飞行过**: CaP 智能体评估的 v11 是在 5 次真实试验用完后修改的，修改后从未被飞行，只通过回放日志验证。Pilot 自己标注了这一点。

11. **Mission C 观测条件**: 两位智能体的观测试验日志均包含所有其他飞行器的真实轨迹（由 环境的仪器记录），这对真实场景过于理想，真实传感器不可能提供完美的轨迹数据。

12. **状态空间低维**: 所有任务的状态空间相对简单（Doom ~20D 特征空间, 四旋翼 12D, Push-T 5D, Cable 20 个追踪点 + 力）。高维视觉观测或更复杂的接触动力学（如灵巧操作）下 世界模型 是否可行尚不清楚。

13. **依赖 LLM 能力**: 世界模型质量取决于 LLM 的物理直觉和编码能力。不同模型或同一模型的不同运行可能产生不同质量的世界。

14. **离线评估例外**: 推块 世界模型智能体在评分 CaP程序之前，在 Push-T v5 的一条 docstring 中看到了该版本的官方成绩 (10/50)。34 程序离线评估除此之外是盲的。

15. **Push-L 筛分误差**: 世界模型对 Push-L 的盲预测 MAE 为 10.1 pts，最大偏差 22 pts，远差于其它任务的 0.6–5.1 pts。世界模型在 Push-L 的 7 个版本中低估了 5 个的成功率，匹配 1 个，高估 1 个（4 pts）。

16. **CaP 以世界模型为测试环境的代价**: CaP 以世界模型为测试环境的真实 episode 消耗为 CaP 智能体的命 + 建造世界模型的命（四旋翼 C: 5 + 10；Push-L: 2 + 2，其中世界模型从 Push-T 的 4 次真实试验继承并用 2 次 Push-L 真实试验重新拟合）；CaP agent 在世界模型内零额外真实 episode。

17. **线缆：隐藏任务身份**: 线缆任务依赖规则（手册未命名物体）。事后审计三位线缆智能体的命令日志确认无违规访问。

18. **线缆共享环境**: 线缆仿真器包安装在两位智能体可访问的共享环境中。

---

## 仓库结构

```
.
├── reallab/                      # 不透明实验室（智能体不可见）
│   ├── doom/                     #   VizDoom 环境封装
│   ├── drone/                    #   PyBullet/PyFlyt 四旋翼环境
│   │   └── _site.py              #   隐藏异常：受损旋翼 + 侧风
│   ├── pusht/                    #   pymunk Push-T/Push-L 环境
│   │   └── _deviations.py        #   隐藏异常：质心偏移 + 摩擦 + 释放后滑行
│   ├── station/                  #   MuJoCo 线缆布线环境
│   └── annex/                    #   MuJoCo 绳索抛掷环境
├── worlds/                       # 世界模型（智能体从零编写, pure torch）
│   ├── fps/take_cover_world.py   #   VizDoom world model (16599 bytes)
│   ├── drone/site_world.py       #   四旋翼 world model (25975 bytes, A/B 版 15466)
│   ├── pusht/pusht_world.py      #   Push-T/L world model (14344 bytes, T+L 共用)
│   ├── station/station_world.py  #   Cable world model (13870 bytes)
│   └── annex/                    #   Rope world models
│       ├── annex_world.py        #     (14313 bytes)
│       └── annex_world_v2.py     #     (16069 bytes)
├── pilots/                       # 智能体生成的控制器和策略代码
│   ├── aspire_doom/              #   CaP Doom 策略 (v1–v5)
│   ├── aspire_drone/             #   CaP 四旋翼策略
│   ├── aspire_pusht/             #   CaP Push-T 策略 (v1–v8)
│   ├── aspire_station/           #   CaP Cable 策略
│   ├── aspire_annex/             #   CaP 绳索策略
│   └── aspire_blind/             #   对照 CaP 智能体（无日志，线缆任务）
├── runs/                         # 实验结果与审计数据
│   ├── summary_table.txt         #   主结果表
│   ├── summary.json              #   完整结果数据
│   ├── ledger_summary.txt        #   真实试验预算消耗记录
│   ├── doom/                     #   Doom 审计、GIF、RL 基线曲线
│   ├── drone/                    #   四旋翼审计、GIF、基线、neural WM 保真度
│   ├── pusht/                    #   Push-T 审计、GIF
│   ├── station/                  #   Cable 审计
│   └── annex/                    #   Rope 审计
├── cawm/                         # 基础设施代码
│   ├── ledger.py                 #   真实试验预算计数与强制执行
│   ├── results.py                #   结果记录
│   ├── ppo.py                    #   PPO 实现
│   └── mpc.py                    #   MPC 实现
├── experiments/                  # 实验脚本
│   └── audit/                    #   审计和 montage 脚本
├── tests/                        # 测试套件
├── blog/                         # 页面源文件
├── site/                         # 页面构建与静态资源
├── .github/workflows/            # GitHub Actions 部署
├── reproduce.py                  # 复现脚本（见下文）
├── requirements.txt              # 依赖列表
├── README.md                     # English README
├── README.zh.md                  # 中文 README
└── NEWS.zh.md                    # 中文项目日志
```

---

## 如何复现

安装（Linux x86\_64, Python 3.12；GPU 可选，仅线缆和绳索的世界模型控制器使用）：

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

52 s。`requirements.txt` 顶部有 `venv`/`pip` 等效命令。

测试套件：

```bash
python -m pytest tests reallab -q
```

111 个测试，约 1 分钟。

复现脚本：

```bash
python reproduce.py list                       # 列出所有已注册的 (task, controller) 及记录结果
python reproduce.py real <task> <controller>    # 在环境中重新运行
python reproduce.py world <task>                # 在世界模型中重新评估
```

任务：`doom`、`drone_A`、`drone_B`、`drone_C`、`pusht_T`、`pusht_L`、`cable`、`rope`。46 个已注册控制器。环境在给定种子下是确定性的，每次重跑都逐 episode 匹配。已验证的命令（8 workers）：

| 命令 | 复现结果 | 记录结果 | 耗时 |
|------|---------|---------|------|
| `real rope aspire_style` | 1/50 | 1/50 | 5 s |
| `real rope aspire_style_extended` | 0/50 | 0/50 | 4 s |
| `real cable aspire_style` | 50/50 | 50/50 | 10 s |
| `real cable code_wm` | 50/50 | 50/50 | 16 s |
| `real pusht_L cap_in_code_world` | 42/50 | 42/50 | 1 s |
| `real drone_C cap_in_code_world` | 95/100 | 95/100 | 8 s |
| `real doom aspire_style` | 2,384.56 game ticks | 2,384.56 | 7 s |
| `world pusht`（15 个程序，3,000 episodes） | 15 个全部与记录一致 | | 35 s |
| `world cable`（23 个程序） | 全部一致 | | 214 s |
| `real rope code_wm --episodes 4` | 1/4，逐 episode 匹配 | (全部 50 episode: 16/50) | 125 s |
| `real rope code_wm_extended --episodes 4` | 3/4，逐 episode 匹配 | (全部 50 episode: 39/50) | 257 s |

两个绳索世界模型控制器跑完全部 50 episode 分别约 25 和 50 分钟，此处仅各验证 4 episode。

---

## 模型配置

| 角色 | 模型 |
|------|------|
| 负责调度的模型（分配任务、检查隔离规则） | Claude Fable 5.2 |
| 智能体（世界模型 + CaP） | Claude Opus 5.5 |
| Labs / RL training | Claude Sonnet 5.5 (cable lab: Claude Opus 5.5) |
| 文档撰写 | Claude Opus 4.6 |

---

## 参考文献

- **Code World Models**: Dainese et al. (2024). [arXiv 2405.15383](https://arxiv.org/abs/2405.15383). LLM 写代码形式世界模型的先驱工作。
- **WorldCoder**: Tang, Key and Ellis (2024). [arXiv 2402.12275](https://arxiv.org/abs/2402.12275). LLM 生成环境代码。
- **CWM**: Meta FAIR CodeGen Team (2025). "CWM: An Open-Weights LLM for Research on Code Generation with World Models." [arXiv 2510.02387](https://arxiv.org/abs/2510.02387). 32B 开放权重语言模型，在执行轨迹和智能体交互上训练，建模代码执行。
- **Code as Worlds**: Wang et al. (2026). "Code as Worlds: Agentic Discovery of Executable World Representations for Physical Reasoning." [arXiv 2608.27549](https://arxiv.org/abs/2608.27549). 将物理场景写成可执行代码，验证后监督视觉-语言模型。
- **Code World Model: Coding Agent as World Brain**: Chen, Lin and Zhang (2026). [arXiv 2608.25927](https://arxiv.org/abs/2608.25927). 编码智能体写出保存世界状态的代码，驱动视频模型渲染。
- **Harness Engineering for Self-Improvement**: Weng, L. (2026). [Lil'Log, 4 July 2026](https://lilianweng.github.io/posts/2026-07-04-harness/). Harness 即基础模型外围的系统（规划、工具、上下文、评估）。
- **Self-Harness**: Zhang et al. (2026). "Self-Harness: Harnesses That Improve Themselves." [arXiv 2606.09498](https://arxiv.org/abs/2606.09498). LLM 智能体挖掘自身痕迹发现失败模式，对 harness 做小幅编辑并用回归测试验证。
- **RoboHarness**: Qu et al. (2026). "RoboHarness: A Simple Harness Outperforms VLA and World Action Models." [arXiv 2610.14018](https://arxiv.org/abs/2610.14018). 给 LLM 智能体视觉-几何控制面板，在 BEHAVIOR Challenge 2025 任务上对比 VLA 模型。
- **WorldGym**: Quevedo, Sharma, Sun, Suryavanshi, Liang and Yang (2025). "WorldGym: World Model as An Environment for Policy Evaluation." [arXiv 2506.00613](https://arxiv.org/abs/2506.00613). 动作条件视频世界模型，以 VLM 打分 Monte Carlo rollouts 评估策略。
- **Schwoebel (2026)**. "World-Time Compute with Verified Code World Models." [arXiv 2609.09163](https://arxiv.org/abs/2609.09163). 独立编写并验证的代码世界生成精确标注轨迹用于微调 LLM。
- **Aguilar Martín (2026)**. "When a Verified World Model Still Loses: Play-Adequacy vs Prediction-Accuracy in LLM-Synthesized Code World Models." [arXiv 2607.14169](https://arxiv.org/abs/2607.14169). LLM 写的代码世界模型即使采样转移精度完美，在关键规则的罕见错误上仍可能落败。
- **SIMPLER**: Li et al. (2024). "Evaluating Real-World Robot Manipulation Policies in Simulation." [arXiv 2405.05941](https://arxiv.org/abs/2405.05941). 操控类策略的真实世界评估难以规模化、难以重复。
- **Robot Deployment Challenges**: Kress-Gazit et al. (2024). "Formalizing the Robot Deployment Problem." [arXiv 2409.09491](https://arxiv.org/abs/2409.09491). 形式化机器人部署中的评估瓶颈。
- **Reconciling Real-to-Sim-to-Real**: Torne et al. (2024). "Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation." [arXiv 2403.03949](https://arxiv.org/abs/2403.03949). 从真实场景重建数字孪生，在仿真中微调策略后部署回真实系统。
- **Code as Policies**: Liang et al. (2022; ICRA 2023). [arXiv 2209.07753](https://arxiv.org/abs/2209.07753). LLM 为机器人生成控制代码。
- **ASPIRE**: "ASPIRE: Agentic /Skills Discovery for Robotics", NVIDIA GEAR. [arXiv 2607.00272](https://arxiv.org/abs/2607.00272). [Project page](https://research.nvidia.com/labs/gear/aspire/).
- **World Models**: Ha, D. & Schmidhuber, J. (2018). "World Models." [arXiv 1803.10122](https://arxiv.org/abs/1803.10122). (VizDoom Take Cover: 1092 ± 556 at temperature 1.15, pixel observations, 10000 random rollouts; 非同口径比较)
- **Claude Fable and Mythos 5.2**: Anthropic (2026). [anthropic.com, 9 November 2026](https://www.anthropic.com/claude-fable-and-mythos-5-2). RoboDojo Sim: Fable 5.2 64.9%, GPT-6 Astra 29.8%, Opus 5.5 39.4%.
- **GPT-6 Astra as Embodied Policies**: Chen et al. (2026). "Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies." [arXiv 2609.38537](https://arxiv.org/abs/2609.38537). 六个领域评估，结果参差。
- **Saltline**: 单文件浏览器 FPS（index.html ~1.43 MB），由 Felix Rieseberg (Anthropic) 用 Claude Fable 5.2 一次性生成。本笔记的直接灵感。

---

## 许可

MIT。见 [LICENSE](LICENSE)。

---

## 关于本文

这是一份初步的探索记录，分享出来以便讨论和纠正。如果其中有混淆或错误，我们希望被告知。欢迎提问、质疑和指正，以便一起改进。
