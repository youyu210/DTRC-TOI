# 用户指定的实验设计参考

来源：[调研TOI核心逻辑](chatgpt-conversation://6aba3534-7b88-83ee-924f-084f090b881b)，2026-09-29 读取最后一条回答。以下为参考资料，实施范围以本次用户要求和 EXPERIMENT_PLAN_V2.md 为准。

**这次结果支持把改进重点放在“减少残差验收导致的额外 Newton 迭代”，而不是继续优化补偿公式的计算开销。**不过，需要修正上一轮过于笼统的判断：残差补偿并非完全无效——在 `sphere7` 和 `bunny` 上，相对原 Graph 已分别达到 **1.688×、1.483×**；问题是它在另外三个场景上几乎没有保留提前终止的收益。:chatgpt-content-reference{index="0"}

**我的建议是：先做“TOI 容差与残差容差解耦”的最小改动实验，判断过强残差要求是不是主要阻碍；然后再决定是否删除历史补偿、改成按需验收。需要进一步超过纯 TOI 的速度时，再转向自适应 PCG 精度，而不是继续给外层退出增加限制。**

另外，这份报告明确写明：本次新增的是布料场景入口，求解数值源码仍保持 `residual-toi-graph-v1`。所以它检验的是原补偿方案在更多场景中的表现，**并没有检验上一条提出的低开销修正预算版本**。:chatgpt-content-reference{index="1"}

## 一、先看清楚：加速损失主要发生在哪里？

我根据 `comparison.csv` 重新计算了各场景的加速比、Newton 次数变化，以及“整步时间 ÷ Newton 次数”。

| 场景 | 补偿 / 原 Graph 加速比 | 纯 TOI / 原 Graph 加速比 | 补偿相对 Graph 的 Newton 次数变化 |
|---|---:|---:|---:|
| sphere7 | **1.688×** | **6.747×** | 减少 38.9% |
| bunny | **1.483×** | **8.546×** | 减少 32.5% |
| table | **0.996×** | **3.915×** | 增加 0.8% |
| stack10 | **1.017×** | **1.290×** | 减少 2.7% |
| hang | **1.019×** | **2.044×** | 减少 2.3% |

以上均为本次单次运行数据的计算结果，不是重复实验下的稳定加速结论。:chatgpt-content-reference{index="2"}

### 1. 单次迭代成本没有明显变坏，主要是退出得不够早

例如：

- `sphere7`：Graph 平均约 **20.22 ms/Newton**，补偿约 **19.61 ms/Newton**；
- `bunny`：Graph 约 **29.67 ms/Newton**，补偿约 **29.62 ms/Newton**；
- `table`：Graph 约 **40.14 ms/Newton**，补偿约 **39.99 ms/Newton**。

这些是整步时间除以迭代次数的粗略指标，包含固定开销，不能代替阶段计时；但它们没有显示出“补偿版因为每次 Newton 明显更贵而失去加速”的趋势。**优先应该解释的是为什么补偿版仍然执行了那么多次 Newton。**:chatgpt-content-reference{index="3"}

尤其是 `table`：

\[
N_{\mathrm{Graph}}=2481,\qquad
N_{\mathrm{TOI}}=631,\qquad
N_{\mathrm{comp}}=2501.
\]

纯 TOI 能够显著提前退出，但补偿版几乎恢复了原停止规则的迭代量，附加退出也只有 **2 帧**。**这使 `table` 成为下一步诊断残差门控是否过强的优先场景。**这里还不能把全部差异直接归因于同一批迭代上的拒绝，因为不同停止策略已经改变了后续轨迹。:chatgpt-content-reference{index="4"}

### 2. 五个场景合起来，补偿只保留了较小一部分时间收益

将五个场景的时间直接相加：

\[
T_{\mathrm{Graph}}=568.741\ \mathrm{s},
\]

\[
T_{\mathrm{comp}}=478.631\ \mathrm{s},
\]

\[
T_{\mathrm{TOI}}=227.818\ \mathrm{s}.
\]

对应套件总时间的加速比分别是：

\[
S_{\mathrm{comp}}=1.188\times,
\qquad
S_{\mathrm{TOI}}=2.496\times.
\]

补偿版保留了本次纯 TOI 所呈现时间节省量的约 **26.4%**。这不是等精度比较，也不是可实现加速上限，但说明**目前最值得争取的空间，是补偿版与纯 TOI 之间的差距**。:chatgpt-content-reference{index="5"}

---

## 二、首先修改的不是递推形式，而是“同一个阈值管两件事”

原方案中：

\[
z_k\geq\beta_k,\qquad z_k\geq r_k.
\]

所以采用

\[
z_k\leq10^{-3}
\]

实际上同时要求：

\[
\beta_k\leq10^{-3},
\qquad
r_k\leq10^{-3}.
\]

但二者含义不同：

\[
\beta_k：
\text{早期状态在累计推进中的剩余权重},
\]

\[
r_k：
\text{当前非线性残差相对参考残差的比例}.
\]

**将两者都设为 \(10^{-3}\)，是一个很强的设计选择，并不是数学上必须这样设置。**

你目前的实验中，TOI 条件本来已经允许结束，但补偿又要求非线性残差降低到较严格的相对水平，这就可能把提前终止节省的迭代重新补回来。

更重要的是：在相同状态、相同迭代路径和相同退出前置条件下，只要新条件始终比纯 TOI 更严格，就不可能比纯 TOI 更早触发。因此目标应分成两类：

**相对原 Graph 加速：**争取以更低成本控制必要的误差，而不是恢复原来的充分收敛。

**进一步超过纯 TOI：**必须减少单次迭代工作量，或调整最小迭代要求；单纯增加残差门控做不到。

---

## 三、第一步：做“双容差补偿”，这是现有代码上改动最小的验证

我建议先做这个原型，**不改 CCD、不改 PCG、不改 \(K_{\min}\)、不换残差范数，也先不改变测量频率**。这样能够单独检验“残差要求过严”这一个因素。

### 1. 将两个容差分开

保留：

\[
\varepsilon_{\mathrm{TOI}}=10^{-3},
\]

另外引入独立的残差容差：

\[
\varepsilon_{\mathrm R}.
\]

例如第一轮测试：

\[
\varepsilon_{\mathrm R}\in
\{10^{-2},\,3\times10^{-2},\,10^{-1}\}.
\]

这些只是用于探索的参数点，不是已经证明合适的精度设置。

### 2. 保留原超额残差，只改变它对应的预算尺度

仍定义：

\[
q_k=1-\alpha_k,
\]

\[
d_k=[r_{k+1}-q_kr_k]_+.
\]

取一个固定系数：

\[
\boxed{
\lambda=
\frac{\varepsilon_{\mathrm{TOI}}}
{\varepsilon_{\mathrm R}},
\qquad
\varepsilon_{\mathrm R}\geq\varepsilon_{\mathrm{TOI}}>0.
}
\]

然后改成：

\[
\boxed{
z^{(\lambda)}_{k+1}
=
q_kz^{(\lambda)}_k+\lambda d_k,
\qquad
z^{(\lambda)}_0=1.
}
\]

退出条件仍然是：

\[
\boxed{
z^{(\lambda)}_k\leq\varepsilon_{\mathrm{TOI}}.
}
\]

**这不是在保留旧证明的同时随意缩小补偿项，而是明确放宽残差目标，并相应改变能够承诺的界。**

### 3. 修改后的理论仍然说得清楚

下面是基于原递推的直接推导。

只要初始化满足 \(r_0\leq1\)，并且 \(\lambda\) 在整个累计过程中保持固定，就有：

\[
z^{(\lambda)}_k\geq\beta_k,
\]

以及

\[
z^{(\lambda)}_k\geq\lambda r_k.
\]

因此：

\[
\boxed{
z^{(\lambda)}_k\leq\varepsilon_{\mathrm{TOI}}
\Longrightarrow
\begin{cases}
\beta_k\leq\varepsilon_{\mathrm{TOI}},\\[1mm]
r_k\leq\varepsilon_{\mathrm R}.
\end{cases}
}
\]

现在，累计推进和残差精度可以独立设置。

还可以得到：

\[
\boxed{
z^{(\lambda)}_k
=
\beta_k+\lambda\left(z^{(1)}_k-\beta_k\right).
}
\]

所以对 \(0\leq\lambda\leq1\)：

\[
\boxed{
\beta_k
\leq
z^{(\lambda)}_k
\leq
z^{(1)}_k.
}
\]

这说明：**沿同一迭代路径，它不会比原补偿条件更严格，但也不会比纯 TOI 更宽松。**

这正好提供一个受控的中间区间，而不是从“很严格的补偿”直接跳到“完全不检查残差”。

注意，这仍然是非线性相对残差界，**不是位置误差界或速度误差界**；通过其他原有退出路径结束的状态，也不能自动套用这个附加分支的结论。

### 4. 代码修改位置

下面只表示需要替换的逻辑，不是针对本地源码验证过的完整补丁：

```cpp
// 配置条件：0 < eps_toi <= eps_residual。
// residual_weight 在一次累计过程中保持固定。
const double residual_weight = eps_toi / eps_residual;

// 保留现有的：
// 最终 accepted alpha、新旧真实残差、归一化参考、
// Kmin、边界约束及非有限数值检查。

const double q = 1.0 - alpha_accepted;

const double excess =
    std::max(0.0, residual_new - q * residual_old);

// 原来：budget = q * budget + excess;
budget = q * budget + residual_weight * excess;

if (original_exit_guards &&
    budget_valid &&
    beta <= eps_toi &&
    budget <= eps_toi) {
    break;
}
```

这里的 `residual_old/new` 指已经按同一个固定参考归一化的真实非线性残差。

**第一轮不要同时改 \(K_{\min}\) 或 PCG 容差。**否则即使加速了，也无法判断到底是哪项修改生效。

### 5. 必须配一个更简单的对照

同时测试：

\[
\boxed{
\beta_k\leq\varepsilon_{\mathrm{TOI}}
\quad\land\quad
r_k\leq\varepsilon_{\mathrm R}.
}
\]

也就是**相同双容差的当前残差门控**。

如果这个简单门控在相近轨迹质量下更快，就删除历史补偿。不要为了保留公式而维护一个没有独立收益的历史状态。

我将“双容差补偿”定位为**最快能够验证问题来源的工程原型**；单独“补偿项前乘一个系数”并不足以成为有说服力的论文核心贡献。

---

## 四、第二步：将“每轮都测残差”改为“需要验收时才测”

第一步解决的是退出条件太严格；第二步解决验收成本。

但这里必须区分两种实现，不能混在一起。

### 1. 保留历史递推，就不能直接跳过中间残差

原公式需要每轮的

\[
r_k,\quad r_{k+1},\quad d_k.
\]

如果中间几轮不测残差，却仍声称计算的是原来的 \(z_k\)，公式与实现就不一致，其上界证明也不能直接沿用。

因此：

> **保留历史补偿，就保留它需要的完整输入；想减少残差测量，就优先改成当前状态验收。**

### 2. 当前残差门控可以按需执行

建议流程：

```text
正常 Newton–PCG、CCD、线搜索
                ↓
接受位置更新，更新 beta
                ↓
beta 尚未达到阈值？
    是：继续，不专门进行附加残差验收
    否：评估当前状态的真实非线性残差
                ↓
满足独立残差容差？
    是：退出
    否：复用该梯度，继续后续求解
```

具体实现要检查：能否在构造下一轮 Hessian 或执行下一次 PCG **之前**完成验收；不能为了“检查是否可以省掉下一轮”，先把下一轮最贵的部分全部算完。

如果 \(\beta\) 已经为零而残差仍未达标，后面可能每轮都要验收，所以收益未必很大。这仍需阶段计时，而不能宣称“按需测量必然显著加速”。

### 3. 另一条低成本路线：复用未截断 Newton 修正量

若原停止条件已经提供未截断方向范数，可以测试：

\[
\boxed{
\beta_{k+1}\leq\varepsilon_{\mathrm{TOI}}
\quad\land\quad
\frac{\|p_k\|_\infty}{h}
\leq
s_{\mathrm{corr}}\varepsilon_{\mathrm N}.
}
\]

它用现成的修正幅度替代额外的真实残差验收，**不要再与严格残差补偿同时取 AND，否则只是又增加了一道限制**。

必须使用未乘 \(\alpha_k\) 的 \(p_k\)，避免 CCD 把步长压小后造成“更新很小，所以已经收敛”的误判。它检查的是本轮方向所反映的修正尺度，不是更新后重新评估的梯度，理论表述也应相应改变。

使用修正量来控制非线性求解精度有成熟数值软件依据：CVODE 的停止测试就结合修正量、估计收敛速率与时间积分误差要求。不过，这不能直接给 StiffGIPC 的近似 Hessian 和接触更新提供无条件误差保证。:chatgpt-content-reference{index="6"}

**建议先完成双容差对照，再决定是否换度量，避免同时改变验收强度、验收量和验收时机。**

---

## 五、需要进一步提高速度时：把残差用于“少解 PCG”，而不是“多做 Newton”

这是后续比继续强化外层补偿更有直接加速作用的方向：

> **外层还在明显变化时，不要把当前线性系统解得过于精确；接近退出或出现异常时，再恢复严格 PCG。**

### 1. 数学来源：不精确 Newton 与 Eisenstat–Walker forcing term

不精确 Newton 不要求每次精确求解 Newton 方程，而是控制线性残差：

\[
\boxed{
\|B_kp_k+g_k\|
\leq
\eta_k\|g_k\|.
}
\]

Eisenstat–Walker 方法根据非线性求解进展调整 \(\eta_k\)，避免内层过度求解；PETSc 也提供了对应实现，按 Newton 迭代动态调整线性求解相对容差。:chatgpt-content-reference{index="7"}

针对你的代码，可以先用简化的、自适应 forcing 原型：

\[
\boxed{
\eta_k
=
\operatorname{clip}
\left(
\gamma
\left(\frac{R_k}{R_{k-1}}\right)^p,
\eta_{\min},
\eta_{\max}
\right).
}
\]

这里 \(R_k\) 是同一范数、同一缩放下的当前非线性残差；\(\eta_{\min}\) 保持现有严格 PCG 设置，\(\eta_{\max}\) 允许早期更宽松的线性求解。这是借鉴该框架的原型，不能把简化公式直接称为完整 Eisenstat–Walker 实现。

### 2. 怎么实施才不容易得不偿失？

第一轮只改变 PCG 的停止容差，不改预条件器、不改 CCD，也不改 Graph 分段结构。

从较小的放宽开始：**如果现有容差是 \(10^{-4}\)，先试上限 \(10^{-3}\)，再考虑 \(10^{-2}\)**；报告没有给出本次 PCG 容差的具体范数定义，不能直接把这些数值填进一个含义不同的变量。

接近候选退出、出现非下降方向、非有限值或明显线搜索困难时，恢复原严格容差。对同一系统重试时应复用可复用的矩阵和预条件器，而不是无条件重新组装。

尤其需要核对：

\[
\|r_{\mathrm{lin}}\|_2
\]

与

\[
r_{\mathrm{lin}}^TM^{-1}r_{\mathrm{lin}}
\]

不是同一个停止量。如果当前 PCG 比较的是平方量，容差的换算也必须对应。

在 Graph 实现中，要保证新容差被实际执行的节点读取，而不是只修改了没有生效的主机变量。CUDA 支持在适用条件下更新已实例化图的节点参数，不必仅为容差变化重新捕获整张图。:chatgpt-content-reference{index="8"}

### 3. 为什么还不能承诺一定更快？

因为更松的 PCG 可能导致方向质量下降，增加 Newton 次数或线搜索次数。最终比较应看：

\[
T_{\mathrm{total}},
\]

而不能只看

\[
N_{\mathrm{PCG}}\downarrow.
\]

此外，StiffGIPC 使用的 \(B_k\) 可以不同于真实 Jacobian/Hessian，经典不精确 Newton 的收敛结论不能不加条件地直接套用。

**这一方向的优势是它真正试图减少每轮求解工作，而不是通过新增外层限制增加工作。**

---

## 六、`stack10` 应该单独走性能分析路线

`stack10` 与另外几个场景不同：纯 TOI 相对 Graph 也只有 **1.290×**，但已经占本次纯 TOI 五场景总耗时的约 **73.5%**。因此，一旦前面几个场景恢复了提前终止收益，`stack10` 会成为整套测试的主要耗时来源。:chatgpt-content-reference{index="9"}

从汇总计算，它约有：

\[
T/N_{\mathrm{Newton}}
\approx242\text{–}265\ \mathrm{ms},
\]

明显高于本次另外几个场景；但是**这不能直接证明 CCD 是瓶颈**。更大的系统、接触数量、矩阵运算和线搜索都可能造成这个现象。当前 CSV 没有阶段分解。:chatgpt-content-reference{index="10"}

下一轮至少拆出：

```text
gradient_hessian
pcg_setup_and_solve
ccd
line_search
residual_check
graph_prepare_update
other
```

然后按耗时占比决定优化位置。

例如，假设 PCG 只占 20%，即使把 PCG 加速两倍，理论上总加速也只有：

\[
S=
\frac{1}{0.8+0.2/2}
\approx1.11.
\]

这里的 20% 只是说明方法的假设例子，不是从这份报告测出的比例。

**所以 `stack10` 不能因为“PCG 有很多次调用”就默认继续优化 PCG，也不能因为是多层布料就默认认定 CCD 最贵。**

---

## 七、下一轮实验这样安排，才能快速判断方向是否有效

### 第一轮：只检验“残差要求过严”

优先用 `table` 和 `bunny` 做诊断：前者补偿基本没有提前退出收益，后者有较大的速度差距；最终仍需回到全部五个场景验证，不能只报告有利场景。:chatgpt-content-reference{index="11"}

固定原有 \(K_{\min}=6\)、\(\varepsilon_{\mathrm{TOI}}=10^{-3}\)、Graph 和 PCG 设置：

| 版本 | 修改内容 | 要回答的问题 |
|---|---|---|
| 纯 TOI | 不变 | 当前可达到的低成本参照 |
| 原补偿 | 不变 | 现有方案基线 |
| 双容差补偿 | 只改变 \(\varepsilon_{\mathrm R}\) | 放宽残差目标能否恢复提前终止 |
| 双容差当前门控 | 相同两容差，不保留历史 | 历史补偿到底是否值得保留 |

每次只测试一个全局参数配置，不按场景名称硬编码阈值。

### 第二轮：记录“TOI 已经可以停以后，还花了多少时间”

这比仅统计附加退出帧数更有帮助：

\[
k_{\mathrm{ready}}
=
\text{首次满足 TOI 及其前置守卫的迭代号},
\]

\[
k_{\mathrm{exit}}
=
\text{实际退出迭代号}.
\]

重点统计：

\[
\boxed{
N_{\mathrm{tail}}
=
k_{\mathrm{exit}}-k_{\mathrm{ready}},
\qquad
T_{\mathrm{tail}}
=
\text{候选退出之后继续求解的时间}.
}
\]

同时记录拒绝原因：当前残差未达标、历史预算未达标、边界条件未满足，或其他守卫未通过。

如果双容差让 \(N_{\mathrm{tail}}\) 明显下降，并且总时间也下降，说明找对了主要方向；如果次数下降而时间不降，就继续查阶段成本和额外验收，而不是再改阈值。

### 第三轮：正确比较质量，再决定是否接受加速

这次报告没有严格收敛参考，位置 RMS 是相对 `base` 的轨迹差异，而且包含固定障碍顶点。`bunny` 的 Graph 本身相对 base 已有 **16.561 mm** 的最大帧 RMS，因此不能把补偿版的 **24.483 mm** 全部解释成残差补偿导致的误差，也不能简单把两个数相减。:chatgpt-content-reference{index="12"} :chatgpt-content-reference{index="13"}

下一轮应补充**仅布料/自由顶点的位置与速度指标**，并从相同完整状态出发做单时间步参考求解，用来隔离停止准则影响；再做完整轨迹测试。当前所有相交标志为零是有用的诊断结果，但报告明确说明它不覆盖全面布料自交、穿透和退化，不能据此宣称全部几何安全。:chatgpt-content-reference{index="14"}

另外，当前是一次运行、`warmup=0`。下一轮需要重复运行，并把冷启动与稳定运行成本区分报告；若为补偿审计增加的残差测量也在 `base/graph/toi` 中执行，应另做生产路径计时，避免让不需要该测量的方法承担额外观测成本。:chatgpt-content-reference{index="15"}

---

### 最终建议

**当前最优先的修改是：保持 TOI 推进要求不变，把残差容差独立出来，降低其对提前退出的限制强度。先用最小代码改动验证，再用同容差的简单门控决定是否保留历史补偿。**

之后，速度路线按以下顺序推进：

\[
\boxed{
\text{减少不必要的 Newton 尾部迭代}
\;\rightarrow\;
\text{减少附加验收成本}
\;\rightarrow\;
\text{自适应降低早期 PCG 精度}
}
\]

`stack10` 同时单独做阶段计时。**不要先降低 \(K_{\min}\)**：本次前三个纯 TOI 场景已接近平均每帧 6 次 Newton，而其轨迹差异也明显增大；在尚未校准质量要求前，继续降低最小迭代数不是第一选择。:chatgpt-content-reference{index="16"} :chatgpt-content-reference{index="17"}
