# SubtypeGrouped 模型参数默认值说明

本文档说明 `AgeSexSubtypeGroupedHPVModel` 中亚型组特异参数的当前默认值、设定逻辑和引用来源。

## 1. 当前参数

每个高危亚型组现在保留 3 个参数：

- `initial_weight`
  - 用于把初始感染状态 `If/Pf/LC/RC/DC/Rf` 的总量拆分到各亚型组
- `persistence_multiplier`
  - 乘在 `beta_I` 上，表示 `If -> Pf` 这一步形成持续感染的相对倍率
- `cancer_progression_multiplier`
  - 乘在 `beta_P / beta_LC / beta_RC` 上，表示 `Pf -> LC -> RC -> DC` 这一整条癌症链的相对倍率

默认亚型组为：

- `hr_16_18`
- `hr_31_33_45_52_58`
- `hr_other`

## 2. 为什么不再设置 `infection_weight`

在当前 subtype 模型中，每个亚型组已经有自己独立的感染者池：

- 女性：`If__group / Pf__group`
- 男性：`Im__group / Pm__group`

因此易感者接触到哪个亚型组的感染者，就应通过该组自己的感染压力进入对应组的感染状态。这里不需要再额外乘一个“新发感染权重”。

如果额外再设 `infection_weight`，会把亚型组分布重复压到传播项中，导致总传播力被不必要地缩小。因此当前实现中：

- 保留 `initial_weight`
- 删除 `infection_weight`
- 新发感染仅由各亚型组感染者自身的感染压力决定

## 3. 当前默认数值

当前默认值写在 [model.py](/home/rongzw/projects/hpv_tdm/src/hpv_tdm/config/model.py)：

| 亚型组 | initial_weight | persistence_multiplier | cancer_progression_multiplier |
| --- | ---: | ---: | ---: |
| `hr_16_18` | 0.147 | 2.616 | 1.768 |
| `hr_31_33_45_52_58` | 0.468 | 0.931 | 0.492 |
| `hr_other` | 0.385 | 0.467 | 0.587 |

这些值的原则是：

- `initial_weight` 使用全国中国口径
- `multiplier` 使用更细的 `CIN1/CIN2/CIN3/宫颈癌` 型别分布数据构造
- 两类 `multiplier` 都按“前一阶段分布”做加权归一化，使其加权平均为 `1`

## 4. `initial_weight` 的设定逻辑

### 4.1 为什么使用全国中国口径

单地区研究容易把亚型参数过度绑定到某一地区。对于面向“全国中国”或“东亚背景”的基础模型，默认值更适合使用全国汇总数据。

因此当前默认值优先使用：

- 杜克昆山网页中的中国系统综述/荟萃分析总结
- BMC Medicine 2025 全国大样本研究

### 4.2 具体计算

BMC Medicine 2025 研究纳入了 2017 到 2023 年全国 31 个省级区域、2728321 名女性。其表 1 给出了 14 个高危型别的型别流行率。按当前三组合并：

- `hr_16_18`
  - `HPV16 + HPV18 = 2.15 + 0.74 = 2.89`
- `hr_31_33_45_52_58`
  - `HPV31 + HPV33 + HPV45 + HPV52 + HPV58`
  - `= 0.83 + 0.91 + 0.41 + 4.40 + 2.65 = 9.20`
- `hr_other`
  - `HPV35 + HPV39 + HPV51 + HPV56 + HPV59 + HPV66 + HPV68`
  - `= 0.41 + 1.55 + 1.59 + 1.04 + 0.80 + 0.72 + 1.45 = 7.56`

三组合计：

- `19.65`

归一化后：

- `hr_16_18 = 2.89 / 19.65 = 0.1471`
- `hr_31_33_45_52_58 = 9.20 / 19.65 = 0.4682`
- `hr_other = 7.56 / 19.65 = 0.3847`

因此当前默认：

- `initial_weight = [0.147, 0.468, 0.385]`

参考来源：

- Duke Kunshan VaxLab, *我国 HPV 感染的流行情况*  
  https://vaxlab.dukekunshan.edu.cn/evidence-db-expert/hpv-vaccine-policy-advocacy-evidence-repository/the-prevalence-of-hpv-infection-in-our-country/
- BMC Medicine 2025, *Prevalence, trends, and geographic distribution of human papillomavirus infection in Chinese women*  
  https://link.springer.com/article/10.1186/s12916-025-03975-6

## 5. `CIN1/CIN2/CIN3` 与模型状态的关系

这里先明确一个重要点：`CIN1/CIN2/CIN3` 不能和当前模型状态一一对应。

- `I`
  - 更像“初始感染”
- `P`
  - 更像“持续感染到癌前病变这一整段过程的压缩状态”
- `LC / RC / DC`
  - 是浸润性宫颈癌分期
  - 并不是 `CIN1/2/3`

因此：

- `CIN1/2/3` 不能直接映射到 `LC/RC/DC`
- 也不能说 `P = CIN2` 或 `P = CIN3`
- 更合理的理解是：`P` 代表一个“前驱病变综合态”

这也是为什么这里需要先把 `CIN1/CIN2/CIN3` 压缩成一个 precursor 分布，再用它构造 multiplier。

## 6. `persistence_multiplier` 的设定逻辑

`persistence_multiplier` 只作用在：

- `If -> Pf`

它表示不同亚型组从初始感染走向持续感染/前驱病变综合态的相对快慢。

### 6.1 中国分病变级别型别分布

你提供的表格按当前三组合并后，可得到：

- `CIN1 = [0.2784, 0.4694, 0.2521]`
- `CIN2 = [0.3716, 0.4507, 0.1777]`
- `CIN3 = [0.5043, 0.3869, 0.1088]`

### 6.2 按等权平均构造 precursor 分布

按你的建议，对 `CIN1/CIN2/CIN3` 采用 `1/3` 等权平均：

- `precursor = (CIN1 + CIN2 + CIN3) / 3`

得到：

- `precursor = [0.3848, 0.4357, 0.1795]`

### 6.3 用 `precursor / infection` 构造相对风险

用 infection 分布：

- `[0.1471, 0.4682, 0.3847]`

得到：

- `hr_16_18 = 0.3848 / 0.1471 = 2.616`
- `hr_31_33_45_52_58 = 0.4357 / 0.4682 = 0.931`
- `hr_other = 0.1795 / 0.3847 = 0.467`

这组值以 infection 分布加权后的平均值为 `1`。因此它们表示：

- 各亚型组相对 aggregate 基线的 `If -> Pf` 阶段特异倍率
- 但不会整体改变这一阶段的 aggregate 平均强度

因此当前默认：

- `persistence_multiplier = [2.616, 0.931, 0.467]`

## 7. `cancer_progression_multiplier` 的设定逻辑

`cancer_progression_multiplier` 作用在：

- `Pf -> LC`
- `LC -> RC`
- `RC -> DC`

它表示一个亚型组从 precursor 综合态走向浸润癌，并沿癌症链继续进展的相对强度。

### 7.1 宫颈癌组别分布

你提供的表格按当前三组合并后：

- `hr_16_18 = 59.6 + 18.5 = 78.1`
- `hr_31_33_45_52_58 = 8.8 + 6.4 + 4.4 + 3.2 + 1.8 = 24.6`
- `hr_other = 3.0 + 2.0 + 1.9 + 1.8 + 1.4 + 1.1 + 0.9 = 12.1`

归一化后：

- `cancer = [0.6803, 0.2143, 0.1054]`

### 7.2 用 `cancer / precursor` 构造相对风险

用：

- `cancer = [0.6803, 0.2143, 0.1054]`
- `precursor = [0.3848, 0.4357, 0.1795]`

得到：

- `hr_16_18 = 0.6803 / 0.3848 = 1.768`
- `hr_31_33_45_52_58 = 0.2143 / 0.4357 = 0.492`
- `hr_other = 0.1054 / 0.1795 = 0.587`

这组值以 precursor 分布加权后的平均值为 `1`。因此它们表示：

- 各亚型组相对 aggregate 基线的癌症链阶段特异倍率
- 但不会整体改变 `Pf -> LC -> RC -> DC` 这条链的 aggregate 平均强度

因此当前默认：

- `cancer_progression_multiplier = [1.768, 0.492, 0.587]`

## 8. 为什么 `group_protection` 仍然默认 0/1

`SubtypeGrouped` 模型中的 `group_protection` 当前默认表示：

- 疫苗覆盖到的亚型组：`1`
- 疫苗未覆盖到的亚型组：`0`

也就是说，它在当前模型里被解释为“对已接种且原本易感的人，是否完全阻断该亚型组感染”。这是一个结构假设，而不是在宣称现实中已经证明了永久、100% 的型别特异性感染阻断。

这样做的原因是：

- 官方资料普遍支持 HPV 疫苗具有长期保护，目前没有明确的保护衰减证据；
- 原项目中的 `0.691` 和 `0.921` 更接近“覆盖型别对宫颈癌归因的总体占比”，并不适合继续直接解释为 subtype 模型里的型别特异性保护效力。

参考来源：

- CDC, *Impact of the HPV Vaccine*  
  https://www.cdc.gov/hpv/vaccination-impact/

## 9. 后续如果要替换默认值，建议顺序

1. 用本地 HPV 型别流行数据重估 `initial_weight`
2. 收集本地 `CIN1/CIN2/CIN3` 型别分布
3. 按 `precursor / infection` 重估 `persistence_multiplier`
4. 按 `cancer / precursor` 重估 `cancer_progression_multiplier`
5. 对两类 multiplier 分别按其前一阶段分布做加权归一化，使平均值为 `1`

## 10. 当前默认值的逻辑摘要

- 默认只保留 `initial_weight`
- 新发感染不再额外乘一个 `infection_weight`
- 用 `CIN1/CIN2/CIN3` 等权平均构造 precursor 分布
- 用 `precursor / infection` 构造 `persistence_multiplier`
- 用 `cancer / precursor` 构造 `cancer_progression_multiplier`
- 两类 multiplier 都按阶段内分布规范化到加权平均为 `1`
