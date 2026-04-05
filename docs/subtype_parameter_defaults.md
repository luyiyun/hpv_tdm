# SubtypeGrouped 模型参数默认值说明

本文档说明 `AgeSexSubtypeGroupedHPVModel` 中新增亚型参数的当前默认值、设定逻辑和引用来源。

## 1. 当前拆分后的参数

每个高危亚型组现在都有 4 个独立参数：

- `initial_weight`
  - 用于把初始感染状态 `If/Pf/LC/RC/DC/Rf` 的总量拆分到各亚型组
- `infection_weight`
  - 用于把新发感染风险拆分到各亚型组
- `persistence_multiplier`
  - 乘在 `beta_I` 上，表示 `If -> Pf` 这一步持续感染形成速度的相对倍率
- `cancer_progression_multiplier`
  - 乘在 `beta_P / beta_LC / beta_RC` 上，表示从持续感染进入癌症链及其后续进展的相对倍率

默认亚型组仍为：

- `hr_16_18`
- `hr_31_33_45_52_58`
- `hr_other`

## 2. 当前默认数值

当前默认值写在 [model.py](/home/rongzw/projects/hpv_tdm/src/hpv_tdm/config/model.py)：

| 亚型组 | initial_weight | infection_weight | persistence_multiplier | cancer_progression_multiplier |
| --- | ---: | ---: | ---: | ---: |
| `hr_16_18` | 0.147 | 0.147 | 2.62 | 1.77 |
| `hr_31_33_45_52_58` | 0.468 | 0.468 | 0.93 | 0.49 |
| `hr_other` | 0.385 | 0.385 | 0.47 | 0.59 |

这些值的原则是：

- `initial_weight` / `infection_weight` 使用全国中国口径
- `multiplier` 使用更细的 `CIN1/CIN2/CIN3/宫颈癌` 型别分布数据构造

## 3. 为什么 `group_protection` 仍然默认 0/1

`SubtypeGrouped` 模型中的 `group_protection` 现在默认表示：

- 疫苗覆盖到的亚型组：`1`
- 疫苗未覆盖到的亚型组：`0`

也就是说，它在当前模型里被解释为“对已接种且原本易感的人，是否完全阻断该亚型组感染”。这个设定是一个结构假设，而不是在宣称现实中已经证明了永久、100% 的型别特异性感染阻断。

这样做的原因是：

- 官方资料普遍支持 HPV 疫苗具有长期保护，目前并没有明确的保护衰减证据；
- 原项目中的 `0.691` 和 `0.921` 更接近“覆盖型别对宫颈癌归因的总体占比”，并不适合继续直接解释为 subtype 模型里的型别特异性保护效力。

参考来源：

- CDC, *Impact of the HPV Vaccine*  
  https://www.cdc.gov/hpv/vaccination-impact/

## 4. `initial_weight` 和 `infection_weight` 的设定逻辑

### 4.1 为什么使用全国中国口径

之前如果采用单地区研究，容易把 subtype 参数过度绑定到某一地区。对于面向“全国中国”或“东亚背景”的基础模型，这种做法代表性不够。

因此当前默认值优先使用：

- 你提供的杜克昆山网页中的中国系统综述/荟萃分析总结
- 你提供的 BMC Medicine 2025 全国大样本研究

### 4.2 具体计算过程

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

- `hr_16_18 = 2.89 / 19.65 = 0.147`
- `hr_31_33_45_52_58 = 9.20 / 19.65 = 0.468`
- `hr_other = 7.56 / 19.65 = 0.385`

因此当前设置：

- `initial_weight = infection_weight = [0.147, 0.468, 0.385]`

### 4.3 为什么这里用“型别流行率”而不是“阳性人数”

BMC Medicine 2025 的表 1 中，不同型别的检测样本量并不完全相同。尤其是 HPV16/18 的检测总人数和另外 12 个高危型别的检测总人数存在差异。因此，如果直接把“阳性人数”加总，会引入偏差。

所以这里采用的是：

- 每个型别的流行率
- 三组合并
- 再归一化

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

这也是为什么这里需要把 `CIN1/CIN2/CIN3` 先压缩成一个 precursor 分布，再用它去构造 multiplier。

## 6. `persistence_multiplier` 的设定逻辑

`persistence_multiplier` 只作用在：

- `If -> Pf`

它表示不同亚型组从初始感染走向持续感染/前驱病变综合态的相对快慢。

### 6.1 使用更细的病变分级数据

你提供的表格给出了中国不同病变级别中的型别分布，我们按当前三组合并后，先分别得到各病变级别的组内分布。

#### CIN1

原始合并值：

- `hr_16_18 = 25.2 + 7.6 = 32.8`
- `hr_31_33_45_52_58 = 15.6 + 20.8 + 10.5 + 7.0 + 1.4 = 55.3`
- `hr_other = 3.7 + 5.4 + 4.5 + 3.8 + 4.6 + 2.0 + 5.7 = 29.7`

归一化后：

- `CIN1 = [0.278, 0.469, 0.252]`

#### CIN2

原始合并值：

- `hr_16_18 = 38.8 + 9.1 = 47.9`
- `hr_31_33_45_52_58 = 20.5 + 16.5 + 10.8 + 9.1 + 1.2 = 58.1`
- `hr_other = 3.7 + 5.1 + 4.6 + 2.9 + 1.7 + 2.5 + 2.4 = 22.9`

归一化后：

- `CIN2 = [0.372, 0.451, 0.178]`

#### CIN3

原始合并值：

- `hr_16_18 = 51.8 + 6.6 = 58.4`
- `hr_31_33_45_52_58 = 15.8 + 11.4 + 9.6 + 6.3 + 1.7 = 44.8`
- `hr_other = 1.4 + 2.1 + 3.0 + 2.1 + 1.6 + 1.1 + 1.3 = 12.6`

归一化后：

- `CIN3 = [0.504, 0.387, 0.109]`

### 6.2 按你的建议使用等权平均

你建议 `CIN1/CIN2/CIN3` 全部采用 `1/3` 等权平均。于是 precursor 综合分布定义为：

- `precursor = (CIN1 + CIN2 + CIN3) / 3`

计算得到：

- `precursor = [0.385, 0.436, 0.180]`

### 6.3 再除以 infection 分布

用 infection 分布：

- `[0.147, 0.468, 0.385]`

得到：

- `hr_16_18 = 0.385 / 0.147 = 2.62`
- `hr_31_33_45_52_58 = 0.436 / 0.468 = 0.93`
- `hr_other = 0.180 / 0.385 = 0.47`

因此当前默认：

- `persistence_multiplier = [2.62, 0.93, 0.47]`

## 7. `cancer_progression_multiplier` 的设定逻辑

`cancer_progression_multiplier` 作用在：

- `Pf -> LC`
- `LC -> RC`
- `RC -> DC`

它表示一个亚型组从 precursor 综合态走向浸润癌，并沿癌症链继续进展的相对强度。

### 7.1 宫颈癌组别分布

你提供的表格中，宫颈癌按当前三组合并后：

- `hr_16_18 = 59.6 + 18.5 = 78.1`
- `hr_31_33_45_52_58 = 8.8 + 6.4 + 4.4 + 3.2 + 1.8 = 24.6`
- `hr_other = 3.0 + 2.0 + 1.9 + 1.8 + 1.4 + 1.1 + 0.9 = 12.1`

归一化后：

- `cancer = [0.680, 0.214, 0.105]`

### 7.2 用宫颈癌分布除以 precursor 综合分布

用：

- `cancer = [0.680, 0.214, 0.105]`
- `precursor = [0.385, 0.436, 0.180]`

得到：

- `hr_16_18 = 0.680 / 0.385 = 1.77`
- `hr_31_33_45_52_58 = 0.214 / 0.436 = 0.49`
- `hr_other = 0.105 / 0.180 = 0.59`

因此当前默认：

- `cancer_progression_multiplier = [1.77, 0.49, 0.59]`

## 8. 和 Aggregate 模型的关系

`Aggregate` 模型不区分亚型，因此仍保留：

- `aggregate_efficacy = 0.691`
- `aggregate_efficacy = 0.921`

它们在 `Aggregate` 模型里继续被解释为“约化后的总体保护效力参数”，主要用于和 subtype 模型做对照，而不是作为 subtype 模型中的型别特异性生物学效力。

## 9. 推荐怎么在本地化研究中替换这些默认值

如果研究对象是特定国家或地区，建议按以下顺序替换参数：

1. 用本地 HPV 型别流行数据重估 `initial_weight` 和 `infection_weight`
2. 用本地 `CIN1/CIN2/CIN3` 型别分布重估 precursor 综合分布
3. 按 `precursor / infection` 重估 `persistence_multiplier`
4. 按 `cancer / precursor` 重估 `cancer_progression_multiplier`
5. 对 `group_protection` 做敏感性分析，例如：
   - 覆盖组 `1.0`
   - 覆盖组 `0.9`
   - 覆盖组 `0.8`

## 10. 需要特别说明的局限

当前 subtype 模型仍然是第一版简化模型，主要局限包括：

- 不显式建模共感染
- 默认把 `initial_weight` 和 `infection_weight` 设成相同
- `P` 不是显式的 `CIN1/CIN2/CIN3`，而是 precursor 综合态
- 当前采用 `CIN1/CIN2/CIN3` 的等权平均，这是一个建模约定，不是病理学上唯一正确的压缩方式
- `group_protection=1/0` 是结构假设，建议在正式分析中配合敏感性分析一起报告
