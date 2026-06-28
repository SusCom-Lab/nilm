# 论文精读笔记：The Design, Creation, Implementation, and Study of a New Dataset Suitable for NILM

论文链接：<https://www.mdpi.com/2076-3417/15/13/7200>  
DOI：<https://doi.org/10.3390/app15137200>  
出处：Applied Sciences, 2025, 15(13), 7200  
阅读目标：理解这篇论文告诉我们 NILM 数据应该如何采集、如何组织、如何预处理、如何进入数据 pipeline。

## 0. 这篇论文一句话告诉我们什么

这篇论文告诉我们：**NILM 数据集不是简单记录几列功率值，而是一个完整工程系统**。一个可用于训练、评估和复现的 NILM 数据集，至少要包含：

```text
采集硬件
  -> 原始电气量
  -> 时间戳
  -> CSV/原始文件
  -> 数据转换器 converter
  -> 标准数据格式 HDF5
  -> YAML/metadata
  -> train/validation/test 配置
  -> 模型训练
  -> 统一指标评估
```

对广州疆海科技自采 NILM 数据来说，这篇论文最大的价值不是某一个模型，而是它把“采集-预处理-转换-评估”作为一条完整 pipeline 来设计。

## 1. Abstract 摘要部分告诉我们什么

摘要主要讲了论文的贡献：作者构建并研究了一个适合 NILM 的新数据集，同时考虑开放硬件、数据格式、时间戳、谐波信息和 NILMTK 兼容性。

摘要里对我们最重要的点有四个：

1. 数据集建设要和硬件采集系统一起设计；
2. 数据格式要能被 NILM 工具链使用，不能只停留在原始 CSV；
3. 时间戳精度很关键，论文特别强调 13 位 timestamp；
4. 除了有功功率，谐波、电压、电流等信息也可能提升 NILM 表达能力。

这说明公司自采数据时，不能只问“电表能不能导出功率”。更应该问：

```text
这个数据能不能对齐？
能不能追溯？
能不能标注设备？
能不能进入训练 pipeline？
能不能被别人复现？
```

## 2. Introduction 引言部分告诉我们什么

引言主要解释 NILM 的背景和问题。NILM 的目标是从总功率 aggregate 中分解出各个设备的用电情况，也就是：

```text
输入：整户/整柜/整条线路的总功率
输出：某个设备或多个设备的功率/开关状态
```

引言隐含了一个很重要的采集前提：如果要训练监督式 NILM 模型，必须有 ground truth，也就是设备级子表数据。

因此，自采数据不能只采：

```text
time, aggregate
```

更应该采：

```text
time, aggregate, appliance_1, appliance_2, appliance_3, ...
```

或者至少保证每个目标设备有独立的设备级功率标签。

对公司采集方案的启发：

```text
必须同时部署总表和子表。
总表负责模型输入。
子表负责训练标签。
没有子表，只能做无监督或弱监督，模型验证会很困难。
```

## 3. Section 2：NILMTK 部分告诉我们什么

论文介绍了 NILMTK。NILMTK 是 NILM 领域常见的数据处理和算法评估工具，它的意义不是“必须使用这个库”，而是提供了一套标准数据思想。

NILMTK 的核心思想包括：

1. 数据按 building、meter、appliance 组织；
2. 时间序列数据和 metadata 分开保存；
3. 数据一般转换为 HDF5；
4. metadata 用 YAML 描述；
5. 模型训练和评估可以在统一格式上进行。

这对我们自建 pipeline 的启发很直接。你们的数据目录不一定非要完全照搬 NILMTK，但一定要表达同样的语义：

```text
site_id / building_id
meter_id / channel_id
aggregate meter
appliance meter
appliance name
sampling rate
unit
start time
end time
```

如果没有 metadata，即使 CSV 里有功率值，后面也会出现这些问题：

```text
不知道某列对应哪个设备
不知道单位是 W 还是 kW
不知道采样率是多少
不知道设备是否属于总表范围
不知道数据缺失是断电还是传感器故障
```

## 4. Section 3：Materials and Methods 告诉我们什么

这是最值得精读的部分。它讲的是论文如何采集数据、用什么硬件、生成什么数据集，以及怎样把原始数据转换为 NILM 可用格式。

### 4.1 采集硬件

论文涉及三类硬件或采集平台：

| 硬件 | 作用 | 特点 |
|---|---|---|
| oZm v1 | 生成 DSUALM、DSUALMH | 单相电能质量分析仪，15.625 Hz，支持到 50 次谐波 |
| oZm v3 | 生成 DSUALM10、DSUALM10H | 三相/单相，60 kHz，高精度，支持 GPS/PPS 同步 |
| OMPM | 生成 UALM2 | ESP32 + PZEM004 模块，可扩展多通道，低成本 |

这告诉我们：NILM 采集硬件可以分两类路线。

第一类是低频、低成本路线：

```text
采 active_power 为主
采样率 1s、6s、10s 级别
适合 Seq2Point、Seq2Seq、RNN 等主流 NILM 模型
成本低，部署方便
```

第二类是高频/谐波路线：

```text
采电压、电流波形或谐波
采样率可到 kHz 甚至更高
适合事件检测、暂态特征、谐波特征识别
成本高，存储和同步压力大
```

对广州疆海科技的建议是：第一阶段先走低频稳定路线。先把总表和子表同步采好，再考虑高频谐波。

### 4.2 采集字段

论文中不同数据集字段不同，但核心字段包括：

```text
timestamp
voltage / VLN
current / A
active power / W
frequency / F
power factor / PF
harmonic features
```

其中 UALM2 的 CSV header 是：

```text
timestamp, VLN, A, W, F, PF
```

这对公司采集字段设计很有参考价值。

最低要求：

```text
timestamp, active_power_W
```

推荐要求：

```text
timestamp, voltage_V, current_A, active_power_W, frequency_Hz, power_factor
```

如果做高频或工业场景，再加：

```text
reactive_power_var, apparent_power_VA, harmonics, THD, waveform
```

### 4.3 时间戳

论文特别强调 13 位 timestamp。13 位 Unix timestamp 通常表示毫秒级时间戳。

它告诉我们：NILM 数据最怕时间轴混乱。因为训练样本本质上是：

```text
同一时间点 aggregate 对应同一时间点 appliance
```

如果总表和子表相差几秒，标签就会错位。比如水壶实际 10:00:03 开启，但标签被记录到 10:00:09，模型看到的就是错误样本。

公司采集约束应写明：

```text
所有采集设备必须使用统一时间源。
推荐 NTP，同一网关统一授时。
高频系统可考虑 GPS/PPS。
原始数据保留毫秒级 timestamp。
训练时可重采样到 1s、6s 或 10s。
```

### 4.4 Converter 数据转换器

论文的一个重要贡献是 converter。converter 的作用是把原始数据转换成 NILMTK 可用的数据格式。

它告诉我们：预处理不应该靠手工 Excel，也不应该每次临时写脚本。应该有固定转换程序。

标准 converter 应该做这些事：

```text
读取 raw CSV/raw files
解析 timestamp
统一字段名
统一单位
排序
去重
重采样
处理缺失值
对齐 aggregate 与 appliance
写出 clean data
写出 metadata
生成质量报告
```

对你们当前项目来说，这个思想和你们已有的 pipeline 很一致：

```text
DataSeparator
-> validation
-> repair
-> export
```

可以把论文作为你们 pipeline 合理性的文献支撑。

## 5. Section 4：Results 结果部分告诉我们什么

结果部分不是只看模型分数，而是看作者如何证明新数据集可用于 NILM 研究。

论文使用不同 NILM 算法进行实验，包括传统方法和深度学习方法，例如：

```text
CO
FHMM
Mean
Hart85
DAE
RNN
Seq2Point
Seq2Seq
WindowGRU
```

这告诉我们：一个 NILM 数据集是否有价值，不只是看它有没有数据文件，还要看它能否跑通多个模型和统一指标。

对公司自采数据来说，采完之后至少要做一个 baseline 验证：

```text
用 aggregate 预测 appliance
跑一个简单模型，比如 Seq2Point 或 Mean baseline
输出 MAE、RMSE、F1-score 等指标
检查模型是否能学到有效信号
```

如果模型完全学不到，常见原因不一定是模型差，而可能是：

```text
aggregate 与 appliance 没有对齐
设备不在总表覆盖范围内
采样率太低
标签列错误
单位混乱
缺失值太多
设备开关样本太少
```

## 6. 论文里的评估指标告诉我们什么

论文使用或讨论了多种 NILM 评估指标。它们可以分为两类。

### 6.1 功率回归指标

用于评估预测功率值准不准：

```text
MAE
RMSE
NDE
EAE
MNEAP
```

一般来说：

```text
MAE 越小越好
RMSE 越小越好
NDE 越小越好
EAE 越小越好
MNEAP 越小越好
```

### 6.2 状态识别指标

用于评估设备开关状态识别：

```text
Precision
Recall
F1-score
```

一般来说：

```text
Precision 越大越好
Recall 越大越好
F1-score 越大越好
```

对公司项目来说，建议同时保留两类指标。因为 NILM 不只是“功率预测”，有时业务更关心“设备有没有开启”。

## 7. Appendix 附录告诉我们什么

论文附录提供了更偏工程化的内容，比如 converter、创建数据集、训练模型、计算指标的示例。

这对我们很重要，因为它说明论文不是停留在概念，而是试图让数据处理流程可复现。

对公司 pipeline 的启发：

```text
每一步都应该脚本化。
每个数据版本都应该可追溯。
每次清洗应该有配置文件。
每次导出应该有日志和质量报告。
```

不要让数据处理变成：

```text
某个人手动改了 CSV，但没人知道改了什么。
```

## 8. 这篇论文对数据采集约束的直接启发

如果把这篇论文转成公司采集规范，可以写成下面这些约束。

### 8.1 采集对象约束

```text
必须采 aggregate 总表。
目标设备必须采 appliance-level 子表。
appliance 子表必须属于 aggregate 覆盖范围。
每个 site/building 至少记录一个 aggregate 通道。
每个目标 appliance 至少记录一个独立通道。
```

### 8.2 时间约束

```text
所有通道使用统一时间源。
原始 timestamp 保留毫秒级或至少秒级。
所有通道 timestamp 必须可解析、递增、可排序。
不允许大量重复 timestamp。
训练前必须重采样到统一采样周期。
```

### 8.3 字段约束

最低字段：

```text
timestamp, active_power_W
```

推荐字段：

```text
timestamp, voltage_V, current_A, active_power_W, frequency_Hz, power_factor
```

扩展字段：

```text
reactive_power_var, apparent_power_VA, harmonics, THD
```

### 8.4 质量约束

```text
功率单位必须统一为 W。
电压单位统一为 V。
电流单位统一为 A。
功率字段必须是数值型。
不能混入字符串、布尔值、空白字符。
负功率、异常尖峰、长时间全 0、长时间不变都要标记。
```

### 8.5 元数据约束

每个采集点至少要记录：

```text
site_id
building_id / house_id
meter_id
channel_id
is_aggregate
appliance_name
location
rated_power_W
sampling_rate
unit
timezone
install_time
remove_time
```

### 8.6 数据留存约束

```text
raw data 必须保留，不允许覆盖。
clean data 单独保存。
aligned data 单独保存。
train_ready data 单独保存。
quality_report 单独保存。
metadata 单独保存。
```

推荐目录结构：

```text
raw/
clean/
aligned/
train_ready/
metadata/
reports/
```

## 9. 对你当前 nilm 项目的对应关系

你当前项目已经有类似论文中的 pipeline 思想：

```text
原始数据
-> DataSeparator
-> 单文件检测 validation
-> 单文件修复 repair
-> aggregate 与 appliance 对齐
-> CSV 导出
-> 模型训练
```

你的训练 CSV 形态类似：

```text
time,aggregate,kettle
2014-05-07 07:17:54,205.0,1.0
```

这和论文强调的 NILM 监督学习数据结构是一致的。

你们后续自采数据时，可以直接把公司采集数据转换成类似格式：

```text
time,aggregate,air_conditioner
2026-06-11 10:00:00,850.2,620.1
```

也可以多设备宽表：

```text
time,aggregate,air_conditioner,fridge,washing_machine
2026-06-11 10:00:00,850.2,620.1,95.0,0.0
```

## 10. 这篇论文没有完全解决什么

这篇论文有启发，但不能直接替你们解决所有公司采集问题。

它没有完全回答：

```text
广州本地家庭/园区应该选哪些典型设备
采集多少户才够训练泛化模型
工业设备和家庭设备是否要分开建模
隐私合规怎么处理
采集网关断网后如何补传
边缘设备如何做长期运维
```

所以你们不能只照论文做，还需要结合业务场景补充工程约束。

## 11. 最适合公司落地的简化方案

第一阶段建议不要追求复杂高频采集，而是先做一个高质量 pilot 数据集。

建议配置：

```text
采集地点：5-10 个点位
采集周期：至少 30 天，最好 3 个月
采样率：1 Hz
通道：1 个 aggregate + 3-6 个重点 appliance
字段：timestamp, voltage, current, active_power, power_factor
同步：NTP 或统一网关授时
保存：raw CSV + clean CSV + metadata.json + quality_report
```

第一阶段目标不是论文级大数据集，而是验证：

```text
硬件能否稳定采集
时间戳是否可靠
aggregate 与 appliance 能否对齐
数据缺失率是否可接受
模型能否跑出有效 baseline
```

## 12. 总结

这篇论文告诉我们：NILM 数据采集的关键不是“记录功率”，而是建设一套可复现的数据工程体系。

对广州疆海科技来说，可以把它转化成三句话：

```text
第一，采集时必须同时有总表和设备级子表。
第二，所有通道必须统一时间轴，并保留元数据。
第三，采集后必须经过标准 converter、质量检测、修复、对齐和导出，才能进入模型训练。
```

最重要的一点是：

```text
aggregate 与 appliance 的时间对齐质量，决定了 NILM 数据集是否可用。
```
