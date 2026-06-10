# 论文精读笔记：The Design, Creation, Implementation, and Study of a New Dataset Suitable for NILM

论文链接：<https://www.mdpi.com/2076-3417/15/13/7200>  
DOI：<https://doi.org/10.3390/app15137200>  
阅读重点：数据如何采集、如何预处理、如何形成 NILM 可用的数据 pipeline。

## 这篇论文最值得学什么

这篇论文不是单纯做 NILM 模型，而是围绕一个可复现的 NILM 数据集体系展开。它的核心价值是：

1. 如何设计 NILM 采集硬件；
2. 如何同时采集总功率和设备级功率；
3. 如何把原始采集数据转成 CSV；
4. 如何处理时间戳、采样率、谐波、缺失或不一致时间；
5. 如何转换成 NILMTK 可用的 HDF5 + YAML metadata；
6. 如何基于统一指标比较模型效果。

可以把它理解成一个完整链路：

```text
电表/采集硬件
  -> 原始电气量测量
  -> CSV 文件
  -> 时间对齐/字段整理/标签整理
  -> NILMTK converter
  -> HDF5 数据集 + YAML 元数据
  -> 训练/验证/测试划分
  -> NILM 模型训练
  -> MAE/RMSE/F1/NDE/EAE/MNEAP 评估
```

## 数据采集设计

论文用了三类开源或低成本采集硬件：

| 硬件 | 作用 | 特点 |
|---|---|---|
| oZm v1 | 生成 DSUALM、DSUALMH | 单相电能质量分析仪，15.625 Hz，支持到 50 次谐波 |
| oZm v3 | 生成 DSUALM10、DSUALM10H | 三相/单相，60 kHz，高精度，支持 GPS/PPS 同步 |
| OMPM | 生成 UALM2 | ESP32 + PZEM004 模块，可扩展多通道，低成本 |

采集思路是 NILM 里最标准的 supervised 数据集做法：

- 一个通道采集总负载 aggregate；
- 多个通道采集单个 appliance；
- 让总负载和设备负载共享时间轴；
- 后续模型输入总负载，目标输出单设备负载或开关状态。

## 采集了哪些字段

不同数据集字段不同，但核心字段包括：

- timestamp：13 位时间戳，通常是毫秒级 Unix timestamp；
- voltage / VLN：电压；
- current / A：电流；
- active power / W：有功功率；
- frequency / F：频率；
- power factor / PF：功率因数；
- harmonic features：电流、电压、功率到 50 次谐波，部分数据集包含。

其中 UALM2 的 CSV header 明确为：

```text
timestamp, VLN, A, W, F, PF
```

对我们来说，最小可行数据结构可以先学 UALM2：

```text
timestamp, aggregate_power, appliance_1_power, appliance_2_power, ...
```

如果要更完整，再扩展到：

```text
timestamp, voltage, current, active_power, frequency, power_factor
```

再进一步才考虑谐波特征。

## 数据预处理流程

论文里的预处理重点不是复杂算法，而是工程一致性：

1. 从采集设备/API 导出原始测量；
2. 修正 date/time 不一致；
3. 转成 CSV；
4. 每个通道保持统一字段；
5. 使用 NILMTK converter 组织成 HDF5；
6. 补充 YAML metadata；
7. 设置 start/end date；
8. 做采样率、填充方法、功率过滤阈值等实验配置。

这里最关键的是时间戳。论文强调 converter 支持 13 位 timestamp，因为毫秒级时间戳可以更好地对齐高频采样数据。

## NILMTK 数据格式

论文使用 NILMTK v0.4.0 和 NILMTK-Contrib。NILMTK 要求数据不只是数值表，还需要 metadata。

一个 NILMTK 数据集通常包括：

- HDF5：存储时间序列测量值；
- YAML metadata：描述建筑、meter、appliance、measurement type、采样周期等。

论文开发了多个 converter，例如：

- `ualm5t.convert_ualmt`
- `ualmt10h.convert_ualmt10h`
- UALM2 converter

这些 converter 的作用是把 CSV/raw files 转成 NILMTK 统一格式。

## 训练与评估 pipeline

论文中模型 pipeline 大致是：

```text
HDF5 dataset
  -> 选择建筑/设备/时间范围
  -> train / validation / test split
  -> 设定采样间隔，例如 1s、90s、30min
  -> 可选功率过滤，例如 10W、50W、100W
  -> 训练算法
  -> 保存 H5 model
  -> 在测试集上 disaggregate
  -> 计算指标
```

评估算法包括：

- 传统方法：CO、FHMM、Mean、Hart85；
- 深度学习：DAE、RNN、Seq2Point、Seq2Seq、WindowGRU。

指标包括：

- F1-score：判断设备开关状态识别好不好；
- MAE：平均绝对误差；
- RMSE：对大误差更敏感；
- NDE：归一化分解误差；
- EAE：能源分配误差；
- MNEAP：归一化功率分配误差。

## 对我们最有用的启发

如果你要自己做 NILM 数据 pipeline，可以先别追求论文里的全部复杂度。建议按三步走：

### 第一阶段：最小可行采集

先采：

```text
timestamp, aggregate_power, appliance_power
```

要求：

- 时间戳统一；
- 聚合功率和设备功率采样频率一致；
- 每个设备有明确标签；
- 至少保存原始 CSV。

### 第二阶段：标准预处理

做这些处理：

```text
parse timestamp
sort by time
remove duplicate timestamps
resample to fixed interval
align aggregate and appliance channels
fill short missing gaps
drop long missing gaps
normalize or standardize if用于深度学习
split train/val/test by time
```

注意：不要随机打乱时间序列切分，NILM 更适合按时间段切分。

### 第三阶段：转换成统一格式

如果后续想接 NILMTK：

- 写 converter；
- 输出 HDF5；
- 写 YAML metadata；
- 固定数据集命名、building、meter、appliance 字段。

如果只服务自己的 PyTorch pipeline：

- 可以先保存为 Parquet/HDF5；
- 同时保留一个 metadata.json；
- 后续再兼容 NILMTK。

## 我建议你重点读的章节

1. Abstract：看它的贡献点，尤其是 converter、timestamp、harmonics、open-source hardware。
2. Section 2 NILMTK：理解标准 NILM pipeline。
3. Section 3 Materials and Methods：重点，采集硬件和数据转换都在这里。
4. Section 4.1：看不同数据集如何生成，特别是 DSUALMH、UALM2、DSUALM10H。
5. Appendix A：看 converter、建 dataset、训练 model、算 metrics 的代码截图。

## 你可以照着复现的简化版 pipeline

```text
1. 采集
   - 总表：aggregate.csv
   - 插座/分路：appliance_x.csv

2. 原始 CSV
   - timestamp
   - voltage
   - current
   - power
   - frequency
   - power_factor

3. 预处理
   - timestamp 转 datetime
   - 按 timestamp 排序
   - 去重
   - resample 到 1s 或 10s
   - aggregate 和 appliance 对齐
   - 缺失值处理
   - 保存 clean CSV/Parquet

4. 标注
   - appliance name
   - meter id
   - building id
   - room/location
   - on_power_threshold

5. 训练数据
   - X = aggregate power window
   - y = appliance power window 或 appliance on/off state

6. 评估
   - 回归：MAE, RMSE, NDE
   - 状态识别：Precision, Recall, F1
```

## 一句话总结

这篇论文最值得借鉴的不是某个模型，而是它把 NILM 数据集建设工程化了：用开放硬件采集可复现数据，用 converter 统一格式，用 metadata 保留语义，用标准指标比较算法。
