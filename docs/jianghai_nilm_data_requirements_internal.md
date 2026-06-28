# 广州疆海科技 NILM 项目数据接入与处理说明（内部版）

版本：v2.0  
适用对象：项目内部数据处理、算法训练、质量评估和 pipeline 对接。

## 1. 内部目标

本文件用于指导项目侧如何基于疆海交付的 `raw/` 和 `metadata/` 生成可训练、可追溯、可评估的 NILM 数据资产。对外沟通时，以 `jianghai_nilm_data_requirements_external.md` 为准；本文件保留内部处理产物、pipeline、训练 CSV、质量分级和建模适用性说明。

核心目标：

1. 读取疆海交付的原始数据和元数据。
2. 统一字段、单位、时间戳、正负号和测点口径。
3. 对齐家庭总负载与设备级高可信标签。
4. 生成当前训练代码可直接读取的 CSV。
5. 输出质量报告和建模适用性分级。
6. 记录每个训练文件的来源、清洗策略、标签来源和划分策略。

## 2. 输入与输出责任边界

疆海侧首要提供：

1. `raw/`：云平台、设备、App、测试系统直接导出的原始数据。
2. `metadata/`：字段字典、测点拓扑、设备绑定、单位、采样、正负号、可用性矩阵等说明。

项目侧生成：

1. `clean/`：统一字段、单位、时间戳、正负号和质量标签后的清洗数据。
2. `aligned/`：按家庭、测点和时间轴对齐后的宽表数据。
3. `train_ready/`：适配本项目训练流程的 CSV，形如 `time,aggregate,<appliance>,segment_id`。
4. `quality/`：缺失、乱序、重复、异常值、标签覆盖、能量平衡和建模适用性报告。

## 3. 项目侧处理产物文件树

```text
jianghai_nilm_processed/
  clean/                            # 清洗层：统一字段名、时间戳、单位、正负号和质量标签
    H0001/
      aggregate_1s.parquet
      meter_ct_1s.parquet
      smart_plug_1s.parquet
      labels.parquet
    H1001/
      aggregate_1s.parquet
      meter_ct_1s.parquet
      pv_1s.parquet
      storage_1s.parquet
      inverter_1s.parquet
      labels.parquet

  aligned/                          # 对齐层：把同一家庭、同一时间轴的数据对齐为宽表
    H0001/
      aligned_wide_1s.parquet
      aligned_wide_6s.parquet
      quality_report.json
    H1001/
      aligned_wide_1s.parquet
      aligned_wide_6s.parquet
      quality_report.json

  train_ready/                      # 训练层：可直接进入本项目 NILM 训练流程的数据
    project_training_csv/
      no_pv_storage/
        fridge_H0001.csv
        fridge_H0001.meta.yaml
        washing_machine_H0001.csv
        dishwasher_H0001.csv
      pv_storage/
        fridge_H1001.csv
        washing_machine_H1001.csv
        heat_pump_H1001.csv
    multi_target/
      no_pv_storage_H0001_1s.csv
      pv_storage_H1001_1s.csv

  quality/                          # 质量报告：用于判断数据能否进入训练和评估
    dataset_summary.csv
    missing_rate_by_file.csv
    timestamp_gap_report.csv
    meter_balance_report.csv
    label_coverage_report.csv
    pv_storage_energy_balance_report.csv
    modeling_applicability_report.csv
```

## 4. Pipeline 总览

```text
疆海 raw + metadata
        ↓
输入端准入检查
        ↓
字段映射与单位转换
        ↓
单信号检测
        ↓
单信号修复
        ↓
测点拓扑与负载口径确认
        ↓
aggregate 与设备标签时间对齐
        ↓
连续片段 segment 构建
        ↓
训练 CSV + sidecar metadata 导出
        ↓
质量报告与建模适用性评估
```

处理原则：

1. raw 原始文件不覆盖。
2. raw 层允许重复、乱序、缺口和云端回填，但必须保留 `source_sequence_id` 或 `ingest_order`。
3. clean/aligned/train_ready 层要求同一 `house_id`、`source_channel_id`、`timestamp_utc` 下无重复且时间单调。
4. 长缺口不强行补齐，必须切新 `segment_id`。
5. 训练窗口不得跨越不同 `segment_id`。

## 5. 无光储数据与当前训练流程

无光储数据可以较直接进入本项目训练流程，但前提是生成标准训练 CSV：

```text
time,aggregate,<appliance_name>,segment_id
```

最小条件：

1. 有全屋总负载或总表/CT 功率通道，可作为 `aggregate`。
2. 有至少一个目标设备的高可信设备级功率标签，可作为 `<appliance_name>`。
3. 总表测点覆盖该目标设备所在回路。
4. 总表和设备标签来自同一家庭，时间戳可解析并可对齐。
5. 功率单位可统一为 W，采样间隔可重采样到固定频率。
6. 缺失、重复、异常值、设备离线和长缺口能够被标记或修复。
7. 可以生成连续片段 `segment_id`，训练窗口不会跨越长缺口。

如果只有家庭总负载、没有设备级标签，则不能用于监督式 NILM 训练，只能用于无监督探索、能耗画像、异常检测或后续人工标注候选片段。

## 6. 有光储数据的 aggregate 口径

有光储家庭不能默认把并网点 `grid_power_w` 当作家庭负载。负载口径必须根据 `measurement_points.csv` 中的测点拓扑确认。

优先级：

1. 若存在负载侧直接测点，优先使用 `measured_load_power_w`。
2. 若只有并网点、PV、电池和逆变器测点，则通过 `formula_id` 推导 `load_estimated_power_w`。
3. 推导字段必须记录 `input_fields`、`loss_assumption`、`formula_id`、`quality_flag_primary` 和 `energy_balance_residual`。

不要写死单一公式，因为 AC 耦合、DC 耦合、三相、零馈网、备电回路和 CT 安装位置都会改变功率平衡关系。

## 7. 训练 CSV 规范

当前训练代码直接读取 CSV，并按列位置使用：

```csv
time,aggregate,dishwasher,segment_id
2026-01-01 00:00:00,223.45,0.00,0
2026-01-01 00:00:01,225.12,0.00,0
2026-01-01 00:00:02,1203.87,981.40,0
```

要求：

1. 第 1 列：`time`。
2. 第 2 列：`aggregate`，单位 W。
3. 第 3 列：目标设备名，单位 W。
4. 第 4 列：`segment_id`。
5. 不含 NaN。
6. 每个 `segment_id` 内采样间隔固定。
7. `aggregate` 和目标设备标签属于同一家庭、同一时间轴、同一负载覆盖范围。

每个训练 CSV 必须配套 sidecar metadata，例如 `dishwasher_H0001.meta.yaml`：

```yaml
house_id: H0001
house_system_type: no_pv_no_storage
operation_state: grid_tied
aggregate_type: measured_load_power_w
sampling_interval_s: 1
source_files:
  - raw/no_pv_storage/H0001/aggregate/mains_1s_2026-01-01.csv
  - raw/no_pv_storage/H0001/smart_plug/plug_P0002_dishwasher_1s_2026-01-01.csv
label_source: smart_plug
label_confidence: 0.98
quality_policy: default_v1
split_policy: cross_house_v1
preprocess_config: preprocess_default_v1.yaml
schema_version: nilm_train_csv_v1
```

## 8. 数据划分与评估协议

必须提供或生成 `metadata/splits.csv`，避免相邻时间窗随机切分造成数据泄漏。

| 字段 | 说明 |
|---|---|
| `house_id` | 家庭 ID。 |
| `date_start` | 起始日期。 |
| `date_end` | 结束日期。 |
| `split` | `train`、`val`、`test`。 |
| `seen_household_flag` | 训练中是否见过该家庭。 |
| `seen_appliance_model_flag` | 训练中是否见过同型号设备。 |
| `season` | 季节。 |
| `house_system_type` | 家庭硬件形态。 |
| `operation_state` | 运行状态。 |

建议至少支持三类评估：

1. within-house temporal split：同一家庭按时间切分。
2. cross-house split：训练和测试家庭不同。
3. cross-scenario split：无光储、有光储、不同运行状态之间的迁移评估。

## 9. 数据质量分级与建模适用性评估

不要把理想数据质量写成企业交付的绝对硬门槛。内部建议把质量报告分成建模适用性等级。

| 等级 | 含义 | 适用用途 |
|---|---|---|
| A | 时间连续、总表和标签稳定、缺失低、测点口径清晰 | 主训练和正式评估。 |
| B | 有少量缺失或短时异常，可修复 | baseline 训练和消融实验。 |
| C | 标签少、采样较低或测点口径部分缺失 | 弱监督、统计分析、策略分析。 |
| D | 缺失严重、口径不明、无法对齐 | 仅用于问题诊断，不进入训练。 |

建议报告：

1. `dataset_summary.csv`
2. `missing_rate_by_file.csv`
3. `timestamp_gap_report.csv`
4. `meter_balance_report.csv`
5. `label_coverage_report.csv`
6. `pv_storage_energy_balance_report.csv`
7. `modeling_applicability_report.csv`

## 10. 修复与质量标签

修复端目标是把可修复问题转换为训练可用数据，同时保留质量标签。

推荐顺序：

1. 保留 raw 原始文件不覆盖。
2. 字段映射和单位转换。
3. 时间排序和去重。
4. 按目标频率重采样。
5. 短缺口插值或有限前向填充。
6. 长缺口标记并切分 segment。
7. 非负裁剪和异常上限裁剪。
8. 测点拓扑检查和负载口径确认。
9. aggregate 与设备标签按时间 join。
10. 导出训练 CSV 和 sidecar metadata。

质量标签：

| 字段 | 说明 |
|---|---|
| `quality_flag_primary` | 主质量状态：`valid`、`missing`、`interpolated`、`outlier`、`estimated`、`invalid`。 |
| `is_missing` | 是否缺失。 |
| `is_interpolated` | 是否插值或前向填充。 |
| `is_outlier` | 是否异常值。 |
| `is_clipped` | 是否被裁剪。 |
| `is_estimated` | 是否为推导或估算值。 |
| `is_device_offline` | 设备是否离线。 |
| `is_clock_suspect` | 时间戳是否可疑。 |
| `quality_score` | 0-1 质量分。 |
| `quality_reason_code` | 原因码，如 `long_gap`、`duplicate_timestamp`、`ct_direction_changed`。 |

