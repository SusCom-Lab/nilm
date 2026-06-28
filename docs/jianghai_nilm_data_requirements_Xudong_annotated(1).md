# 广州疆海科技 NILM 项目数据提供需求说明

版本：v1.0  
适用项目：家庭负载 NILM 识别；分为“无光伏储能家庭”和“有光伏/储能/逆变器家庭”两类场景。  


## 1. 背景与结论

本项目目标是训练和验证 NILM（Non-Intrusive Load Monitoring，非侵入式负荷识别）模型：以家庭总负载、智能电表/CT 等总表数据为输入，识别冰箱、洗衣机、洗碗机、空调、热泵、热水器、厨房电器、照明、待机负载等设备级用电功率或运行状态。

疆海科技的公开信息显示，其业务重点不是单一电表，而是家庭储能和能源管理系统。疆海科技官网介绍其旗下品牌为 Zendure/征拓，核心聚焦家庭储能和能源管理系统研发与销售，并覆盖 BMS、PCS、物联网、云服务等链条；36 氪项目信息也说明其产品由主机、扩展电池包、周边智能配件和 ZEN+ 智能云平台组成。Zendure 官方页面公开展示了 SolarFlow、Hyper 2000、Smart Meter 3CT、Satellite Monitor CT、HEMS/Zenki 等产品与能力，其中 Smart Meter 3CT 用于三相家庭用电监测，Satellite Monitor CT 可通过 Zendure App 与 SolarFlow 联动获取实时用电统计，HEMS/Zenki 会结合用电习惯、天气、电池容量和动态电价优化家庭能源计划。

(Xudong:[建议对外发送版本弱化或删除这一整段公开商业信息引用，改为“根据项目合同目标和前期沟通，本项目覆盖无光储与有光储两类家庭能源场景”。个人感觉合作方比我们更清楚自身产品线，过多引用公开信息可能显得像外部调研报告，也可能因公开信息过时或不完整引发不必要争议？内部版本可以保留作为背景。])


因此，本项目向疆海索取的数据不能只停留在“总功率 CSV”。需要同时覆盖：

1. 无光伏储能场景：家庭总负载 + 电表/CT + 智能插座 + 设备/用户/App 日志 + 设备标签。
2. 有光伏储能场景：在上述基础上增加 PV 发电、储能充放电、电池 SoC/BMS、并网点输入输出、逆变器/EMS 状态与控制日志。
3. 数据工程支撑：字段字典、采样频率、时间戳规则、单位、正负号、业务口径、质量报告和数据血缘。

## 2. 总体交付原则

(Xudong:[建议在本节开头增加“交付优先级分级”：M0 为字段样例包，M1 为小样本联调包，M2 为 baseline 训练包，M3 为长期 benchmark 增强包。原因：当前文档中“必须”“必填”“一次交付”的语气较强，容易与实际合作初期的数据可用性冲突；分级后既保留技术完整性，也便于第一轮推进。])


### 2.1 两类场景分开标记

每个家庭、每一天、每条数据都必须明确 `scenario_type`：

| scenario_type | 含义 | 是否必须提供 |
|---|---|---|
| `no_pv_storage` | 无光伏、无家庭储能，只有市电和家庭负载 | 必须 |
| `pv_only` | 有光伏，无电池储能 | 如存在则必须 |
| `storage_only` | 无光伏，有电池储能 | 如存在则必须 |
| `pv_storage_grid_tied` | 有光伏、有储能、并网 | 必须，若疆海产品覆盖该类家庭 |
| `pv_storage_off_grid` | 有光伏、有储能、离网/备电 | 如存在则必须 |
| `mixed_unknown` | 设备状态不完整，暂无法确认 | 仅允许在原始层出现，清洗层必须修正或剔除 |

(Xudong:[这里我使用GPT5.5Pro检查给了一些细的建议，我觉得比较合理的，因为家庭硬件类型相对稳定，但并网、离网、备电、充放电、限功率等状态会随时间变化；如果都塞进 `scenario_type`，会导致每条记录的语义混乱，也不利于后续建模和统计。 建议将 `scenario_type` 拆成两个字段：`house_system_type` 和 `operation_state`。`house_system_type` 表示家庭硬件形态，例如 `no_pv_no_storage`、`pv_only`、`storage_only`、`pv_storage`；`operation_state` 表示运行状态，例如 `grid_tied`、`off_grid`、`backup`、`islanding`、`zero_export`、`charging`、`discharging`、`curtailment`、`fault`、`unknown`。])


### 2.2 原始数据和训练数据都要交付

只给训练 CSV 不够，因为后续需要复查问题、调整口径、重采样和生成不同模型数据集。建议一次交付四层数据：

1. `raw/`：从云平台、设备、App、测试系统直接导出的原始数据，不覆盖、不改值。
2. `clean/`：统一时间、单位、字段、正负号后的清洗数据。
3. `aligned/`：按家庭和时间轴对齐后的宽表数据，供特征工程和质量检查。
4. `train_ready/`：适配本项目训练流程的 CSV，形如 `time,aggregate,<appliance>,segment_id`。

(Xudong:[建议补充责任边界：`raw/` 和必要 `metadata/` 是甲方首要提供内容；`clean/`、`aligned/`、`train_ready/`、`quality/` 更适合作为乙方基于原始数据和字段说明生成的处理产物，若甲方已有内部清洗版可作为补充交付。因为第一阶段本身包含数据读取、格式转换、时间对齐、重采样、清洗、异常标记、质量评分和标准化输出；如果这里写成四层都由甲方一次交付，容易混淆责任边界并增加沟通成本。])


### 2.3 采样频率要求

| 数据类型 | 最低可用 | 推荐 | 高价值增强 |
|---|---:|---:|---:|
| 家庭总负载/智能电表/CT 有功功率 | 10 s | 1 s | 100 Hz 至 1 kHz 以上波形或 RMS |
| 智能插座设备级功率标签 | 10 s | 1 s | 100 Hz 以上事件片段 |
| PV/储能/并网功率 | 10 s | 1 s | 100 ms 至 1 s 控制闭环数据 |
| 电池 SoC/BMS 状态 | 60 s | 1 s 至 10 s | 事件触发高频日志 |
| 逆变器/EMS/告警/保护日志 | 事件级 | 事件级 + 1 s 状态快照 | 控制指令前后 30 s 高频片段 |
| App 操作日志 | 事件级 | 事件级 | 操作前后能源状态快照 |

若设备端历史数据只能提供 5 分钟或 15 分钟聚合数据，该数据只能作为业务分析辅助，不适合作为 NILM 主训练数据。

(Xudong:[个人建议，把这句话改成任务依赖型表述？ 5 分钟或 15 分钟数据确实不适合严格的细粒度 appliance-level NILM 主训练，但仍可用于日内行为画像、能量平衡、异常检测、用电推荐、储能策略分析和业务统计。])


### 2.4 数据周期和样本量

| 阶段 | 家庭数量 | 连续天数 | 目的 |
|---|---:|---:|---|
| 小样本联调 | 每类场景 3-5 户 | 14 天以上 | 验证字段、时间同步、正负号、文件结构和 pipeline。 |
| 第一版训练 | 无光储 20 户以上；有光储 20 户以上 | 60-90 天 | 覆盖工作日/周末、晴雨天、季节变化和常用电器。 |
| 稳定评估 | 每类场景 50 户以上 | 180 天以上 | 评估跨家庭泛化、PV/储能扰动、设备迁移能力。 |
| 长周期优化 | 每类场景 100 户以上 | 365 天 | 支持季节性设备、热泵/空调、储能策略、动态电价场景。 |



## 3. 建议数据文件树

请按以下目录交付。`H0001` 表示家庭匿名编号，不应使用真实姓名、手机号、详细地址。

```text
jianghai_nilm_dataset/
  README.md
  manifest.yaml
  data_license_and_privacy.md

  metadata/
    dataset.yaml
    households.csv
    meters.csv
    circuits.csv
    appliances.csv
    smart_plugs.csv
    pv_systems.csv
    storage_systems.csv
    inverters.csv
    tariff_plans.csv
    firmware_versions.csv
    field_dictionary.csv
    sign_convention.md
    sampling_policy.md
    appliance_taxonomy.yaml

  raw/
    no_pv_storage/
      H0001/
        aggregate/
          mains_1s_2026-01-01.csv
          mains_1s_2026-01-02.csv
        meter_ct/
          smart_meter_3ct_1s_2026-01-01.csv
          phase_l1_1s_2026-01-01.csv
          phase_l2_1s_2026-01-01.csv
          phase_l3_1s_2026-01-01.csv
        smart_plug/
          plug_P0001_fridge_1s_2026-01-01.csv
          plug_P0002_washing_machine_1s_2026-01-01.csv
        device_logs/
          device_status_events_2026-01.jsonl
          firmware_events_2026-01.jsonl
        app_logs/
          app_user_events_2026-01.jsonl
        labels/
          strong_labels_2026-01.csv
          weak_labels_2026-01.csv
          user_feedback_2026-01.csv

    pv_storage/
      H1001/
        aggregate/
          mains_1s_2026-01-01.csv
        meter_ct/
          smart_meter_3ct_1s_2026-01-01.csv
          grid_point_1s_2026-01-01.csv
        smart_plug/
          plug_P0001_fridge_1s_2026-01-01.csv
        pv/
          pv_string_1s_2026-01-01.csv
          pv_mppt_1s_2026-01-01.csv
        storage/
          battery_power_1s_2026-01-01.csv
          battery_soc_1s_2026-01-01.csv
          bms_status_events_2026-01.jsonl
        inverter/
          inverter_status_1s_2026-01-01.csv
          inverter_events_2026-01.jsonl
          protection_alarm_events_2026-01.jsonl
        hems/
          energy_plan_2026-01.jsonl
          control_commands_2026-01.jsonl
          forecast_inputs_2026-01.csv
        app_logs/
          app_user_events_2026-01.jsonl
        labels/
          strong_labels_2026-01.csv
          weak_labels_2026-01.csv
          user_feedback_2026-01.csv

  clean/
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

  aligned/
    H0001/
      aligned_wide_1s.parquet
      aligned_wide_6s.parquet
      quality_report.json
    H1001/
      aligned_wide_1s.parquet
      aligned_wide_6s.parquet
      quality_report.json

  train_ready/
    project_training_csv/
      no_pv_storage/
        fridge_H0001.csv
        washing_machine_H0001.csv
        dishwasher_H0001.csv
      pv_storage/
        fridge_H1001.csv
        washing_machine_H1001.csv
        heat_pump_H1001.csv
    multi_target/
      no_pv_storage_H0001_1s.csv
      pv_storage_H1001_1s.csv

  quality/
    dataset_summary.csv
    missing_rate_by_file.csv
    timestamp_gap_report.csv
    meter_balance_report.csv
    label_coverage_report.csv
    pv_storage_energy_balance_report.csv
```

(Xudong:[建议在 `metadata/` 新增补充文件？：一是 `data_availability_matrix.csv`，记录每类数据是否可提供、覆盖户数、时间范围、最小粒度、导出格式等；二是 `measurement_points.csv`，记录 CT/电表/逆变器/PV/电池等测点位置、相别、方向、上下游关系和正负号；三是 `splits.csv`，看到有关 train_ready 这个目录，如果对方有偏好？可以也给出一个train/val/test 的家庭、日期、场景和是否跨家庭划分？方便验收])

## 4. 通用字段要求

所有 CSV/Parquet 时间序列都必须包含以下公共字段。

| 字段名 | 类型 | 单位/枚举 | 必填 | 说明 |
|---|---|---|---|---|
| `timestamp_utc` | datetime/string | ISO 8601, UTC | 是 | 统一时间轴，例如 `2026-01-01T00:00:00.000Z`。 |
| `timestamp_local` | datetime/string | 本地时间 | 是 | 用户所在时区的本地时间，用于行为模式分析。 |
| `timezone` | string | IANA TZ | 是 | 例如 `Europe/Berlin`、`Asia/Shanghai`。 |
| `house_id` | string | `H0001` | 是 | 匿名家庭编号。 |
| `scenario_type` | string | 见 2.1 | 是 | 区分无光储、有光储等。 |
| `source_device_id` | string | 匿名 ID | 是 | 采集设备、插座、逆变器、电池或 App 侧 ID。 |
| `source_channel_id` | string | 匿名 ID | 是 | CT 相线、插座通道、MPPT 通道、电池包通道等。 |
| `sampling_interval_s` | number | s | 是 | 该记录所属文件的目标采样间隔。 |
| `quality_flag` | string | `raw/valid/interpolated/missing/outlier/clipped/estimated` | 是 | 数据质量标记。原始层可为 `raw`。 |
| `ingest_batch_id` | string | 批次 ID | 是 | 便于追溯本次交付批次。 |

(Xudong:[这条建议是我使用GPT5.5Pro检查后的, 建议新增公共字段或伴随字段：`house_system_type`、`operation_state`、`timestamp_device_utc`、`timestamp_server_utc`、`timestamp_ingest_utc`、`source_sequence_id`、`clock_drift_ms`、`availability_status`、`source_system`、`schema_version`。原因：真实 IoT 数据可能存在设备时钟漂移、云端回填、服务器延迟、字段版本变化和部分数据暂不可提供；这些字段可以显著提升追溯能力和数据质量诊断能力。])

(Xudong:[这条建议是我使用GPT5.5Pro检查后的,建议把单一 `quality_flag` 扩展为可组合质量标签，例如 `quality_flag_primary`、`is_missing`、`is_interpolated`、`is_outlier`、`is_clipped`、`is_estimated`、`is_device_offline`、`is_clock_suspect`、`quality_score`、`quality_reason_code`。原因：一个点可能同时是估算值、插值点和低可信点；单一字符串无法支持后续训练时的 mask、loss weighting 和样本筛选。])


时间戳规则：

1. 必须提供 UTC 时间戳，不允许只给本地时间或设备相对时间。
2. 同一文件中时间戳必须严格单调递增。
3. 重复时间戳必须保留在 raw 层，并在 clean 层按规则去重。
4. 数据缺口不得用 0 填充；缺失就是缺失，修复层用 `quality_flag=interpolated` 标记。
5. 如果云端入库时间和设备采样时间不同，必须同时给 `timestamp_device_utc` 和 `timestamp_server_utc`。

(Xudong:[第 2 条和第 3 条存在潜在冲突：如果 raw 层要保留重复时间戳和原始乱序，那么同一文件在 raw 层就不应强制“严格单调递增”。建议改为：raw 层保留源系统原貌，可存在重复、乱序、缺口和回填，但必须保留 `source_sequence_id` 或 `ingest_order`；clean、aligned、train_ready 层才要求同一 `house_id`、`source_channel_id`、`timestamp_utc` 下无重复且时间单调。原因：这样既保留审计能力，又保证建模层数据干净。])


功率正负号总规则：

| 字段类别 | 推荐正负号口径 |
|---|---|
| 家庭负载 `load_power_w` | 正值表示家庭消耗功率；理论上不应为负。 |
| 电网 `grid_power_w` | 正值表示从电网买电/输入家庭；负值表示向电网馈电。 |
| PV `pv_power_w` | 正值表示光伏发电输出。 |
| 电池 `battery_power_w` | 正值表示电池放电供家庭/电网；负值表示电池充电。 |
| 逆变器 AC 输出 `inverter_ac_power_w` | 正值表示向家庭 AC 侧输出；负值表示从 AC 侧吸收。 |
| 智能插座 `plug_power_w` | 正值表示插座下游设备消耗。 |

若疆海内部已有相反口径，请不要直接改值，应同时提供原始字段和 `sign_convention.md`，再在 clean 层统一。

(Xudong:[这条建议是我使用GPT5.5Pro检查后的, 建议把正负号说明与 `measurement_points.csv` 绑定，而不是只在全局写一个口径。原因：同一个 `grid_power_w` 或 `active_power_w` 在不同 CT 安装位置、不同相别、不同逆变器 AC/DC 侧可能含义不同；有光储 NILM 的核心风险不是字段名缺失，而是测点拓扑和正负方向不清楚。])


## 5. 需要提供的数据清单与字段


### 5.1 家庭总负载数据

用途：NILM 主输入。训练数据中的 `aggregate` 列来自该数据。

文件建议：`raw/<scenario>/Hxxxx/aggregate/mains_1s_YYYY-MM-DD.csv`

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `timestamp_utc` | datetime | UTC | 1 s | 是 | 设备采样时间。 |
| `house_id` | string | - | 每行 | 是 | 家庭匿名编号。 |
| `aggregate_active_power_w` | number | W | 1 s | 是 | 家庭总有功功率；无光储场景可直接作为 `aggregate`。 |
| `aggregate_reactive_power_var` | number | var | 1 s | 推荐 | 总无功功率，辅助区分电机类负载。 |
| `aggregate_apparent_power_va` | number | VA | 1 s | 推荐 | 视在功率。 |
| `voltage_v` | number | V | 1 s | 推荐 | 单相或相均电压。 |
| `current_a` | number | A | 1 s | 推荐 | 单相或总电流。 |
| `power_factor` | number | 0-1 | 1 s | 推荐 | 功率因数。 |
| `frequency_hz` | number | Hz | 1 s | 可选 | 电网频率。 |
| `energy_import_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计购电量。 |
| `energy_export_kwh_total` | number | kWh | 10-60 s | 有光储必填 | 累计馈电量。 |
| `quality_flag` | string | - | 每行 | 是 | 质量标记。 |

重要业务口径：

1. 无光储：`aggregate_active_power_w` 应等于家庭负载总消耗。
2. 有光储：总表位置可能测到“电网交换功率”而不是“家庭真实负载”。必须明确 CT 安装位置。若 CT 装在并网点，不能直接把 `grid_power_w` 当成负载；需由 `load = grid_import - grid_export + pv_self_consumed + battery_discharge - battery_charge` 等口径推导，具体公式必须由疆海确认。

(Xudong:[这条建议是我使用GPT5.5Pro检查后的, 建议不要把这里的 `load = ...` 写成默认固定公式，而应写成“根据测点拓扑确认公式”。更稳妥的交付要求是：若存在负载侧直接测点，优先使用 `measured_load_power_w`；若只有并网点、PV、电池和逆变器数据，则通过 `measurement_points.csv` 中的拓扑和 `formula_id` 推导 `load_estimated_power_w`，同时记录 `input_fields`、`loss_assumption`、`quality_flag` 和 `energy_balance_residual`。原因：不同家庭可能存在 AC 耦合、DC 耦合、三相、零馈网、备电回路等差异，固定公式容易从根上算错 aggregate。])


### 5.2 智能电表/CT 数据

用途：校验家庭总负载、三相平衡、并网点功率、CT 安装方向；Smart Meter 3CT/Satellite Monitor CT 类产品应重点提供。

文件建议：`raw/<scenario>/Hxxxx/meter_ct/smart_meter_3ct_1s_YYYY-MM-DD.csv`

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `meter_id` | string | - | 每行 | 是 | 电表或 CT 设备匿名 ID。 |
| `meter_model` | string | - | 每文件/每行 | 是 | 例如 Smart Meter 3CT、Satellite Monitor CT 或内部型号。 |
| `phase` | string | `total/L1/L2/L3` | 每行 | 是 | 三相或单相。 |
| `ct_direction` | string | `import_positive/export_positive/unknown` | 每文件 | 是 | CT 安装方向和正负号。 |
| `ct_ratio` | number | A/A | 每文件 | 推荐 | CT 变比，例如 120A CT。 |
| `active_power_w` | number | W | 1 s | 是 | 有功功率。 |
| `reactive_power_var` | number | var | 1 s | 推荐 | 无功功率。 |
| `voltage_v` | number | V | 1 s | 推荐 | 相电压。 |
| `current_a` | number | A | 1 s | 推荐 | 相电流。 |
| `power_factor` | number | - | 1 s | 推荐 | 功率因数。 |
| `import_energy_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计正向电量。 |
| `export_energy_kwh_total` | number | kWh | 10-60 s | 有光储必填 | 累计反向电量。 |
| `rssi_dbm` | number | dBm | 60 s | 可选 | 设备通信质量。 |
| `firmware_version` | string | - | 事件/每日 | 推荐 | 方便排查固件差异。 |

(Xudong:[这条建议是我使用GPT5.5Pro检查后的, 建议在 CT/电表数据中增加 `measurement_point_id`、`bus_or_node`、`upstream_component`、`downstream_component`、`measures_load_or_grid_or_pv_or_battery`、`is_bidirectional`、`installed_start_utc`、`installed_end_utc`。原因：有光储场景下，CT 方向、安装位置和生效时间比单个 `active_power_w` 字段本身更关键；CT 改向、迁移或固件升级都可能造成训练标签系统性错误。])


### 5.3 智能插座数据

用途：监督式 NILM 的强标签，即设备级 ground truth。没有智能插座或子表标签，模型只能做弱监督/无监督，精度和评估可信度会明显下降。

(Xudong:[建议将“智能插座强标签”扩展为“高可信设备级标签”。标签来源可以是智能插座、子表、回路 CT、设备遥测、BMS/逆变器日志、App 控制日志或人工确认，并通过 `label_source` 和 `label_confidence` 标明可信度。原因：空调、热泵、热水器、EV 充电器和照明回路往往不是普通插座负载，若只要求智能插座会错过最有业务价值的大功率设备。])


文件建议：`raw/<scenario>/Hxxxx/smart_plug/plug_Pxxxx_<appliance>_1s_YYYY-MM-DD.csv`

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `plug_id` | string | - | 每行 | 是 | 智能插座匿名 ID。 |
| `appliance_id` | string | - | 每行 | 是 | 下游设备匿名 ID。 |
| `appliance_name` | string | taxonomy | 每行 | 是 | 如 `fridge`、`washing_machine`。 |
| `room_type` | string | taxonomy | 每行 | 推荐 | 厨房、客厅、洗衣房等。 |
| `plug_active_power_w` | number | W | 1 s | 是 | 插座下游设备有功功率。 |
| `plug_reactive_power_var` | number | var | 1 s | 推荐 | 无功功率。 |
| `plug_voltage_v` | number | V | 1 s | 推荐 | 插座电压。 |
| `plug_current_a` | number | A | 1 s | 推荐 | 插座电流。 |
| `plug_power_factor` | number | - | 1 s | 推荐 | 功率因数。 |
| `plug_energy_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计电量。 |
| `relay_state` | string | `on/off/unknown` | 1 s/事件 | 推荐 | 插座继电器状态。 |
| `is_label_source` | boolean | - | 每行 | 是 | 是否作为训练标签。 |
| `label_confidence` | number | 0-1 | 每行 | 推荐 | 设备绑定可信度。 |

优先覆盖设备：

| 优先级 | 设备 |
|---|---|
| P0 必须 | 冰箱/冰柜、洗衣机、洗碗机、热水器、空调/热泵、厨房大功率设备、照明总回路、路由器/常开待机负载。 |
| P1 推荐 | 微波炉、电水壶、烤箱、电磁炉、干衣机、咖啡机、除湿机、新风/风扇、电脑/电视娱乐系统。 |
| P2 有则提供 | EV 充电器、泳池泵、地暖、空气净化器、扫地机器人、其他疆海 HEMS 可控负载。 |

(Xudong:[这条建议是我使用GPT5.5Pro检查后的, 建议把 P0 表述从“必须”调整为“首批优先覆盖或优先评估可获得标签的设备”，并将设备拆成三类：`plug_metered_appliance`，例如冰箱、洗衣机、洗碗机、微波炉、电水壶；`circuit_metered_or_submetered`，例如空调、热泵、热水器、照明回路、EV 充电器；`system_telemetry_labeled`，例如逆变器、储能、HEMS 可控负载。原因：不同设备的可计量方式不同，不能把所有 P0 设备都默认为可由智能插座提供强标签。])


### 5.4 设备运行日志

用途：构造弱标签、解释负荷变化、排查异常；尤其对储能、逆变器、智能插座、网关、App 控制设备重要。

文件建议：`raw/<scenario>/Hxxxx/device_logs/device_status_events_YYYY-MM.jsonl`

每行 JSON 示例：

```json
{"timestamp_utc":"2026-01-01T08:30:12.400Z","house_id":"H0001","device_id":"D0001","device_type":"smart_plug","event_type":"relay_on","old_value":"off","new_value":"on","trigger_source":"app","firmware_version":"1.2.3","quality_flag":"raw"}
```

字段要求：

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `timestamp_utc` | datetime | 是 | 事件发生时间，不是入库时间。 |
| `device_id` | string | 是 | 匿名设备 ID。 |
| `device_type` | string | 是 | `meter/ct/smart_plug/inverter/battery/gateway/router/hems` 等。 |
| `event_type` | string | 是 | `power_on/power_off/relay_on/relay_off/mode_change/fault/firmware_update/communication_lost/reconnect` 等。 |
| `old_value` | string/number | 推荐 | 事件前状态。 |
| `new_value` | string/number | 推荐 | 事件后状态。 |
| `trigger_source` | string | 推荐 | `app/auto/hems/local_button/cloud/api/schedule/protection`。 |
| `error_code` | string | 有故障必填 | 故障码。 |
| `firmware_version` | string | 推荐 | 固件版本。 |

### 5.5 用户 App 操作日志

用途：识别用户主动开关、模式切换、定时任务、储能策略变化；对区分“设备自然运行”和“用户控制”很关键。

文件建议：`raw/<scenario>/Hxxxx/app_logs/app_user_events_YYYY-MM.jsonl`

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `timestamp_utc` | datetime | 是 | 用户操作发生时间。 |
| `user_id_hash` | string | 是 | 脱敏用户 ID。 |
| `house_id` | string | 是 | 家庭匿名编号。 |
| `app_version` | string | 推荐 | App 版本。 |
| `operation_type` | string | 是 | `device_on/off`、`mode_change`、`schedule_create`、`schedule_update`、`battery_reserve_set`、`energy_plan_set` 等。 |
| `target_device_id` | string | 推荐 | 被操作设备。 |
| `target_system` | string | 推荐 | `plug/inverter/battery/hems/pv/ev/heat_pump`。 |
| `old_setting` | json/string | 推荐 | 旧设置。 |
| `new_setting` | json/string | 推荐 | 新设置。 |
| `operation_result` | string | 是 | `success/fail/timeout/cancelled`。 |
| `client_timezone` | string | 推荐 | 用户端时区。 |

隐私要求：不得提供手机号、邮箱、姓名、GPS 精确地址、家庭成员画像等直接个人信息；只保留匿名 ID 与粗粒度区域/气候带。



### 5.6 家庭或设备基础信息

用途：跨家庭泛化、设备 taxonomy、设备额定功率先验、PV/储能建模和质量检查。

文件建议：`metadata/households.csv`、`metadata/appliances.csv`、`metadata/meters.csv`

`households.csv`：

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `house_id` | string | 是 | 匿名家庭编号。 |
| `scenario_type` | string | 是 | 见 2.1。 |
| `country_region` | string | 推荐 | 国家/地区即可，不要详细地址。 |
| `timezone` | string | 是 | IANA 时区。 |
| `grid_phase_type` | string | 是 | `single_phase/three_phase/split_phase/unknown`。 |
| `nominal_voltage_v` | number | 推荐 | 例如 230。 |
| `floor_area_band_m2` | string | 推荐 | 面积区间，如 `50-80`。 |
| `occupant_count_band` | string | 推荐 | 人数区间。 |
| `has_pv` | boolean | 是 | 是否有光伏。 |
| `has_storage` | boolean | 是 | 是否有储能。 |
| `has_ev_charger` | boolean | 推荐 | 是否有 EV 充电。 |
| `has_heat_pump` | boolean | 推荐 | 是否有热泵。 |
| `data_start_utc` | datetime | 是 | 数据起始时间。 |
| `data_end_utc` | datetime | 是 | 数据结束时间。 |

`appliances.csv`：

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `appliance_id` | string | 是 | 匿名设备 ID。 |
| `house_id` | string | 是 | 所属家庭。 |
| `appliance_name` | string | 是 | 标准英文小写下划线命名。 |
| `appliance_name_cn` | string | 推荐 | 中文名。 |
| `brand_model` | string | 可选 | 可脱敏；若涉及隐私可只给型号族。 |
| `rated_power_w` | number | 推荐 | 额定功率。 |
| `standby_power_w` | number | 推荐 | 待机功率。 |
| `room_type` | string | 推荐 | 房间。 |
| `metered_by_plug_id` | string | 强标签必填 | 对应智能插座或子表。 |
| `label_available` | boolean | 是 | 是否有强标签。 |
| `install_start_utc` | datetime | 推荐 | 接入起始时间。 |
| `install_end_utc` | datetime | 推荐 | 接入结束时间。 |

(Xudong:[这条建议是我使用GPT5.5Pro检查后的, 建议在 `households.csv` 中增加 `region_or_climate_zone`、`floor_area_band`、`occupant_count_band` 等区间字段即可，避免详细小区、街道、GPS 和完整邮编；在 `appliances.csv` 中增加 `label_source`、`label_confidence`、`metered_by_channel_id`、`binding_start_utc`、`binding_end_utc`。原因：既能支持跨家庭泛化和设备先验，又能降低重识别风险，并处理设备绑定变更导致的标签漂移。])


### 5.7 电器级标签、弱标签、用户反馈、人工标注样本

用途：监督训练、弱监督训练、评估和人工纠错。

强标签文件：`labels/strong_labels_YYYY-MM.csv`

| 字段名 | 类型 | 单位/枚举 | 必填 | 说明 |
|---|---|---|---|---|
| `timestamp_utc` | datetime | UTC | 是 | 对齐到采样点。 |
| `house_id` | string | - | 是 | 家庭 ID。 |
| `appliance_id` | string | - | 是 | 设备 ID。 |
| `appliance_name` | string | taxonomy | 是 | 设备名。 |
| `appliance_power_w` | number | W | 是 | 设备级真实功率。 |
| `appliance_state` | string | `on/off/standby/running/heating/cooling/defrost/unknown` | 推荐 | 状态标签。 |
| `label_source` | string | `smart_plug/submeter/bms/app/manual` | 是 | 标签来源。 |
| `label_confidence` | number | 0-1 | 是 | 标签可信度。 |

弱标签文件：`labels/weak_labels_YYYY-MM.csv`

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `event_start_utc` | datetime | 是 | 事件开始时间。 |
| `event_end_utc` | datetime | 推荐 | 事件结束时间。 |
| `house_id` | string | 是 | 家庭 ID。 |
| `appliance_name` | string | 是 | 推测设备。 |
| `event_type` | string | 是 | `turn_on/turn_off/cycle_start/cycle_end/mode_change`。 |
| `evidence_source` | string | 是 | `app/device_log/user_feedback/manual/power_event`。 |
| `confidence` | number | 是 | 0-1。 |
| `annotator` | string | 人工标注必填 | 标注人或标注系统。 |

用户反馈文件：`labels/user_feedback_YYYY-MM.csv`

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `feedback_time_utc` | datetime | 是 | 反馈时间。 |
| `predicted_appliance` | string | 是 | 系统预测设备。 |
| `user_confirmed_appliance` | string | 推荐 | 用户确认设备。 |
| `feedback_type` | string | 是 | `correct/wrong/missing/unknown`。 |
| `comment_category` | string | 可选 | 反馈类型，不要提供自由文本隐私内容。 |

(Xudong:[建议补充：用户反馈默认只提供结构化类别，不提供原始自由文本；如果确有必要使用文本，应先由甲方做脱敏、敏感词过滤和发布范围审批。原因：用户反馈文本可能包含设备昵称、家庭成员、地址、电话号码或使用习惯等敏感信息。])


### 5.8 光伏发电数据

有光伏家庭必填。用途：从并网点/总表数据中剥离 PV 扰动，识别真实负载；也用于 PV+储能能量平衡。

(Xudong:[这条建议是我使用GPT5.5Pro检查后的, 建议在这里明确 PV 测点是 DC 侧、AC 侧、MPPT 侧还是逆变器输出侧，并记录 `pv_measurement_side` 和 `pv_formula_id`。原因：PV 功率若来自 DC 侧，与 AC 侧负载平衡之间还涉及逆变器效率和损耗；如果直接混用，会导致能量平衡残差和负载估计偏差。])


文件建议：`raw/pv_storage/Hxxxx/pv/pv_mppt_1s_YYYY-MM-DD.csv`

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `pv_system_id` | string | - | 每行 | 是 | 光伏系统 ID。 |
| `mppt_id` | string | - | 每行 | 推荐 | MPPT 通道。 |
| `pv_active_power_w` | number | W | 1 s | 是 | 光伏 DC/AC 输出功率，需说明测点。 |
| `pv_voltage_v` | number | V | 1 s | 推荐 | PV 电压。 |
| `pv_current_a` | number | A | 1 s | 推荐 | PV 电流。 |
| `pv_energy_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计发电量。 |
| `irradiance_w_m2` | number | W/m2 | 60 s | 推荐 | 辐照度；若无实测可提供天气 API 来源。 |
| `module_temperature_c` | number | C | 60 s | 可选 | 组件温度。 |
| `pv_limit_power_w` | number | W | 1-10 s | 推荐 | 限发功率。 |
| `curtailment_flag` | boolean | - | 1-10 s | 推荐 | 是否限功率/弃光。 |

### 5.9 家庭储能充放电数据

有储能家庭必填。用途：区分家庭负载、PV、储能充放电、电网交换；否则有光储家庭的 aggregate 会被严重污染。

(Xudong:[这条建议是我使用GPT5.5Pro检查后的, 建议补充 `battery_measurement_side`，例如 DC battery side、inverter DC bus、AC output side，并说明 `battery_power_w` 与 `charge_power_w`、`discharge_power_w` 是否为原始字段还是由正负号拆分得到。原因：电池侧功率和 AC 侧功率之间存在效率损耗，若口径不清，会把损耗错误归因到家庭负载或未知负载。])


文件建议：`raw/pv_storage/Hxxxx/storage/battery_power_1s_YYYY-MM-DD.csv`

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `storage_system_id` | string | - | 每行 | 是 | 储能系统 ID。 |
| `battery_pack_id` | string | - | 每行 | 推荐 | 电池包 ID。 |
| `battery_power_w` | number | W | 1 s | 是 | 正值放电，负值充电。 |
| `charge_power_w` | number | W | 1 s | 推荐 | 若内部口径分开记录则提供。 |
| `discharge_power_w` | number | W | 1 s | 推荐 | 若内部口径分开记录则提供。 |
| `battery_voltage_v` | number | V | 1 s | 推荐 | 电池电压。 |
| `battery_current_a` | number | A | 1 s | 推荐 | 电池电流。 |
| `battery_energy_charged_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计充电量。 |
| `battery_energy_discharged_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计放电量。 |
| `charge_source` | string | `pv/grid/mixed/unknown` | 1-10 s | 推荐 | 充电来源。 |
| `discharge_target` | string | `home/grid/backup/unknown` | 1-10 s | 推荐 | 放电去向。 |

### 5.10 电池 SoC 及相关状态数据

有储能家庭必填。用途：储能状态约束、EMS 策略解释、MPC/优化模型参数估计。


文件建议：`raw/pv_storage/Hxxxx/storage/battery_soc_1s_YYYY-MM-DD.csv`

| 字段名 | 类型 | 单位/枚举 | 推荐频率 | 必填 | 说明 |
|---|---|---|---:|---|---|
| `soc_percent` | number | % | 1-10 s | 是 | 电池 SoC。 |
| `soh_percent` | number | % | 60 s | 推荐 | 健康度。 |
| `battery_mode` | string | `idle/charging/discharging/standby/protect/fault` | 1-10 s | 是 | 电池状态。 |
| `min_soc_limit_percent` | number | % | 事件/60 s | 推荐 | 用户或系统设置的最低 SoC。 |
| `max_soc_limit_percent` | number | % | 事件/60 s | 推荐 | 最高 SoC。 |
| `reserve_soc_percent` | number | % | 事件/60 s | 推荐 | 备电保留 SoC。 |
| `cell_min_voltage_v` | number | V | 10-60 s | 推荐 | 单体最低电压。 |
| `cell_max_voltage_v` | number | V | 10-60 s | 推荐 | 单体最高电压。 |
| `cell_min_temperature_c` | number | C | 10-60 s | 推荐 | 单体最低温。 |
| `cell_max_temperature_c` | number | C | 10-60 s | 推荐 | 单体最高温。 |
| `bms_alarm_code` | string | - | 事件 | 有告警必填 | BMS 告警码。 |
| `bms_protection_code` | string | - | 事件 | 有保护必填 | BMS 保护码。 |

### 5.11 电网输入输出功率数据

有光储家庭必填；无光储家庭推荐。用途：识别买电、馈电、零馈网控制、负载真实功率。

文件建议：`raw/pv_storage/Hxxxx/meter_ct/grid_point_1s_YYYY-MM-DD.csv`

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `grid_power_w` | number | W | 1 s | 是 | 正值从电网输入家庭，负值向电网输出。 |
| `grid_import_power_w` | number | W | 1 s | 推荐 | 买电功率，非负。 |
| `grid_export_power_w` | number | W | 1 s | 推荐 | 馈电功率，非负。 |
| `grid_import_energy_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计买电量。 |
| `grid_export_energy_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计馈电量。 |
| `zero_export_target_w` | number | W | 1-10 s | 推荐 | 零馈网目标。 |
| `grid_connection_state` | string | `grid_tied/off_grid/islanding/backup/unknown` | 1-10 s/事件 | 是 | 并网/离网状态。 |
| `phase` | string | `total/L1/L2/L3` | 每行 | 推荐 | 相别。 |

### 5.12 逆变器运行状态、告警、保护、限功率、并网/离网日志

有光储/逆变器家庭必填。用途：解释功率突变、控制策略、保护停机、离网切换、限功率导致的 PV/储能异常。

(Xudong:[这条建议是我使用GPT5.5Pro检查后的, 建议同时要求提供 `state_machine.yaml` 或 `inverter_state_transition_table.csv`，描述 `standby`、`grid_tied`、`off_grid`、`backup`、`fault`、`protection`、`power_limit` 等状态的合法跃迁。原因：逆变器状态切换很容易被 NILM 模型误判为大功率负载事件；状态机能直接支持异常检测、样本切分和模型解释。])


状态快照文件：`raw/pv_storage/Hxxxx/inverter/inverter_status_1s_YYYY-MM-DD.csv`

| 字段名 | 类型 | 单位/枚举 | 推荐频率 | 必填 | 说明 |
|---|---|---|---:|---|---|
| `inverter_id` | string | - | 每行 | 是 | 逆变器 ID。 |
| `inverter_model` | string | - | 每文件/每日 | 是 | 例如 Hyper 2000 或内部型号。 |
| `inverter_mode` | string | `auto/expert/zenki/backup/off_grid/grid_tied/standby/fault` | 1 s | 是 | 工作模式。 |
| `ac_output_power_w` | number | W | 1 s | 是 | AC 侧输出功率。 |
| `dc_input_power_w` | number | W | 1 s | 推荐 | DC 输入功率。 |
| `mppt_power_w` | number | W | 1 s | 推荐 | MPPT 功率。 |
| `temperature_c` | number | C | 10 s | 推荐 | 逆变器温度。 |
| `efficiency_percent` | number | % | 10 s | 可选 | 实测或估算效率。 |
| `power_limit_w` | number | W | 1-10 s | 推荐 | 当前限功率值。 |
| `limit_reason` | string | - | 事件/10 s | 推荐 | 温度、电网、SoC、用户设置等。 |
| `alarm_active` | boolean | - | 1-10 s | 是 | 是否有告警。 |
| `protection_active` | boolean | - | 1-10 s | 是 | 是否保护中。 |

事件日志文件：`raw/pv_storage/Hxxxx/inverter/inverter_events_YYYY-MM.jsonl`

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `timestamp_utc` | datetime | 是 | 事件时间。 |
| `event_type` | string | 是 | `mode_change/grid_connect/grid_disconnect/off_grid_enter/off_grid_exit/alarm/protection/power_limit/firmware_update`。 |
| `event_code` | string | 推荐 | 内部事件码。 |
| `severity` | string | 推荐 | `info/warn/error/critical`。 |
| `old_state` | string/json | 推荐 | 事件前状态。 |
| `new_state` | string/json | 推荐 | 事件后状态。 |
| `recover_time_utc` | datetime | 可选 | 恢复时间。 |
| `related_command_id` | string | 推荐 | 关联 EMS/App 控制指令。 |

### 5.13 字段说明、采样频率、时间戳、单位、正负号和业务口径

必须交付以下说明文件：


| 文件 | 必填 | 内容 |
|---|---|---|
| `metadata/field_dictionary.csv` | 是 | 每个字段的英文名、中文名、类型、单位、枚举、是否必填、来源系统、计算公式、正负号含义。 |
| `metadata/sampling_policy.md` | 是 | 每类设备原始采样率、云端聚合周期、重采样规则、插值规则、缺口处理规则。 |
| `metadata/sign_convention.md` | 是 | 电网、PV、电池、逆变器、负载、CT 方向的正负号口径。 |
| `metadata/appliance_taxonomy.yaml` | 是 | 标准电器名、中文名、同义词、类别、默认 on/off 阈值。 |
| `metadata/firmware_versions.csv` | 推荐 | 设备型号、固件版本、升级时间、影响字段。 |
| `quality/*.csv/json` | 是 | 缺失率、重复时间戳、异常值、能量平衡、标签覆盖率报告。 |

`field_dictionary.csv` 示例：

```csv
field_name,field_name_cn,category,dtype,unit,required,frequency,positive_direction,source,description
aggregate_active_power_w,家庭总有功功率,aggregate,float,W,yes,1s,consumption_positive,smart_meter_or_ct,家庭总负载或总表测点有功功率
grid_power_w,电网交换功率,grid,float,W,yes,1s,import_positive,smart_meter_3ct,正值买电负值馈电
battery_power_w,电池功率,storage,float,W,yes,1s,discharge_positive,bms_or_inverter,正值放电负值充电
pv_active_power_w,光伏发电功率,pv,float,W,yes,1s,generation_positive,inverter_mppt,光伏输出功率
soc_percent,电池SoC,storage,float,%,yes,1-10s,na,bms,电池剩余容量百分比
```

## 6. 训练数据要求

本项目训练流程支持直接读取 CSV，要求如下：

(Xudong:[这条建议是我使用GPT5.5Pro检查后的, 建议在训练 CSV 之外增加 sidecar metadata，例如 `<appliance>_Hxxxx.meta.yaml`，记录 `house_id`、`house_system_type`、`operation_state`、`aggregate_type`、`sampling_interval_s`、`source_files`、`label_source`、`quality_policy`、`split_policy`、`preprocess_config` 和 `schema_version`。原因：只有 `time,aggregate,<appliance>,segment_id` 四列虽然方便训练，但不足以复现实验，也无法解释 aggregate 口径、标签来源和清洗策略。])


| 列位置 | 字段 | 类型 | 单位 | 说明 |
|---:|---|---|---|---|
| 1 | `time` | datetime/string | - | 可被 pandas 解析。 |
| 2 | `aggregate` | number | W | NILM 输入。 |
| 3 | `<appliance_name>` | number | W | 目标设备功率标签。 |
| 4 | `segment_id` | integer/string | - | 连续片段 ID；断点、缺口、重启、严重异常后必须新建片段。 |

示例：

```csv
time,aggregate,fridge,segment_id
2026-01-01 00:00:00,245.2,83.1,0
2026-01-01 00:00:01,247.8,84.0,0
2026-01-01 00:00:02,249.1,83.5,0
```

导出规则：

1. 文件名：`<appliance_name>_H<house_id>.csv`，例如 `fridge_H0001.csv`。
2. `aggregate` 必须是该家庭同一时间点的总负载功率，不应是设备功率之和的局部子集。
3. `<appliance_name>` 必须来自智能插座、子表、人工确认或高可信设备日志；若为弱标签推断，必须另附 `label_confidence`，不建议直接进入强监督训练。
4. 不同设备可以各自一个 CSV，也可以另交 `multi_target` 宽表用于后续多目标模型。
5. 缺口超过 `2 * sampling_interval_s`、设备离线、CT 方向异常、逆变器重启、固件升级、App 大规模策略切换等情况，必须切新 `segment_id`。

有光储家庭的 `aggregate` 推荐两种版本都交付：

| 文件 | `aggregate` 口径 | 用途 |
|---|---|---|
| `load_aggregate_<appliance>_Hxxxx.csv` | 估算家庭真实负载 | 主 NILM 训练。 |
| `grid_aggregate_<appliance>_Hxxxx.csv` | 并网点电网交换功率 | 研究 PV/储能扰动下的识别鲁棒性。 |

(Xudong:[建议新增 `splits.csv`，至少包含 `house_id`、`date_start`、`date_end`、`split`、`seen_household_flag`、`seen_appliance_model_flag`、`season`、`house_system_type`、`operation_state`。评估协议建议包含 within-house temporal split、cross-house split 和 cross-scenario split。原因：NILM 很容易因同一家庭相邻时间窗随机切分而数据泄漏，导致模型记住家庭基线和设备额定功率，不能反映跨家庭泛化能力。])


## 7. PV/储能/MPC 参数说明表

下表参考“参数类别、参数、描述”的写法，用于疆海补充储能优化、HEMS/MPC 或能量平衡建模所需参数。


| 参数类别 | 参数 | 描述 |
|---|---|---|
| 电池物理模型参数 | `B_cap`（电池标称容量） | 单位 kWh；例如 AB 系列电池包容量、系统总可用容量；需说明是标称容量还是可用容量。 |
| 电池物理模型参数 | `battery_chemistry` | 电芯体系，如 LFP/NCM；如不便披露可给类别。 |
| 电池物理模型参数 | `battery_pack_count` | 电池包数量和串并联系统构成。 |
| 功率约束参数 | `P_charge_max_w` | 最大充电功率，单位 W；需区分电池侧、AC 侧、PV 侧。 |
| 功率约束参数 | `P_discharge_max_w` | 最大放电功率，单位 W；需说明持续功率和峰值功率。 |
| 功率约束参数 | `P_grid_import_max_w` | 家庭或设备允许的最大电网输入功率。 |
| 功率约束参数 | `P_grid_export_max_w` | 最大馈网功率；若零馈网则为 0 或接近 0。 |
| 功率约束参数 | `P_inverter_ac_max_w` | 逆变器 AC 侧最大输出功率。 |
| 效率参数 | `eta_charge` | 充电效率；如随功率/温度变化，请提供曲线或典型值。 |
| 效率参数 | `eta_discharge` | 放电效率；如随功率/温度变化，请提供曲线或典型值。 |
| 效率参数 | `eta_roundtrip` | 往返效率；可由 `eta_charge * eta_discharge` 估计。 |
| 能量约束参数 | `E_min_kwh` | 电池最小允许能量；对应最低 SoC。 |
| 能量约束参数 | `E_max_kwh` | 电池最大允许能量；通常为可用容量上限。 |
| 能量约束参数 | `reserve_soc_percent` | 备电保留 SoC，由用户或 HEMS 设置。 |
| 自放电参数 | `self_discharge_rate_per_h` | 单位时间自放电率；若忽略请明确为 0。 |
| 时间参数 | `T_u` | 优化时间步长，例如 1 min、5 min、1 h。NILM 原始数据仍建议 1 s。 |
| 时间参数 | `control_horizon_steps` | HEMS/MPC 控制预测步数。 |
| PV 系统规格 | `pv_capacity_kwp` | 光伏装机容量，单位 kWp。 |
| PV 系统规格 | `pv_module_model` | 组件型号；如不便披露可给额定功率和数量。 |
| PV 系统规格 | `pv_string_count` | 组串数量。 |
| PV 系统规格 | `mppt_count` | MPPT 通道数量。 |
| PV 系统规格 | `azimuth_deg` | 方位角；可粗粒度。 |
| PV 系统规格 | `tilt_deg` | 倾角；可粗粒度。 |
| 环境参数 | `GHI_w_m2` | 全球水平辐照度；可来自天气 API 或本地传感器。 |
| 环境参数 | `ambient_temperature_c` | 环境温度。 |
| 环境参数 | `weather_condition` | 晴、阴、雨、雪等。 |
| HEMS 策略参数 | `operation_mode` | `zenki/auto/expert/manual/backup` 等模式。 |
| HEMS 策略参数 | `tariff_price` | 动态电价或分时电价，单位建议 CNY/kWh、EUR/kWh 或原币种/kWh。 |
| HEMS 策略参数 | `load_forecast_w` | HEMS 负载预测；若可提供，可用于和 NILM 结果对比。 |
| HEMS 策略参数 | `pv_forecast_w` | PV 预测功率。 |
| HEMS 策略参数 | `control_command` | 充电、放电、限功率、零馈网、备电保留等控制指令。 |

## 8. 数据质量验收标准

(Xudong:[建议把标题改为“数据质量分级与建模适用性评估”，避免使用“验收标准”作为甲方数据交付硬门槛。原因：真实 IoT 数据天然存在缺失、噪声、采样不均、标签不完整和设备异构；合同目标是研究开发和阶段性评估，不宜把理想数据质量写成绝对验收条件。])


| 检查项 | 合格标准 |
|---|---|
| 时间解析 | 100% 可解析；UTC 和本地时间能互相对应。 |
| 重复时间戳 | clean/aligned/train_ready 层不得存在重复时间戳。 |
| 缺失率 | 主训练通道日缺失率小于 1%；单次连续缺口超过 60 s 必须切分 segment。 |
| 单位一致性 | 功率统一 W，电量统一 kWh，电压 V，电流 A，温度 C。 |
| 正负号 | 电网、PV、电池、逆变器、负载必须有明确口径；抽样能量平衡可解释。 |
| 标签覆盖 | P0 设备每类至少 10 户以上强标签；每户连续 30 天以上优先。 |
| 设备绑定 | 智能插座与 appliance_id 的绑定变更必须有起止时间。 |
| 有光储能量平衡 | `load/grid/pv/battery` 在 1 h 聚合尺度上误差建议小于 5%-10%，超出需解释。 |
| 断点处理 | 停电、离线、固件升级、设备重启、CT 改向后必须切新 `segment_id`。 |
| 隐私 | 不包含直接个人身份信息和精确住址。 |


## 9. 首批数据交付优先级

第一批请优先交付以下最小可用集合，便于本项目快速跑通：

(Xudong:[这条建议是我使用GPT5.5Pro检查后的, 建议把第一批拆成“首批 0 字段样例包”和“首批 1 小样本联调包”。首批 0 只需每类场景 1 户、1 至 3 天，目标是确认字段、时间戳、单位、正负号、测点位置和导出格式；首批 1 再扩展到每类 3 至 5 户、7 至 14 天，目标是跑通 pipeline、质量报告、segment 切分和初版 baseline。原因：这样更符合合作启动节奏，也能尽早发现 CT 方向、时间戳、字段口径等根本问题。])


1. `metadata/households.csv`、`appliances.csv`、`meters.csv`、`field_dictionary.csv`、`sign_convention.md`。
2. 无光储家庭 3-5 户，每户 14 天：`aggregate`、`smart_meter/CT`、至少 5 类智能插座强标签、设备日志、App 日志。
3. 有光储家庭 3-5 户，每户 14 天：上述全部数据 + PV、储能功率、SoC、并网点功率、逆变器状态/告警/保护/限功率/并离网日志、HEMS 控制日志。
4. 同步提供 `train_ready/project_training_csv/`，至少包含 `fridge`、`washing_machine`、`dishwasher`、`water_heater`、`air_conditioner_or_heat_pump` 五类设备。

(Xudong:[这条建议是我使用GPT5.5Pro检查后的, 建议把“至少包含五类设备”改为“优先包含 3 至 5 类可稳定计量设备”，且标签来源不限于智能插座，也可以是子表、回路 CT、设备遥测或高可信控制日志。原因：空调、热水器、热泵等高价值负载不一定有插座级标签，过硬的五类要求可能阻碍首批联调。])

5. 每个 CSV 必须带 `segment_id`，每个字段必须能在字段字典中查到单位、含义和来源。

## 10. 需要疆海确认的问题


1. Smart Meter 3CT/Satellite Monitor CT 在现有家庭中的默认安装位置：总进线、逆变器侧、负载侧，还是其他位置？
2. 现有云平台可导出的最小时间粒度是多少：1 s、5 s、10 s、1 min，还是只保留聚合值？
3. 设备端和云端是否同时保存设备采样时间与服务器入库时间？
4. 智能插座是否能稳定绑定到具体电器？绑定变更是否有历史记录？
5. 有光储场景下，家庭真实负载是否已有内部计算字段？公式是什么？
6. 电池功率正负号、并网点功率正负号、CT 方向在不同固件版本中是否一致？
7. HEMS/Zenki/Auto/Expert 模式的策略日志和控制指令是否可导出？
8. 逆变器、BMS、网关、智能插座的告警码/保护码是否有码表？
9. 用户反馈数据是否可匿名导出，是否包含用户纠错或设备命名数据？
10. 是否可提供少量高频波形或事件片段，用于后续提升设备特征识别能力？
