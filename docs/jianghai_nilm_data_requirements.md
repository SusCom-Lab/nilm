# 广州疆海科技 NILM 项目数据提供需求说明

版本：v2.0   
适用项目：家庭负载 NILM 识别；分为“无光伏储能家庭”和“有光伏/储能/逆变器家庭”两类场景。  
交付对象：广州疆海科技有限公司（Zendure/征拓）数据、云平台、App、硬件、算法及测试团队。

## 1. 背景与结论

本项目目标是训练和验证 NILM（Non-Intrusive Load Monitoring，非侵入式负荷识别）模型：以家庭总负载、智能电表/CT 等总表数据为输入，识别冰箱、洗衣机、洗碗机、空调、热泵、热水器、厨房电器、照明、待机负载等设备级用电功率或运行状态。

根据项目合同目标和前期沟通，本项目覆盖无光伏储能和有光伏/储能两类家庭能源场景。无光储场景重点关注家庭总负载、总表/CT、设备级标签和用户/设备日志；有光储场景在此基础上，还需要明确 PV、储能、电网交换、逆变器和能源管理策略对家庭总负载口径的影响。

因此，本项目向疆海索取的数据不能只停留在“总功率 CSV”。需要同时覆盖：

1. 无光伏储能场景：家庭总负载 + 电表/CT + 智能插座 + 设备/用户/App 日志 + 设备标签。
2. 有光伏储能场景：在上述基础上增加 PV 发电、储能充放电、电池 SoC/BMS、并网点输入输出、逆变器/EMS 状态与控制日志。
3. 数据工程支撑：字段字典、采样频率、时间戳规则、单位、正负号、业务口径、质量报告和数据血缘。

## 2. 总体交付原则

### 2.1 交付优先级分级

为便于项目快速启动，建议按 M0-M3 分阶段交付，不要求第一轮一次性提供全部理想字段。

| 级别 | 名称 | 数据范围 | 目标 |
|---|---|---|---|
| M0 | 字段样例包 | 每类场景 1 户，1-3 天，原始字段样例 + 字段字典 + 测点说明 | 确认字段、单位、时间戳、正负号、测点位置和导出格式。 |
| M1 | 小样本联调包 | 每类场景 3-5 户，7-14 天，raw + 必要 metadata + 若干设备标签 | 跑通读取、格式转换、时间对齐、质量报告、segment 切分和初版 baseline。 |
| M2 | baseline 训练包 | 每类场景建议 20 户以上，60-90 天，稳定总表和高可信设备标签 | 训练和评估第一版 NILM baseline。 |
| M3 | 长期 benchmark 增强包 | 每类场景 50-100 户以上，180-365 天，覆盖季节、电价、策略和设备差异 | 做跨家庭、跨季节、跨场景泛化评估。 |

### 2.2 场景与运行状态标记

| 字段 | 含义 | 推荐枚举/示例 | 填写粒度 |
|---|---|---|---|
| `house_system_type` | 家庭硬件形态 | `no_pv_no_storage`、`pv_only`、`storage_only`、`pv_storage`、`unknown` | 家庭级 metadata 必填；时间序列中建议冗余带上 |
| `operation_state` | 某时刻或某时间段的运行状态 | `grid_tied`、`off_grid`、`backup`、`islanding`、`zero_export`、`charging`、`discharging`、`curtailment`、`fault`、`unknown` | 时间序列/事件日志建议提供；无状态数据时填 `unknown` |

字段填写说明：

| 字段值 | 说明 |
|---|---|
| `no_pv_no_storage` | 家庭没有光伏和家庭储能，只有市电输入和家庭负载；这是最适合直接做基础 NILM 的场景。 |
| `pv_only` | 家庭有光伏，但没有电池储能；需要同时提供 PV 发电数据，否则总表功率可能受发电影响。 |
| `storage_only` | 家庭没有光伏，但有电池储能；需要提供电池充放电功率和 SoC。 |
| `pv_storage` | 家庭同时有光伏和储能；需要提供 PV、储能、电网交换、逆变器状态等数据。 |
| `grid_tied` | 并网运行，家庭与电网保持连接。 |
| `off_grid` | 离网运行，家庭负载不从电网取电。 |
| `backup` | 备电/应急供电状态，通常由储能或逆变器给关键负载供电。 |
| `islanding` | 孤岛运行状态，需与并网状态明确区分。 |
| `zero_export` | 零馈网或限制馈网状态，PV/储能输出可能被控制策略限制。 |
| `charging` | 储能正在充电。 |
| `discharging` | 储能正在放电。 |
| `curtailment` | 光伏或逆变器处于限功率/弃光状态。 |
| `fault` | 设备故障、保护、告警或异常运行状态。 |
| `unknown` | 当前无法确认；允许出现在原始层，清洗和建模前应尽量补充或标记为低可信。 |

无光储家庭通常为 `house_system_type=no_pv_no_storage`，`operation_state=grid_tied` 或 `unknown`。有光储家庭必须额外明确并网、离网、备电、充放电、限功率、故障等状态，否则并网点功率不能直接当作家庭负载。

如果一个时间点同时存在多种运行状态，例如并网且电池充电，可在 `operation_state` 中使用主状态，并在补充字段中提供 `battery_mode`、`grid_connection_state`、`curtailment_flag`、`zero_export_target_w` 等细分状态。

### 2.3 责任边界

企业侧首要提供：

1. `raw/`：从云平台、设备、App、测试系统直接导出的原始数据，保留原貌。
2. `metadata/`：字段字典、测点拓扑、设备绑定、单位、采样、正负号、可用性矩阵等必要说明。
3. 可选补充：企业内部已有的清洗版、聚合版或训练版数据。

项目侧基于企业原始数据和 metadata 生成：

1. `clean/`：统一字段、单位、时间戳、正负号和质量标签后的清洗数据。
2. `aligned/`：按家庭、测点和时间轴对齐后的宽表数据。
3. `train_ready/`：适配本项目训练流程的 CSV，形如 `time,aggregate,<appliance>,segment_id`。
4. `quality/`：缺失、乱序、重复、异常值、标签覆盖、能量平衡和建模适用性报告。

### 2.4 采样频率要求

| 数据类型 | 最低可用 | 推荐 | 高价值增强 |
|---|---:|---:|---:|
| 家庭总负载/智能电表/CT 有功功率 | 10 s | 1 s | 100 Hz 至 1 kHz 以上波形或 RMS |
| 智能插座设备级功率标签 | 10 s | 1 s | 100 Hz 以上事件片段 |
| PV/储能/并网功率 | 10 s | 1 s | 100 ms 至 1 s 控制闭环数据 |
| 电池 SoC/BMS 状态 | 60 s | 1 s 至 10 s | 事件触发高频日志 |
| 逆变器/EMS/告警/保护日志 | 事件级 | 事件级 + 1 s 状态快照 | 控制指令前后 30 s 高频片段 |
| App 操作日志 | 事件级 | 事件级 | 操作前后能源状态快照 |

5 分钟或 15 分钟聚合数据不适合严格的细粒度 appliance-level NILM 主训练，但仍可用于日内行为画像、能量平衡、异常检测、用电推荐、储能策略分析和业务统计。

### 2.5 数据周期和样本量

| 阶段 | 家庭数量 | 连续天数 | 目的 |
|---|---:|---:|---|
| 小样本联调 | 每类场景 3-5 户 | 14 天以上 | 验证字段、时间同步、正负号、文件结构和 pipeline。 |
| 第一版训练 | 无光储 20 户以上；有光储 20 户以上 | 60-90 天 | 覆盖工作日/周末、晴雨天、季节变化和常用电器。 |
| 稳定评估 | 每类场景 50 户以上 | 180 天以上 | 评估跨家庭泛化、PV/储能扰动、设备迁移能力。 |
| 长周期优化 | 每类场景 100 户以上 | 365 天 | 支持季节性设备、热泵/空调、储能策略、动态电价场景。 |

## 3. 数据文件树与责任划分

本节将文件树分为两部分：**疆海需交付的数据包** 和 **项目侧处理产物**。疆海首批交付重点是 `metadata/` 和 `raw/`；`clean/`、`aligned/`、`train_ready/`、`quality/` 通常由项目侧基于原始数据生成，若疆海已有内部处理版，可作为补充材料提供。

`H0001` 表示家庭匿名编号，不应使用真实姓名、手机号、详细地址。

### 3.1 疆海需交付的数据包

```text
jianghai_nilm_delivery/
  README.md                         # 数据包说明：交付批次、覆盖场景、联系人、导出时间、已知限制
  manifest.yaml                     # 文件清单：每个文件的路径、大小、时间范围、记录数、校验值
  data_license_and_privacy.md       # 数据授权、脱敏范围、隐私边界、允许使用范围

  metadata/                         # 元数据说明文件；用于解释 raw 数据，不是训练数据本身
    dataset.yaml                    # 数据集级说明：批次、覆盖户数、时间范围、时区、总体采样情况
    data_availability_matrix.csv    # 数据可用性矩阵：哪些数据可提供、覆盖户数、时间范围、最小粒度
    field_dictionary.csv            # 字段字典：字段名、中文名、单位、类型、枚举、来源、计算口径
    sampling_policy.md              # 采样说明：原始采样率、云端聚合周期、缺失/回填规则
    measurement_points.csv          # 测点拓扑：CT/电表/PV/电池/逆变器位置、相别、方向、上下游关系
    sign_convention.md              # 正负号说明：电网、负载、PV、电池、逆变器各字段正负方向
    households.csv                  # 家庭基础信息：匿名家庭 ID、house_system_type、时区、相制等
    meters.csv                      # 电表/CT 设备信息：设备 ID、型号、安装时间、通道信息
    circuits.csv                    # 回路信息：回路 ID、名称、覆盖设备、相别
    appliances.csv                  # 电器清单：设备 ID、类型、额定功率、标签来源、绑定时间
    smart_plugs.csv                 # 智能插座/插排信息：插座 ID、绑定设备、安装时间
    pv_systems.csv                  # 光伏系统信息：装机容量、MPPT/组串、测量侧、安装信息；无光伏可不提供
    storage_systems.csv             # 储能系统信息：电池容量、功率约束、SoC 规则；无储能可不提供
    inverters.csv                   # 逆变器信息：型号、额定功率、并网/离网能力；无逆变器可不提供
    tariff_plans.csv                # 电价信息：分时电价/动态电价；如无相关策略可不提供
    firmware_versions.csv           # 固件版本：设备型号、固件版本、升级时间、影响字段
    appliance_taxonomy.yaml         # 电器命名规范：标准名、中文名、同义词、类别
    state_machine.yaml              # 设备/逆变器状态机；有光储或复杂设备建议提供
    inverter_state_transition_table.csv # 逆变器状态跃迁表；有逆变器家庭建议提供
    splits.csv                      # 若疆海已有训练/测试划分偏好，可提供；否则由项目侧生成

  raw/                              # 疆海原始导出数据；原则上不修改、不覆盖、不人工清洗
    no_pv_storage/                  # 无光伏、无储能家庭
      H0001/
        aggregate/                  # 家庭总负载或总表功率数据，是 NILM 的主要输入来源
          mains_1s_2026-01-01.csv
          mains_1s_2026-01-02.csv
        meter_ct/                   # 智能电表/CT 数据，用于校验总表、相别、方向和测点位置
          smart_meter_3ct_1s_2026-01-01.csv
          phase_l1_1s_2026-01-01.csv
          phase_l2_1s_2026-01-01.csv
          phase_l3_1s_2026-01-01.csv
        smart_plug/                 # 智能插座或插排数据，可作为设备级高可信标签来源
          plug_P0001_fridge_1s_2026-01-01.csv
          plug_P0002_washing_machine_1s_2026-01-01.csv
        device_logs/                # 设备状态/故障/固件/离线重连日志，用于解释异常和切分片段
          device_status_events_2026-01.jsonl
          firmware_events_2026-01.jsonl
        app_logs/                   # 用户 App 操作日志，用于解释人工开关、模式切换和策略变化
          app_user_events_2026-01.jsonl
        labels/                     # 电器级标签、弱标签、用户反馈或人工标注
          strong_labels_2026-01.csv
          weak_labels_2026-01.csv
          user_feedback_2026-01.csv

    pv_storage/                     # 有光伏/储能/逆变器家庭
      H1001/
        aggregate/                  # 家庭总负载或总表功率；需说明是否为真实负载还是并网点功率
          mains_1s_2026-01-01.csv
        meter_ct/                   # 电表/CT/并网点数据；有光储场景必须说明测点位置和方向
          smart_meter_3ct_1s_2026-01-01.csv
          grid_point_1s_2026-01-01.csv
        smart_plug/                 # 插座级设备标签；只覆盖可插座计量设备
          plug_P0001_fridge_1s_2026-01-01.csv
        pv/                         # 光伏发电数据；需说明 DC 侧、MPPT 侧还是 AC 侧
          pv_string_1s_2026-01-01.csv
          pv_mppt_1s_2026-01-01.csv
        storage/                    # 储能充放电、SoC、BMS 状态；用于还原真实负载口径
          battery_power_1s_2026-01-01.csv
          battery_soc_1s_2026-01-01.csv
          bms_status_events_2026-01.jsonl
        inverter/                   # 逆变器运行状态、告警、保护、限功率、并网/离网日志
          inverter_status_1s_2026-01-01.csv
          inverter_events_2026-01.jsonl
          protection_alarm_events_2026-01.jsonl
        hems/                       # 能源管理系统计划、控制指令、预测输入
          energy_plan_2026-01.jsonl
          control_commands_2026-01.jsonl
          forecast_inputs_2026-01.csv
        app_logs/                   # 用户 App 操作日志
          app_user_events_2026-01.jsonl
        labels/                     # 电器级标签、弱标签、用户反馈或人工标注
          strong_labels_2026-01.csv
          weak_labels_2026-01.csv
          user_feedback_2026-01.csv
```

### 3.2 项目侧处理产物

以下目录通常由项目侧基于疆海交付的 `raw/` 和 `metadata/` 生成，不要求疆海首批必须交付。若疆海已有内部清洗版或训练版，可作为补充提供，项目侧会再做一致性校验。

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
        fridge_H0001.csv            # 典型列：time, aggregate, fridge, segment_id
        washing_machine_H0001.csv
        dishwasher_H0001.csv
      pv_storage/
        fridge_H1001.csv
        washing_machine_H1001.csv
        heat_pump_H1001.csv
    multi_target/                   # 多目标宽表，供后续多电器联合建模使用
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

## 4. 通用字段要求

所有 CSV/Parquet 时间序列建议包含以下公共字段或等价伴随字段。

| 字段名 | 类型 | 单位/枚举 | 必填 | 说明 |
|---|---|---|---|---|
| `timestamp_utc` | datetime/string | ISO 8601, UTC | 是 | 统一时间轴，例如 `2026-01-01T00:00:00.000Z`。 |
| `timestamp_device_utc` | datetime/string | UTC | 推荐 | 设备侧采样时间。 |
| `timestamp_server_utc` | datetime/string | UTC | 推荐 | 云端接收或入库时间。 |
| `timestamp_ingest_utc` | datetime/string | UTC | 推荐 | 本批数据导出或项目侧入库时间。 |
| `timestamp_local` | datetime/string | 本地时间 | 是 | 用户所在时区的本地时间，用于行为模式分析。 |
| `timezone` | string | IANA TZ | 是 | 例如 `Europe/Berlin`、`Asia/Shanghai`。 |
| `house_id` | string | `H0001` | 是 | 匿名家庭编号。 |
| `house_system_type` | string | 见 2.2 | 是 | 家庭硬件形态。 |
| `operation_state` | string | 见 2.2 | 推荐 | 运行状态，可按时间变化。 |
| `source_system` | string | - | 推荐 | App、云平台、设备固件、测试平台、API 等。 |
| `source_device_id` | string | 匿名 ID | 是 | 采集设备、插座、逆变器、电池或 App 侧 ID。 |
| `source_channel_id` | string | 匿名 ID | 是 | CT 相线、插座通道、MPPT 通道、电池包通道等。 |
| `measurement_point_id` | string | 测点 ID | 推荐 | 对应 `metadata/measurement_points.csv`。 |
| `source_sequence_id` | string/int | - | 推荐 | 源系统序列号或采集顺序。 |
| `ingest_order` | integer | - | 推荐 | 导出文件中的原始顺序。 |
| `clock_drift_ms` | number | ms | 可选 | 估计设备时钟漂移。 |
| `availability_status` | string | `available/missing/not_collected/not_supported/redacted` | 推荐 | 字段或记录可用状态。 |
| `sampling_interval_s` | number | s | 是 | 该记录所属文件的目标采样间隔。 |
| `schema_version` | string | - | 推荐 | 字段 schema 版本。 |
| `ingest_batch_id` | string | 批次 ID | 是 | 便于追溯本次交付批次。 |

时间戳规则：

1. 必须提供 UTC 时间戳，不允许只给本地时间或设备相对时间。
2. raw 层保留源系统原貌，可存在重复、乱序、缺口和回填，但必须保留 `source_sequence_id` 或 `ingest_order`。
3. clean、aligned、train_ready 层要求同一 `house_id`、`source_channel_id`、`timestamp_utc` 下无重复且时间单调递增。
4. 数据缺口不得用 0 填充；缺失就是缺失，修复层用 `quality_flag_primary=interpolated`、`is_interpolated=true` 等字段标记。
5. 如果云端入库时间和设备采样时间不同，必须同时给 `timestamp_device_utc` 和 `timestamp_server_utc`。

质量标签建议：

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `quality_flag_primary` | string | 是 | 主质量状态：`valid/missing/interpolated/outlier/estimated/invalid`。 |
| `is_missing` | boolean | 推荐 | 是否缺失。 |
| `is_interpolated` | boolean | 推荐 | 是否插值或前向填充。 |
| `is_outlier` | boolean | 推荐 | 是否异常值。 |
| `is_clipped` | boolean | 推荐 | 是否被裁剪。 |
| `is_estimated` | boolean | 推荐 | 是否为推导或估算值。 |
| `is_device_offline` | boolean | 推荐 | 设备是否离线。 |
| `is_clock_suspect` | boolean | 推荐 | 时间戳是否可疑。 |
| `quality_score` | number | 推荐 | 0-1 质量分。 |
| `quality_reason_code` | string | 推荐 | 原因码，例如 `long_gap/duplicate_timestamp/ct_direction_changed`。 |

功率正负号总规则：

| 字段类别 | 推荐正负号口径 |
|---|---|
| 家庭负载 `load_power_w` | 正值表示家庭消耗功率；理论上不应为负。 |
| 电网 `grid_power_w` | 正值表示从电网买电/输入家庭；负值表示向电网馈电。 |
| PV `pv_power_w` | 正值表示光伏发电输出。 |
| 电池 `battery_power_w` | 正值表示电池放电供家庭/电网；负值表示电池充电。 |
| 逆变器 AC 输出 `inverter_ac_power_w` | 正值表示向家庭 AC 侧输出；负值表示从 AC 侧吸收。 |
| 智能插座 `plug_power_w` | 正值表示插座下游设备消耗。 |

若疆海内部已有相反口径，请不要直接改值，应同时提供原始字段和 `sign_convention.md`，再在 clean 层统一。正负号说明应尽量绑定到 `measurement_points.csv` 中的具体测点，而不仅是全局规则，因为同一个字段名在不同 CT 安装位置、相别、逆变器 AC/DC 侧可能含义不同。

`metadata/data_availability_matrix.csv` 建议记录每类数据是否可提供、覆盖户数、时间范围、最小粒度、导出格式和已知限制；`metadata/measurement_points.csv` 建议记录 CT/电表/逆变器/PV/电池等测点位置、相别、方向、上下游关系和正负号。

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
| `quality_flag_primary` | string | - | 每行 | 是 | 主质量标记。 |

重要业务口径：

1. 无光储：`aggregate_active_power_w` 应等于家庭负载总消耗。
2. 有光储：总表位置可能测到“电网交换功率”而不是“家庭真实负载”。必须明确 CT 安装位置和测点拓扑。若存在负载侧直接测点，优先提供 `measured_load_power_w`；若只有并网点、PV、电池和逆变器数据，则通过 `measurement_points.csv` 中的拓扑和 `formula_id` 推导 `load_estimated_power_w`，同时记录 `input_fields`、`loss_assumption`、`quality_flag_primary` 和 `energy_balance_residual`。不建议在交付前写死单一公式。

### 5.2 智能电表/CT 数据

用途：校验家庭总负载、三相平衡、并网点功率、CT 安装方向；Smart Meter 3CT/Satellite Monitor CT 类产品应重点提供。

文件建议：`raw/<scenario>/Hxxxx/meter_ct/smart_meter_3ct_1s_YYYY-MM-DD.csv`

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `measurement_point_id` | string | - | 每行 | 是 | 测点 ID，需能在 `measurement_points.csv` 查到。 |
| `meter_id` | string | - | 每行 | 是 | 电表或 CT 设备匿名 ID。 |
| `meter_model` | string | - | 每文件/每行 | 是 | 例如 Smart Meter 3CT、Satellite Monitor CT 或内部型号。 |
| `phase` | string | `total/L1/L2/L3` | 每行 | 是 | 三相或单相。 |
| `bus_or_node` | string | - | 每文件/每行 | 推荐 | 测点所在母线或节点。 |
| `upstream_component` | string | - | 每文件/每行 | 推荐 | 测点上游组件。 |
| `downstream_component` | string | - | 每文件/每行 | 推荐 | 测点下游组件。 |
| `measures_load_or_grid_or_pv_or_battery` | string | `load/grid/pv/battery/inverter/circuit/unknown` | 每文件/每行 | 是 | 测量对象。 |
| `ct_direction` | string | `import_positive/export_positive/unknown` | 每文件 | 是 | CT 安装方向和正负号。 |
| `ct_ratio` | number | A/A | 每文件 | 推荐 | CT 变比，例如 120A CT。 |
| `is_bidirectional` | boolean | - | 每文件/每行 | 推荐 | 是否双向功率测点。 |
| `installed_start_utc` | datetime | UTC | 每文件/事件 | 推荐 | 测点安装或生效开始时间。 |
| `installed_end_utc` | datetime | UTC | 每文件/事件 | 推荐 | 测点安装或生效结束时间。 |
| `active_power_w` | number | W | 1 s | 是 | 有功功率。 |
| `reactive_power_var` | number | var | 1 s | 推荐 | 无功功率。 |
| `voltage_v` | number | V | 1 s | 推荐 | 相电压。 |
| `current_a` | number | A | 1 s | 推荐 | 相电流。 |
| `power_factor` | number | - | 1 s | 推荐 | 功率因数。 |
| `import_energy_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计正向电量。 |
| `export_energy_kwh_total` | number | kWh | 10-60 s | 有光储必填 | 累计反向电量。 |
| `rssi_dbm` | number | dBm | 60 s | 可选 | 设备通信质量。 |
| `firmware_version` | string | - | 事件/每日 | 推荐 | 方便排查固件差异。 |

### 5.3 高可信设备级标签数据

用途：监督式 NILM 的设备级 ground truth。标签来源不限于智能插座，也可以是子表、回路 CT、设备遥测、BMS/逆变器日志、App 控制日志或人工确认，并通过 `label_source` 和 `label_confidence` 标明可信度。没有设备级高可信标签时，模型只能做弱监督/无监督，精度和评估可信度会明显下降。

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

首批优先评估可获得标签的设备，按计量方式分为：

| 类别 | 设备示例 | 标签来源示例 |
|---|---|
| `plug_metered_appliance` | 冰箱、洗衣机、洗碗机、微波炉、电水壶 | 智能插座、智能插排。 |
| `circuit_metered_or_submetered` | 空调、热泵、热水器、照明回路、EV 充电器 | 子表、回路 CT。 |
| `system_telemetry_labeled` | 逆变器、储能、HEMS 可控负载 | 设备遥测、BMS/逆变器日志、控制日志。 |

### 5.4 设备运行日志

用途：构造弱标签、解释负荷变化、排查异常；尤其对储能、逆变器、智能插座、网关、App 控制设备重要。

文件建议：`raw/<scenario>/Hxxxx/device_logs/device_status_events_YYYY-MM.jsonl`

每行 JSON 示例：

```json
{"timestamp_utc":"2026-01-01T08:30:12.400Z","house_id":"H0001","device_id":"D0001","device_type":"smart_plug","event_type":"relay_on","old_value":"off","new_value":"on","trigger_source":"app","firmware_version":"1.2.3","quality_flag_primary":"raw"}
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
| `house_system_type` | string | 是 | 见 2.2。 |
| `region_or_climate_zone` | string | 推荐 | 区域或气候带，不要详细地址。 |
| `timezone` | string | 是 | IANA 时区。 |
| `grid_phase_type` | string | 是 | `single_phase/three_phase/split_phase/unknown`。 |
| `nominal_voltage_v` | number | 推荐 | 例如 230。 |
| `floor_area_band` | string | 推荐 | 面积区间，如 `50-80m2`。 |
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
| `label_source` | string | 推荐 | 标签来源，如 `smart_plug/submeter/circuit_ct/device_telemetry/manual_annotation`。 |
| `label_confidence` | number | 推荐 | 0-1 标签可信度。 |
| `metered_by_channel_id` | string | 强标签必填 | 对应智能插座、子表、回路 CT 或遥测通道。 |
| `label_available` | boolean | 是 | 是否有强标签。 |
| `binding_start_utc` | datetime | 推荐 | 设备与计量通道绑定起始时间。 |
| `binding_end_utc` | datetime | 推荐 | 设备与计量通道绑定结束时间。 |

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

用户反馈默认只提供结构化类别，不提供原始自由文本。如果确有必要使用文本，应先由疆海侧完成脱敏、敏感词过滤和发布范围审批。

### 5.8 光伏发电数据

有光伏家庭必填。用途：从并网点/总表数据中剥离 PV 扰动，识别真实负载；也用于 PV+储能能量平衡。

文件建议：`raw/pv_storage/Hxxxx/pv/pv_mppt_1s_YYYY-MM-DD.csv`

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `pv_system_id` | string | - | 每行 | 是 | 光伏系统 ID。 |
| `mppt_id` | string | - | 每行 | 推荐 | MPPT 通道。 |
| `pv_active_power_w` | number | W | 1 s | 是 | 光伏 DC/AC 输出功率，需说明测点。 |
| `pv_measurement_side` | string | `dc_string_side/mppt_side/inverter_ac_side/unknown` | 每行/每文件 | 是 | PV 测量侧。 |
| `pv_formula_id` | string | - | 每行/每文件 | 推荐 | 参与负载推导的公式 ID。 |
| `pv_voltage_v` | number | V | 1 s | 推荐 | PV 电压。 |
| `pv_current_a` | number | A | 1 s | 推荐 | PV 电流。 |
| `pv_energy_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计发电量。 |
| `irradiance_w_m2` | number | W/m2 | 60 s | 推荐 | 辐照度；若无实测可提供天气 API 来源。 |
| `module_temperature_c` | number | C | 60 s | 可选 | 组件温度。 |
| `pv_limit_power_w` | number | W | 1-10 s | 推荐 | 限发功率。 |
| `curtailment_flag` | boolean | - | 1-10 s | 推荐 | 是否限功率/弃光。 |

### 5.9 家庭储能充放电数据

有储能家庭必填。用途：区分家庭负载、PV、储能充放电、电网交换；否则有光储家庭的 aggregate 会被严重污染。

文件建议：`raw/pv_storage/Hxxxx/storage/battery_power_1s_YYYY-MM-DD.csv`

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `storage_system_id` | string | - | 每行 | 是 | 储能系统 ID。 |
| `battery_pack_id` | string | - | 每行 | 推荐 | 电池包 ID。 |
| `battery_power_w` | number | W | 1 s | 是 | 正值放电，负值充电。 |
| `battery_measurement_side` | string | `dc_battery_side/inverter_dc_bus/ac_output_side/unknown` | 每行/每文件 | 是 | 电池功率测量侧。 |
| `charge_power_w` | number | W | 1 s | 推荐 | 若内部口径分开记录则提供；若由 `battery_power_w` 拆分需说明。 |
| `discharge_power_w` | number | W | 1 s | 推荐 | 若内部口径分开记录则提供；若由 `battery_power_w` 拆分需说明。 |
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

建议同时提供 `metadata/state_machine.yaml` 或 `metadata/inverter_state_transition_table.csv`，描述 `standby`、`grid_tied`、`off_grid`、`backup`、`fault`、`protection`、`power_limit` 等状态的合法跃迁。逆变器状态切换容易被 NILM 模型误判为大功率负载事件，状态机可直接支持异常检测、样本切分和模型解释。

### 5.13 字段说明、采样频率、时间戳、单位、正负号和业务口径

必须交付以下说明文件：

| 文件 | 必填 | 内容 |
|---|---|---|
| `metadata/field_dictionary.csv` | 是 | 每个字段的英文名、中文名、类型、单位、枚举、是否必填、来源系统、计算公式、正负号含义。 |
| `metadata/sampling_policy.md` | 是 | 每类设备原始采样率、云端聚合周期、重采样规则、插值规则、缺口处理规则。 |
| `metadata/measurement_points.csv` | 是 | CT/电表/逆变器/PV/电池等测点位置、相别、方向、上下游关系、正负号和生效时间。 |
| `metadata/sign_convention.md` | 是 | 全局正负号口径；具体测点以 `measurement_points.csv` 为准。 |
| `metadata/data_availability_matrix.csv` | 是 | 每类数据是否可提供、覆盖户数、时间范围、最小粒度、导出格式和已知限制。 |
| `metadata/appliance_taxonomy.yaml` | 是 | 标准电器名、中文名、同义词、类别、默认 on/off 阈值。 |
| `metadata/firmware_versions.csv` | 推荐 | 设备型号、固件版本、升级时间、影响字段。 |
| `metadata/splits.csv` | 推荐 | train/val/test 的家庭、日期、场景、是否跨家庭划分等评估协议。 |
| `metadata/state_machine.yaml` 或 `metadata/inverter_state_transition_table.csv` | 有光储推荐 | 逆变器、储能、HEMS 状态及合法跃迁。 |
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

每个训练 CSV 建议配套 sidecar metadata，例如 `<appliance>_Hxxxx.meta.yaml`，用于复现实验和解释 aggregate 口径：

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

有光储家庭的 `aggregate` 推荐两种版本都交付：

| 文件 | `aggregate` 口径 | 用途 |
|---|---|---|
| `load_aggregate_<appliance>_Hxxxx.csv` | 估算家庭真实负载 | 主 NILM 训练。 |
| `grid_aggregate_<appliance>_Hxxxx.csv` | 并网点电网交换功率 | 研究 PV/储能扰动下的识别鲁棒性。 |

建议提供或由项目侧生成 `metadata/splits.csv`，至少包含 `house_id`、`date_start`、`date_end`、`split`、`seen_household_flag`、`seen_appliance_model_flag`、`season`、`house_system_type`、`operation_state`。评估协议建议包含 within-house temporal split、cross-house split 和 cross-scenario split，避免同一家庭相邻时间窗随机切分造成数据泄漏。

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

## 8. 数据质量分级与建模适用性评估

真实 IoT 数据天然存在缺失、噪声、采样不均、标签不完整和设备异构。建议将数据质量作为建模适用性分级，而不是第一轮交付的绝对硬门槛。

| 等级 | 含义 | 适用用途 |
|---|---|
| A | 时间连续、总表和标签稳定、缺失低、测点口径清晰 | 主训练和正式评估。 |
| B | 有少量缺失或短时异常，可修复 | baseline 训练和消融实验。 |
| C | 标签少、采样较低或测点口径部分缺失 | 弱监督、统计分析、策略分析。 |
| D | 缺失严重、口径不明、无法对齐 | 仅用于问题诊断，不进入训练。 |

关键检查项包括：时间解析、重复时间戳、缺失率、单位一致性、正负号、标签覆盖、设备绑定、CT 改向或迁移、有光储能量平衡、隐私合规、长缺口和 `segment_id` 切分。

## 9. 首批数据交付优先级

建议首批拆成两个阶段，降低启动成本并尽早暴露字段、时间戳、单位、正负号和测点问题。

1. M0 字段样例包：每类场景 1 户、1-3 天；优先提供原始字段样例、`field_dictionary.csv`、`data_availability_matrix.csv`、`measurement_points.csv`、`sampling_policy.md`、`sign_convention.md`。
2. M1 小样本联调包：每类场景 3-5 户、7-14 天；无光储提供 `aggregate`、`smart_meter/CT`、3-5 类可稳定计量设备标签、设备日志和 App 日志；有光储在此基础上增加 PV、储能功率、SoC、并网点功率、逆变器状态/告警/保护/限功率/并离网日志、HEMS 控制日志。
3. 训练 CSV 若由疆海侧提供，建议同步提供 `train_ready/project_training_csv/` 和 sidecar `.meta.yaml`；若只提供 raw + metadata，则由项目侧转换生成。
4. 首批标签来源不限于智能插座，也可以是子表、回路 CT、设备遥测或高可信控制日志。

## 10. 无光储数据与本项目 Pipeline 的关系

无光储场景比有光储场景更容易接入本项目 pipeline，但仍需要满足最小数据条件。

如果疆海提供的数据已经整理为以下训练 CSV 形式：

```text
time,aggregate,<appliance_name>,segment_id
```

并且满足时间对齐、功率单位统一为 W、有设备级高可信标签、`segment_id` 已正确切分，则可以直接进入本项目训练流程。

如果疆海提供的是云平台原始导出、设备原始 CSV、JSONL 或 API 数据，则通常不能不经转换直接进入训练。项目侧需要基于 `field_dictionary.csv`、`measurement_points.csv`、`sampling_policy.md` 和设备标签说明，先完成字段映射、单位转换、时间对齐、缺失/异常标记、`segment_id` 切分和训练 CSV 导出。

无光储数据进入本项目 pipeline 的最小条件如下：

1. 有全屋总负载或总表/CT 功率通道，可作为 `aggregate`。
2. 有至少一个目标设备的高可信设备级功率标签，可作为 `<appliance_name>`。
3. 总表测点覆盖该目标设备所在回路。
4. 总表和设备标签来自同一家庭，时间戳可解析并可对齐。
5. 功率单位可统一为 W，采样间隔可重采样到固定频率。
6. 缺失、重复、异常值、设备离线和长缺口能够被标记或修复。
7. 可以生成连续片段 `segment_id`，训练窗口不会跨越长缺口。

如果只提供家庭总负载而没有设备级标签，则不能用于监督式 NILM 训练，但可用于无监督探索、能耗画像、异常检测或后续人工标注候选片段。如果只提供 5 分钟或 15 分钟聚合数据，则不适合作为细粒度 appliance-level NILM 主训练数据，但可用于业务统计、能量平衡和储能策略分析。

## 11. 需要疆海确认的问题

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
