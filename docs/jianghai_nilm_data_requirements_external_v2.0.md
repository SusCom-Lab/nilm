# 广州疆海科技 NILM 项目数据交付需求说明

版本：v2.0  
适用项目：家庭负载 NILM 识别；覆盖无光伏储能家庭和有光伏/储能家庭。  
交付对象：广州疆海科技有限公司数据、云平台、App、硬件、算法及测试团队。

## 0. 首批最低交付摘要

首批不要求一次性交付全部字段。为尽快完成数据可用性确认和 NILM 联调，请优先提供以下内容：

1. 无光储家庭：家庭总负载或总表功率、智能电表/CT 数据、可作为电器标签的智能插座/子表/回路 CT/设备遥测数据、设备日志和 App 操作日志。
2. 有光伏或储能家庭：在无光储数据基础上，额外提供光伏功率、储能充放电功率、电池 SoC、电网输入输出功率、逆变器状态/告警/保护/限功率/并离网日志。
3. 必要说明文件：字段字典、采样频率说明、时间戳规则、单位说明、正负号说明、测点位置说明、设备与计量通道绑定关系、数据可用性说明。

对无光储家庭，只要同一时间段内能提供 `aggregate` 总负载数据、至少 1-3 类高可信设备级标签，以及字段/采样/测点/正负号说明，即可先进入项目侧数据转换和 NILM 联调。有光储家庭则必须先确认测点拓扑和功率正负号，否则不能直接把并网点功率当作家庭真实负载。

## 1. 交付目标

本项目需要从家庭总负载、智能电表/CT 等总表数据中识别设备级用电功率或运行状态。为保证后续模型训练、评估和问题追溯，请优先交付两类内容：

1. `raw/`：疆海侧原始导出数据，原则上不修改、不覆盖、不人工清洗。
2. `metadata/`：解释原始数据所必需的说明文件，包括字段、单位、采样频率、时间戳、正负号、测点位置、设备绑定和数据可用性。

`clean/`、`aligned/`、`train_ready/`、`quality/` 通常由项目侧基于 `raw/` 和 `metadata/` 生成；若疆海已有内部清洗版或训练版，可作为补充提供。

## 2. 交付优先级

| 级别 | 名称 | 数据范围 | 目标 |
|---|---|---|---|
| M0 | 字段样例包 | 每类场景 1 户，1-3 天，原始字段样例 + 字段字典 + 测点说明 | 确认字段、单位、时间戳、正负号、测点位置和导出格式。 |
| M1 | 小样本联调包 | 每类场景 3-5 户，7-14 天，raw + 必要 metadata + 若干设备标签 | 跑通读取、格式转换、时间对齐、质量检查和连续片段切分。 |
| M2 | 初版模型评估包 | 每类场景建议 20 户以上，60-90 天，稳定总表和高可信设备标签 | 训练和评估第一版 NILM 模型。 |
| M3 | 长期 benchmark 增强包 | 每类场景 50-100 户以上，180-365 天，覆盖季节、电价、策略和设备差异 | 做跨家庭、跨季节、跨场景泛化评估。 |

## 3. 场景与运行状态字段

正式交付时建议使用 `house_system_type` 和 `operation_state` 两个字段，不再把家庭硬件形态和运行状态混在一个字段中。

| 字段 | 含义 | 推荐枚举/示例 | 填写粒度 |
|---|---|---|---|
| `house_system_type` | 家庭硬件形态 | `no_pv_no_storage`、`pv_only`、`storage_only`、`pv_storage`、`unknown` | 家庭级 metadata 必填；时间序列中建议冗余带上。 |
| `operation_state` | 某时刻或某时间段的运行状态 | `grid_tied`、`off_grid`、`backup`、`islanding`、`zero_export`、`charging`、`discharging`、`curtailment`、`fault`、`unknown` | 时间序列/事件日志建议提供；无状态数据时填 `unknown`。 |

无光储家庭通常为 `house_system_type=no_pv_no_storage`，`operation_state=grid_tied` 或 `unknown`。有光储家庭必须额外明确并网、离网、备电、充放电、限功率、故障等状态，否则并网点功率不能直接当作家庭负载。

## 4. 疆海需交付的数据包

`H0001` 表示家庭匿名编号，不应使用真实姓名、手机号、详细地址。

以下为建议组织方式，不要求疆海内部系统完全按该目录存储；只要导出数据能按相同含义归类，并提供对应字段说明、测点说明和文件清单即可。

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
    pv_systems.csv                  # 光伏系统信息；无光伏可不提供
    storage_systems.csv             # 储能系统信息；无储能可不提供
    inverters.csv                   # 逆变器信息；无逆变器可不提供
    tariff_plans.csv                # 电价信息；如无相关策略可不提供
    firmware_versions.csv           # 固件版本：设备型号、固件版本、升级时间、影响字段
    appliance_taxonomy.yaml         # 电器命名规范：标准名、中文名、同义词、类别
    state_machine.yaml              # 设备/逆变器状态机；有光储或复杂设备建议提供
    inverter_state_transition_table.csv # 逆变器状态跃迁表；有逆变器家庭建议提供

  raw/                              # 疆海原始导出数据
    no_pv_storage/                  # 无光伏、无储能家庭
      H0001/
        aggregate/                  # 家庭总负载或总表功率数据，是 NILM 的主要输入来源
        meter_ct/                   # 智能电表/CT 数据，用于校验总表、相别、方向和测点位置
        smart_plug/                 # 智能插座或插排数据，可作为设备级高可信标签来源
        device_logs/                # 设备状态/故障/固件/离线重连日志
        app_logs/                   # 用户 App 操作日志
        labels/                     # 电器级标签、弱标签、用户反馈或人工标注

    pv_storage/                     # 有光伏/储能/逆变器家庭
      H1001/
        aggregate/                  # 家庭总负载或总表功率；需说明是否为真实负载还是并网点功率
        meter_ct/                   # 电表/CT/并网点数据；必须说明测点位置和方向
        smart_plug/                 # 插座级设备标签；只覆盖可插座计量设备
        pv/                         # 光伏发电数据；需说明 DC 侧、MPPT 侧还是 AC 侧
        storage/                    # 储能充放电、SoC、BMS 状态
        inverter/                   # 逆变器运行状态、告警、保护、限功率、并网/离网日志
        hems/                       # 能源管理系统计划、控制指令、预测输入
        app_logs/                   # 用户 App 操作日志
        labels/                     # 电器级标签、弱标签、用户反馈或人工标注
```

## 5. 通用字段要求

所有 CSV/Parquet 时间序列建议包含以下公共字段或等价伴随字段。若某些字段无法逐行提供，可以在文件名、文件级 metadata 或对应的 `metadata/*.csv` 中提供，但必须能唯一映射到每条数据。

优先级说明：

- **最低必需**：首批联调必须能提供或能从 metadata 唯一映射。
- **强烈推荐**：建议首批提供；如果暂时没有，需要说明原因和后续可补充方式。
- **可选增强**：用于更细的数据质量诊断，可在后续批次补充。

| 字段名 | 类型 | 单位/枚举 | 必填 | 说明 |
|---|---|---|---|---|
| `timestamp_utc` | datetime/string | ISO 8601, UTC | 最低必需 | 统一时间轴，例如 `2026-01-01T00:00:00.000Z`。 |
| `timestamp_device_utc` | datetime/string | UTC | 强烈推荐 | 设备侧采样时间。若与 `timestamp_utc` 相同，可说明后不重复提供。 |
| `timestamp_server_utc` | datetime/string | UTC | 强烈推荐 | 云端接收或入库时间，用于判断网络延迟、回填和乱序。 |
| `timestamp_ingest_utc` | datetime/string | UTC | 强烈推荐 | 本批数据导出或项目侧入库时间。 |
| `timestamp_local` | datetime/string | 本地时间 | 推荐 | 用户所在时区的本地时间；若提供 UTC 和时区，可由项目侧换算。 |
| `timezone` | string | IANA TZ | 最低必需 | 例如 `Asia/Shanghai`。 |
| `house_id` | string | `H0001` | 最低必需 | 匿名家庭编号。 |
| `house_system_type` | string | 见第 3 节 | 最低必需 | 家庭硬件形态；至少需在 `metadata/households.csv` 中提供。 |
| `operation_state` | string | 见第 3 节 | 强烈推荐 | 运行状态，可按时间变化；有光储家庭强烈建议提供。 |
| `source_system` | string | - | 强烈推荐 | App、云平台、设备固件、测试平台、API 等。 |
| `source_device_id` | string | 匿名 ID | 最低必需 | 采集设备、插座、逆变器、电池或 App 侧 ID；若不在每行提供，必须能从文件名或 metadata 唯一映射。 |
| `source_channel_id` | string | 匿名 ID | 最低必需 | CT 相线、插座通道、MPPT 通道、电池包通道等；若不在每行提供，必须能从文件名或 metadata 唯一映射。 |
| `measurement_point_id` | string | 测点 ID | 强烈推荐 | 对应 `metadata/measurement_points.csv`，用于说明这个功率到底测在总进线、负载侧、PV 侧、电池侧还是逆变器侧。 |
| `source_sequence_id` | string/int | - | 强烈推荐 | 源系统序列号或采集顺序，用于识别重复、乱序和回填。 |
| `ingest_order` | integer | - | 强烈推荐 | 导出文件中的原始顺序；若无 `source_sequence_id`，建议提供该字段。 |
| `clock_drift_ms` | number | ms | 可选增强 | 估计设备时钟漂移。 |
| `availability_status` | string | `available/missing/not_collected/not_supported/redacted` | 强烈推荐 | 字段或记录可用状态。 |
| `sampling_interval_s` | number | s | 最低必需 | 该记录所属文件的目标采样间隔；也可在 `sampling_policy.md` 中按文件说明。 |
| `schema_version` | string | - | 强烈推荐 | 字段 schema 版本。 |
| `ingest_batch_id` | string | 批次 ID | 最低必需 | 便于追溯本次交付批次。 |

raw 层保留源系统原貌，可存在重复、乱序、缺口和回填，但必须保留 `source_sequence_id` 或 `ingest_order`。后续清洗建模时再统一去重、排序和对齐。

## 6. 质量标签建议

建议不要只使用单一 `quality_flag`，而是使用主质量状态加可组合质量标签。

如果原始系统暂不支持逐点质量标签，可先在 `metadata/data_availability_matrix.csv` 和 `sampling_policy.md` 中说明缺失、离线、估算、插值、回填、裁剪等规则；逐点质量字段可作为后续增强交付。

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

## 7. 需要提供的数据清单与字段

### 7.1 家庭总负载数据

用途：NILM 主输入。无光储场景下，该数据通常可直接作为 `aggregate`。

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `timestamp_utc` | datetime | UTC | 1 s | 是 | 设备采样时间。 |
| `house_id` | string | - | 每行 | 是 | 家庭匿名编号。 |
| `aggregate_active_power_w` | number | W | 1 s | 是 | 家庭总有功功率。 |
| `aggregate_reactive_power_var` | number | var | 1 s | 推荐 | 总无功功率。 |
| `aggregate_apparent_power_va` | number | VA | 1 s | 推荐 | 视在功率。 |
| `voltage_v` | number | V | 1 s | 推荐 | 电压。 |
| `current_a` | number | A | 1 s | 推荐 | 电流。 |
| `power_factor` | number | 0-1 | 1 s | 推荐 | 功率因数。 |
| `frequency_hz` | number | Hz | 1 s | 可选 | 电网频率。 |
| `energy_import_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计购电量。 |
| `energy_export_kwh_total` | number | kWh | 10-60 s | 有光储必填 | 累计馈电量。 |
| `quality_flag_primary` | string | - | 每行 | 是 | 主质量标记。 |

有光储家庭中，总表位置可能测到“电网交换功率”而不是“家庭真实负载”。若存在负载侧直接测点，优先提供 `measured_load_power_w`；若只有并网点、PV、电池和逆变器数据，则需通过 `measurement_points.csv` 中的拓扑和 `formula_id` 推导 `load_estimated_power_w`。

### 7.2 智能电表/CT 数据

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `measurement_point_id` | string | - | 每行 | 是 | 测点 ID，需能在 `measurement_points.csv` 查到。 |
| `meter_id` | string | - | 每行 | 是 | 电表或 CT 设备匿名 ID。 |
| `meter_model` | string | - | 每文件/每行 | 是 | 电表或 CT 型号。 |
| `phase` | string | `total/L1/L2/L3` | 每行 | 是 | 三相或单相。 |
| `bus_or_node` | string | - | 每文件/每行 | 推荐 | 测点所在母线或节点。 |
| `upstream_component` | string | - | 每文件/每行 | 推荐 | 测点上游组件。 |
| `downstream_component` | string | - | 每文件/每行 | 推荐 | 测点下游组件。 |
| `measures_load_or_grid_or_pv_or_battery` | string | `load/grid/pv/battery/inverter/circuit/unknown` | 每文件/每行 | 是 | 测量对象。 |
| `ct_direction` | string | `import_positive/export_positive/unknown` | 每文件 | 是 | CT 安装方向和正负号。 |
| `ct_ratio` | number | A/A | 每文件 | 推荐 | CT 变比。 |
| `is_bidirectional` | boolean | - | 每文件/每行 | 推荐 | 是否双向功率测点。 |
| `installed_start_utc` | datetime | UTC | 每文件/事件 | 推荐 | 测点安装或生效开始时间。 |
| `installed_end_utc` | datetime | UTC | 每文件/事件 | 推荐 | 测点安装或生效结束时间。 |
| `active_power_w` | number | W | 1 s | 是 | 有功功率。 |
| `reactive_power_var` | number | var | 1 s | 推荐 | 无功功率。 |
| `voltage_v` | number | V | 1 s | 推荐 | 相电压。 |
| `current_a` | number | A | 1 s | 推荐 | 相电流。 |
| `power_factor` | number | - | 1 s | 推荐 | 功率因数。 |

### 7.3 高可信设备级标签数据

标签来源不限于智能插座，也可以是子表、回路 CT、设备遥测、BMS/逆变器日志、App 控制日志或人工确认。

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `timestamp_utc` | datetime | UTC | 1-10 s | 是 | 标签时间。 |
| `house_id` | string | - | 每行 | 是 | 家庭 ID。 |
| `appliance_id` | string | - | 每行 | 是 | 设备 ID。 |
| `appliance_name` | string | taxonomy | 每行 | 是 | 标准设备名。 |
| `appliance_power_w` | number | W | 1-10 s | 是 | 设备级真实功率。 |
| `appliance_state` | string | `on/off/standby/running/unknown` | 1-10 s/事件 | 推荐 | 设备状态。 |
| `label_source` | string | `smart_plug/submeter/circuit_ct/device_telemetry/app_log/manual_annotation` | 每行 | 是 | 标签来源。 |
| `label_confidence` | number | 0-1 | 每行 | 是 | 标签可信度。 |
| `metered_by_channel_id` | string | - | 每行/每文件 | 推荐 | 标签来源通道。 |

首批优先评估 3-5 类可稳定计量设备。标签来源不限于智能插座。

### 7.4 设备运行日志

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `timestamp_utc` | datetime | 是 | 事件发生时间。 |
| `device_id` | string | 是 | 匿名设备 ID。 |
| `device_type` | string | 是 | `meter/ct/smart_plug/inverter/battery/gateway/hems` 等。 |
| `event_type` | string | 是 | `power_on/power_off/relay_on/relay_off/mode_change/fault/firmware_update/communication_lost/reconnect` 等。 |
| `old_value` | string/number | 推荐 | 事件前状态。 |
| `new_value` | string/number | 推荐 | 事件后状态。 |
| `trigger_source` | string | 推荐 | `app/auto/hems/local_button/cloud/api/schedule/protection`。 |
| `error_code` | string | 有故障必填 | 故障码。 |
| `firmware_version` | string | 推荐 | 固件版本。 |

### 7.5 用户 App 操作日志

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

### 7.6 家庭或设备基础信息

`households.csv`：

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `house_id` | string | 是 | 匿名家庭编号。 |
| `house_system_type` | string | 是 | 家庭硬件形态。 |
| `region_or_climate_zone` | string | 推荐 | 区域或气候带，不要详细地址。 |
| `timezone` | string | 是 | IANA 时区。 |
| `grid_phase_type` | string | 是 | `single_phase/three_phase/split_phase/unknown`。 |
| `floor_area_band` | string | 推荐 | 面积区间。 |
| `occupant_count_band` | string | 推荐 | 人数区间。 |
| `has_pv` | boolean | 是 | 是否有光伏。 |
| `has_storage` | boolean | 是 | 是否有储能。 |
| `data_start_utc` | datetime | 是 | 数据起始时间。 |
| `data_end_utc` | datetime | 是 | 数据结束时间。 |

`appliances.csv`：

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `appliance_id` | string | 是 | 匿名设备 ID。 |
| `house_id` | string | 是 | 所属家庭。 |
| `appliance_name` | string | 是 | 标准英文小写下划线命名。 |
| `appliance_name_cn` | string | 推荐 | 中文名。 |
| `rated_power_w` | number | 推荐 | 额定功率。 |
| `label_source` | string | 推荐 | 标签来源。 |
| `label_confidence` | number | 推荐 | 0-1 标签可信度。 |
| `metered_by_channel_id` | string | 强标签必填 | 对应智能插座、子表、回路 CT 或遥测通道。 |
| `binding_start_utc` | datetime | 推荐 | 设备与计量通道绑定起始时间。 |
| `binding_end_utc` | datetime | 推荐 | 设备与计量通道绑定结束时间。 |

### 7.7 标签、弱标签、用户反馈

用户反馈默认只提供结构化类别，不提供原始自由文本。如果确有必要使用文本，应先由疆海侧完成脱敏、敏感词过滤和发布范围审批。

| 字段名 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `timestamp_utc` 或 `event_start_utc` | datetime | 是 | 标签或事件开始时间。 |
| `event_end_utc` | datetime | 推荐 | 事件结束时间。 |
| `house_id` | string | 是 | 家庭 ID。 |
| `appliance_id` | string | 推荐 | 设备 ID。 |
| `appliance_name` | string | 是 | 设备名。 |
| `event_type` | string | 推荐 | `turn_on/turn_off/cycle_start/cycle_end/mode_change`。 |
| `evidence_source` | string | 是 | `smart_plug/submeter/app/device_log/user_feedback/manual`。 |
| `confidence` | number | 是 | 0-1。 |
| `feedback_type` | string | 用户反馈必填 | `correct/wrong/missing/unknown`。 |
| `comment_category` | string | 可选 | 结构化反馈类别，不提供原始自由文本。 |

### 7.8 光伏发电数据

有光伏家庭提供。

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `pv_system_id` | string | - | 每行 | 是 | 光伏系统 ID。 |
| `mppt_id` | string | - | 每行 | 推荐 | MPPT 通道。 |
| `pv_active_power_w` | number | W | 1 s | 是 | 光伏输出功率。 |
| `pv_measurement_side` | string | `dc_string_side/mppt_side/inverter_ac_side/unknown` | 每行/每文件 | 是 | PV 测量侧。 |
| `pv_formula_id` | string | - | 每行/每文件 | 推荐 | 参与负载推导的公式 ID。 |
| `pv_voltage_v` | number | V | 1 s | 推荐 | PV 电压。 |
| `pv_current_a` | number | A | 1 s | 推荐 | PV 电流。 |
| `pv_energy_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计发电量。 |
| `curtailment_flag` | boolean | - | 1-10 s | 推荐 | 是否限功率/弃光。 |

### 7.9 储能、电池 SoC 与 BMS 数据

有储能家庭提供。

| 字段名 | 类型 | 单位/枚举 | 推荐频率 | 必填 | 说明 |
|---|---|---|---:|---|---|
| `storage_system_id` | string | - | 每行 | 是 | 储能系统 ID。 |
| `battery_pack_id` | string | - | 每行 | 推荐 | 电池包 ID。 |
| `battery_power_w` | number | W | 1 s | 是 | 电池功率；正负号需按 `measurement_points.csv` 和 `sign_convention.md` 解释。 |
| `battery_measurement_side` | string | `dc_battery_side/inverter_dc_bus/ac_output_side/unknown` | 每行/每文件 | 是 | 电池功率测量侧。 |
| `charge_power_w` | number | W | 1 s | 推荐 | 充电功率；需说明是否由 `battery_power_w` 拆分。 |
| `discharge_power_w` | number | W | 1 s | 推荐 | 放电功率；需说明是否由 `battery_power_w` 拆分。 |
| `soc_percent` | number | % | 1-10 s | 是 | 电池 SoC。 |
| `soh_percent` | number | % | 60 s | 推荐 | 电池健康度。 |
| `battery_mode` | string | `idle/charging/discharging/standby/protect/fault` | 1-10 s | 是 | 电池状态。 |
| `bms_alarm_code` | string | - | 事件 | 有告警必填 | BMS 告警码。 |
| `bms_protection_code` | string | - | 事件 | 有保护必填 | BMS 保护码。 |

### 7.10 电网输入输出功率数据

有光储家庭必填；无光储家庭推荐。

| 字段名 | 类型 | 单位 | 推荐频率 | 必填 | 说明 |
|---|---|---:|---:|---|---|
| `grid_power_w` | number | W | 1 s | 是 | 正负号需按测点说明解释。 |
| `grid_import_power_w` | number | W | 1 s | 推荐 | 买电功率，非负。 |
| `grid_export_power_w` | number | W | 1 s | 推荐 | 馈电功率，非负。 |
| `grid_import_energy_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计买电量。 |
| `grid_export_energy_kwh_total` | number | kWh | 10-60 s | 推荐 | 累计馈电量。 |
| `zero_export_target_w` | number | W | 1-10 s | 推荐 | 零馈网目标。 |
| `grid_connection_state` | string | `grid_tied/off_grid/islanding/backup/unknown` | 1-10 s/事件 | 是 | 并网/离网状态。 |

### 7.11 逆变器与 HEMS 数据

有逆变器、储能或 HEMS 控制的家庭提供。

| 字段名 | 类型 | 单位/枚举 | 推荐频率 | 必填 | 说明 |
|---|---|---|---:|---|---|
| `inverter_id` | string | - | 每行 | 是 | 逆变器 ID。 |
| `inverter_model` | string | - | 每文件/每日 | 是 | 逆变器型号。 |
| `inverter_mode` | string | `auto/expert/backup/off_grid/grid_tied/standby/fault` | 1 s | 是 | 工作模式。 |
| `ac_output_power_w` | number | W | 1 s | 是 | AC 侧输出功率。 |
| `dc_input_power_w` | number | W | 1 s | 推荐 | DC 输入功率。 |
| `power_limit_w` | number | W | 1-10 s | 推荐 | 当前限功率值。 |
| `limit_reason` | string | - | 事件/10 s | 推荐 | 限功率原因。 |
| `alarm_active` | boolean | - | 1-10 s | 是 | 是否有告警。 |
| `protection_active` | boolean | - | 1-10 s | 是 | 是否保护中。 |
| `control_command` | string/json | - | 事件 | 推荐 | HEMS 或 App 控制指令。 |

建议同时提供 `metadata/state_machine.yaml` 或 `metadata/inverter_state_transition_table.csv`，描述 `standby`、`grid_tied`、`off_grid`、`backup`、`fault`、`protection`、`power_limit` 等状态的合法跃迁。

## 8. 首批建议交付

1. M0 字段样例包：每类场景 1 户、1-3 天；优先提供原始字段样例、`field_dictionary.csv`、`data_availability_matrix.csv`、`measurement_points.csv`、`sampling_policy.md`、`sign_convention.md`。
2. M1 小样本联调包：每类场景 3-5 户、7-14 天；无光储提供 `aggregate`、`smart_meter/CT`、3-5 类可稳定计量设备标签、设备日志和 App 日志；有光储在此基础上增加 PV、储能功率、SoC、并网点功率、逆变器状态/告警/保护/限功率/并离网日志、HEMS 控制日志。
3. 首批标签来源不限于智能插座，也可以是子表、回路 CT、设备遥测或高可信控制日志。

## 9. 需要疆海确认的问题

1. Smart Meter/CT 在现有家庭中的默认安装位置：总进线、逆变器侧、负载侧，还是其他位置？
2. 现有云平台可导出的最小时间粒度是多少：1 s、5 s、10 s、1 min，还是只保留聚合值？
3. 设备端和云端是否同时保存设备采样时间与服务器入库时间？
4. 智能插座或其他标签来源是否能稳定绑定到具体电器？绑定变更是否有历史记录？
5. 有光储场景下，家庭真实负载是否已有内部计算字段？计算口径是什么？
6. 电池功率正负号、并网点功率正负号、CT 方向在不同固件版本中是否一致？
7. HEMS/Auto/Expert 等模式的策略日志和控制指令是否可导出？
8. 逆变器、BMS、网关、智能插座的告警码/保护码是否有码表？
9. 用户反馈数据是否可匿名导出，是否包含用户纠错或设备命名数据？
10. 是否可提供少量高频波形或事件片段，用于后续提升设备特征识别能力？
