# 卫星过境观测 API（Satellite Pass Observation API）

给定一个历元时刻的卫星 ECI 状态与地面观测站坐标，预测指定 UTC 时间区间内
卫星相对观测站的可见窗口（升/落时刻、峰值仰角及其时刻、方位角等）。
`POST /relay-plan` 进一步接收**多个**带唯一 ID 与各自最低仰角的测站，逐站
复用同一套两体传播与可见性证明，求搜索区间内**最长的连续接力时段**：
哪些测站各负责哪一段、在什么时刻切换。

- **语言/框架**：Python 3.11+、FastAPI、NumPy、Pydantic v2
- **动力学**：纯两体开普勒传播（WGS84 引力常数），牛顿迭代解开普勒方程
  （带回退二分法），机器精度守恒
- **坐标系统**：固定 WGS84 椭球与固定地球自转率
  （ω = 7.2921151467×10⁻⁵ rad/s），IAU 1982 GMST 模型做 ECI↔ECEF 旋转；
  观测站按 WGS84 大地坐标转换为 ECEF，再到站心地平坐标（东-北-天，ENU）
- **求交策略**：两级自适应扫描 + 二分精确求仰角穿越 + 黄金分割精化峰值，
  搜索区间按**半开区间 [start, end)** 处理；双端低值区间须经 Lipschitz
  变化率界证明不可见才计为已覆盖，否则递归细分寻找隐藏穿越，细分到
  1 µs 仍不可判定的擦边区间如实登记 `uncovered_intervals`
  （见下文“算法说明”）

---

## 1. 一键启动（Docker Compose）

```bash
docker compose up --build
```

服务监听 `http://localhost:8000`：

| 用途 | 地址 |
| --- | --- |
| 健康检查 | `GET http://localhost:8000/health` |
| 过境预测（单站） | `POST http://localhost:8000/passes` |
| **多站接力计划** | **`POST http://localhost:8000/relay-plan`** |
| 交互式文档（Swagger UI） | `http://localhost:8000/docs` |
| OpenAPI 模式 | `http://localhost:8000/openapi.json` |

停止服务：`docker compose down`。

### 不使用 Docker（本地运行）

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

运行测试：

```bash
pip install -r requirements.txt pytest httpx
pytest -q
```

---

## 2. 请求格式（POST /passes）

Content-Type: `application/json`。所有时间必须是**带时区的 UTC**
（ISO-8601，推荐以 `Z` 结尾）。

| 字段 | 类型 | 单位 / 范围 | 说明 |
| --- | --- | --- | --- |
| `epoch` | string | UTC 时间 | 卫星状态向量的历元 |
| `start` / `end` | string | UTC 时间 | 搜索区间，`end > start`，跨度 **≤ 7 天** |
| `min_elevation_deg` | number | 度，(-90, 90) | 最低仰角（含该仰角即为可见） |
| `satellite.position_km` | number[3] | **km**，有限数 | 历元 ECI 位置 [x, y, z] |
| `satellite.velocity_km_s` | number[3] | **km/s**，有限数 | 历元 ECI 速度 [vx, vy, vz] |
| `station.latitude_deg` | number | 度，[-90, 90] | WGS84 大地纬度 |
| `station.longitude_deg` | number | 度，[-180, 180] | WGS84 经度 |
| `station.altitude_km` | number | **km**，[-12, 100000] | WGS84 椭球高 |

### 422 校验规则

以下情况返回 **HTTP 422**（FastAPI/Pydantic 标准错误体）：

- 时间顺序非法（`end <= start`）或跨度超过 7 天；时间不带时区
- 经纬高、最低仰角超出物理范围；向量长度不是 3 或含 NaN/Infinity
- 轨道退化：位置/速度为零、r 与 v 共线（角动量为零）
- 轨道非束缚：抛物线/双曲线（比机械能 ≥ 0）或离心率 ≥ 1
- 近地点位于地表以下（两体传播无物理意义）
- 未知字段（请求体禁止额外字段）

> 单位提示：位置必须用 **km**、速度必须用 **km/s**。若误把位置按米
> （放大 1000 倍）提交，近地点将落到地球内部，服务会以 422
> `orbit_impact_earth` 拒绝；速度按 m/s 提交则会因双曲线轨道被拒绝。

---

## 3. 实际请求示例

以下是一次真实请求（420 km 圆轨道、倾角 51.6° 的类 ISS 卫星，
北京附近观测站，最低仰角 10°，搜索 6 小时；完整文件见
[`examples/request.json`](examples/request.json)）：

```bash
curl -X POST http://localhost:8000/passes \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

请求体（即 `examples/request.json`）：

```json
{
  "epoch": "2025-06-01T00:00:00Z",
  "start": "2025-06-01T00:00:00Z",
  "end": "2025-06-01T06:00:00.000Z",
  "min_elevation_deg": 10.0,
  "satellite": {
    "position_km": [4270.307144917519, -1323.5276198332833, 5121.271150639637],
    "velocity_km_s": [3.950957000514434, 6.347260232247907, -1.6540865405752112]
  },
  "station": {
    "latitude_deg": 39.9,
    "longitude_deg": 116.4,
    "altitude_km": 0.05
  }
}
```

### 实际响应（HTTP 200）

与 [`examples/response.json`](examples/response.json) 内容一致：

```json
{
  "status": "complete",
  "count": 1,
  "windows": [
    {
      "start": "2025-06-01T00:01:40.294555Z",
      "end": "2025-06-01T00:08:29.127935Z",
      "peak_time": "2025-06-01T00:05:05.712339Z",
      "peak_elevation_deg": 85.67588427634527,
      "peak_azimuth_deg": 218.30123915031336,
      "rise_azimuth_deg": 306.5090307458435,
      "set_azimuth_deg": 130.01647942552455,
      "max_range_km": 1521.862803277411,
      "duration_s": 408.8333794882783,
      "start_open": false,
      "end_open": false,
      "truncated": false
    }
  ],
  "uncovered_intervals": [],
  "search_start": "2025-06-01T00:00:00Z",
  "search_end": "2025-06-01T06:00:00Z",
  "min_elevation_deg": 10.0,
  "diagnostics": {
    "orbit": {
      "semi_major_axis_km": 6798.136999999997,
      "eccentricity": 3.6354953649115305e-16,
      "period_s": 5578.222707272108,
      "perigee_radius_km": 6798.136999999995,
      "apogee_radius_km": 6798.137,
      "perigee_altitude_km": 419.99999999999545,
      "apogee_altitude_km": 420.0
    },
    "coarse_samples": 297,
    "dense_samples": 1898,
    "search_span_s": 21600.0,
    "uncovered_seconds": 0,
    "coverage_fraction": 1.0,
    "notes": "Elevation crossings refined by bisection; peaks by golden-section search. Half-open search interval [start, end). Below-threshold intervals count as covered only when a Lipschitz rate bound proves them invisible; unprovable brackets are bisected down to the 1 us reporting resolution to expose hidden crossings, and whatever stays unresolvable (grazing contacts) is reported in uncovered_intervals instead of being silently skipped."
  }
}
```

字段含义：

- `windows[]` 按 **`start` 升序**排列；每个窗口含起止时刻、峰值时刻、
  峰值仰角（度）、峰值/升/落方位角（度，正北顺时针）、最大斜距（km）、
  持续时间（秒）。
- `start_open` / `end_open`：窗口在搜索区间起点已可见 / 到区间终点仍可见
  （半开区间被边界截断）。
- `truncated`：窗口因数值失败被提前截断（此时 `status` 为 `partial`）。
- `uncovered_intervals`：未能覆盖的时间区间及原因——传播/求根数值失败，
  或**无法在 1 µs 报告精度内判定**的擦边接触（峰值与门限之差小于数值
  分辨率，既不能证明不可见也无法解析出穿越）。不为空时 `status` 为
  `partial`（有结果但不完整）或 `failed`（没有任何可信窗口），
  `coverage_fraction` 相应小于 1。
- `diagnostics`：轨道根数、粗/细扫描采样数、未覆盖秒数与覆盖率。

### 422 响应示例（搜索区间 8 天 > 7 天上限）

将 `examples/request.json` 的 `end` 改为 `"2025-06-09T00:00:00Z"` 后
（完整响应见 [`examples/error_422.json`](examples/error_422.json)）：

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body"],
      "msg": "Value error, search span (end - start) must not exceed 7 days",
      "input": { "epoch": "2025-06-01T00:00:00Z", "...": "..." },
      "ctx": { "error": {} }
    }
  ]
}
```

轨道退化/非束缚错误示例：

```json
{
  "detail": [
    {
      "type": "orbit_degenerate_or_invalid",
      "loc": ["body", "satellite"],
      "msg": "orbit is not bound (parabolic/hyperbolic): only elliptic orbits are supported"
    }
  ]
}
```

---

## 3bis. 多站接力计划（POST /relay-plan）

在 `/passes` 的同一颗卫星、同一个历元与搜索区间上，给出一个**非空**测站
列表（每站有唯一 `id`、经纬高与各自的 `min_elevation_deg`）。服务逐站复用
现有的两体传播与 `find_passes`，只使用其**已证**结果：

- 每个返回窗口都是**已证可见**的半开区间 `[start, end)`（`start_open` /
  `end_open` 表示被搜索边界截断，`truncated` 表示被数值失败截断）；
- `uncovered_intervals` 是该站既不能证明可见、也不能证明不可见的时段。

随后在“已证可见”窗口构成的图上求搜索期内**最长的连续接力时段**：

- 相邻两段由**不同**测站负责；只有当两站窗口有**严格正时长的重叠**时
  才允许切换（仅端点相接不算重叠）。切换时刻取进入窗口的起点，保证
  切换瞬间两站都在各自的已证可见区间内，覆盖无空洞。
- 同一测站首尾相接的多个窗口无需切换，自动并为一段连续可见跨度。
- 等长方案：先取**切换次数更少**者；再相同则按**开始时刻更早**、
  最后按**站 ID 序列字典序**裁决，结果完全确定。
- 某站在某时段未覆盖、但**别站可证可见**时，接力照常进行。
- 只有当**没有任何站可证可见**、且至少有一站把该时段报为未覆盖时，
  该时段才进入 `unknown_intervals`——此时**不会**报“确定不可见”，
  也不会把它算作接力覆盖。

### 请求格式

Content-Type: `application/json`，所有时间为带时区 UTC。字段
`epoch` / `start` / `end` / `satellite` 的含义与校验同 `/passes`；
区别在于用 `stations` 数组替代单个 `station`，且最低仰角**逐站指定**。

| 字段 | 类型 | 单位 / 范围 | 说明 |
| --- | --- | --- | --- |
| `stations` | array | 长度 ≥ 1 | 候选测站，`id` 必须唯一且非空白 |
| `stations[].id` | string | 非空 | 测站唯一标识（计划/切换中引用） |
| `stations[].latitude_deg` | number | 度，[-90, 90] | WGS84 大地纬度 |
| `stations[].longitude_deg` | number | 度，[-180, 180] | WGS84 经度 |
| `stations[].altitude_km` | number | km，[-12, 100000] | WGS84 椭球高 |
| `stations[].min_elevation_deg` | number | 度，(-90, 90) | 该站最低仰角 |

### 实际请求示例（5 站 4 次切换的接力链）

类 ISS 轨道、25° 最低仰角、30 分钟搜索区间；5 个沿轨迹由西向东排布的
测站（蒙古高原 ULN、北京 BJS、黄海 YEL、琉球 RYU、菲律宾以东 PHL），
其已证可见窗口首尾正重叠，强制形成一条 5 站接力链。完整文件见
[`examples/relay_request.json`](examples/relay_request.json)：

```bash
curl -X POST http://localhost:8000/relay-plan \
  -H 'Content-Type: application/json' \
  -d @examples/relay_request.json
```

请求体（节选）：

```json
{
  "epoch": "2025-06-01T00:00:00Z",
  "start": "2025-06-01T00:00:00Z",
  "end": "2025-06-01T00:30:00Z",
  "satellite": {
    "position_km": [4270.307144917519, -1323.5276198332833, 5121.271150639637],
    "velocity_km_s": [3.950957000514434, 6.347260232247907, -1.6540865405752112]
  },
  "stations": [
    {"id": "ULN",  "latitude_deg": 45.0, "longitude_deg": 108.0, "altitude_km": 0.05, "min_elevation_deg": 25.0},
    {"id": "BJS",  "latitude_deg": 39.9, "longitude_deg": 116.4, "altitude_km": 0.05, "min_elevation_deg": 25.0},
    {"id": "YEL",  "latitude_deg": 35.0, "longitude_deg": 128.0, "altitude_km": 0.05, "min_elevation_deg": 25.0},
    {"id": "RYU",  "latitude_deg": 30.0, "longitude_deg": 132.0, "altitude_km": 0.05, "min_elevation_deg": 25.0},
    {"id": "PHL",  "latitude_deg": 20.0, "longitude_deg": 135.0, "altitude_km": 0.05, "min_elevation_deg": 25.0}
  ]
}
```

### 实际响应（HTTP 200）

与 [`examples/relay_response.json`](examples/relay_response.json) 一致：

```json
{
  "status": "complete",
  "plan": {
    "start": "2025-06-01T00:01:05.760119Z",
    "end": "2025-06-01T00:13:36.099663Z",
    "duration_s": 750.339544,
    "switch_count": 4,
    "segments": [
      {"station_id": "ULN", "start": "2025-06-01T00:01:05.760119Z", "end": "2025-06-01T00:03:14.455707Z", "duration_s": 128.695588},
      {"station_id": "BJS", "start": "2025-06-01T00:03:14.455707Z", "end": "2025-06-01T00:06:07.934530Z", "duration_s": 173.478823},
      {"station_id": "YEL", "start": "2025-06-01T00:06:07.934530Z", "end": "2025-06-01T00:07:38.505114Z", "duration_s": 90.570584},
      {"station_id": "RYU", "start": "2025-06-01T00:07:38.505114Z", "end": "2025-06-01T00:10:12.483339Z", "duration_s": 153.978225},
      {"station_id": "PHL", "start": "2025-06-01T00:10:12.483339Z", "end": "2025-06-01T00:13:36.099663Z", "duration_s": 203.616324}
    ],
    "switches": [
      {"time": "2025-06-01T00:03:14.455707Z", "from_station_id": "ULN", "to_station_id": "BJS"},
      {"time": "2025-06-01T00:06:07.934530Z", "from_station_id": "BJS", "to_station_id": "YEL"},
      {"time": "2025-06-01T00:07:38.505114Z", "from_station_id": "YEL", "to_station_id": "RYU"},
      {"time": "2025-06-01T00:10:12.483339Z", "from_station_id": "RYU", "to_station_id": "PHL"}
    ]
  },
  "stations": [
    {"id": "ULN", "status": "complete", "min_elevation_deg": 25.0,
     "proven_windows": [{"start": "2025-06-01T00:01:05.760119Z", "end": "2025-06-01T00:04:46.735022Z",
                         "start_open": false, "end_open": false, "truncated": false}],
     "uncovered_intervals": []}
  ],
  "unknown_intervals": [],
  "search_start": "2025-06-01T00:00:00Z",
  "search_end": "2025-06-01T00:30:00Z",
  "diagnostics": {
    "search_span_s": 1800.0,
    "proven_visible_seconds": 750.339544,
    "proven_visible_fraction": 0.41685530222222217,
    "unknown_seconds": 0.0,
    "relay_covered_seconds": 750.339544,
    "relay_coverage_fraction": 0.4168553022222222,
    "station_count": 5
  }
}
```

> 上例 `stations` 实际含 5 个元素，此处为节省篇幅只完整展示了 ULN 一项；
> 其余 4 项结构相同。每个切换时刻都落在前后两站窗口的**正重叠内**：
> 例如 `00:03:14.455707` 既是 BJS 窗口的起点，又严格位于 ULN 窗口
> （止于 `00:04:46.735022`）之中，故切换瞬间两站都已证可见。

如何读取接力结果：

- **各站负责区间**：`plan.segments[]`，按时间排列、首尾相接，恰好铺满
  `plan.start`～`plan.end`；`station_id` 即该段的负责站。
- **切换时刻**：`plan.switches[]`，每个含 `time` 与 `from/to_station_id`；
  切换时刻的个数就是 `plan.switch_count`，且 `segments` 的边界与
  `switches[].time` 一一对应。
- **最长连续接力**：`plan.start`/`end`/`duration_s` 给出搜索区间内最长的
  一段无空洞接力（等长时按“切换更少 → 开始更早 → 站 ID 字典序”裁决）。
- `stations[]`：逐站回传 `find_passes` 的 `status`、已证可见半开窗口
  （含 `start_open`/`end_open`/`truncated`）与其 `uncovered_intervals`。
- `unknown_intervals[]`：无人可证可见、但存在未覆盖的**未知时段**（含
  原因）；它**不**计入接力，也**不**被宣称确定不可见。

### status 语义与未知时段

顶层 `status`：

- `complete`：存在接力计划且没有未知时段；或**没有任何站能看见**且整个
  搜索区间都被证明不可见（此时 `plan` 为 `null`）。
- `partial`：存在接力计划，但搜索区间内另有 `unknown_intervals`。
- `failed`：没有任何已证可见窗口，且至少有一段未知（无法判定）。

**全部站都已证不可见**时返回**空计划**（不是错误）：`plan` 为 `null`、
`unknown_intervals` 为空、`status` 为 `complete`。

下例中 ULN（25°）已证可见到 `00:04:46`，而 GRAZE 站把最低仰角设在其峰值
之上 1e-7°，峰值附近约 4 ms 的擦边区间无法在 1 µs 精度内判定；该时段
ULN 已落、无人可证可见，故如实列为未知，`status=partial`
（请求/响应见 [`examples/relay_unknown_request.json`](examples/relay_unknown_request.json)
与 [`examples/relay_unknown_response.json`](examples/relay_unknown_response.json)）：

```json
{
  "status": "partial",
  "plan": {
    "start": "2025-06-01T00:01:05.760119Z",
    "end": "2025-06-01T00:04:46.735022Z",
    "duration_s": 220.974903,
    "switch_count": 0,
    "segments": [{"station_id": "ULN", "start": "2025-06-01T00:01:05.760119Z",
                  "end": "2025-06-01T00:04:46.735022Z", "duration_s": 220.974903}],
    "switches": []
  },
  "unknown_intervals": [
    {"start": "2025-06-01T00:05:05.709124Z", "end": "2025-06-01T00:05:05.713400Z",
     "reason": "station 'GRAZE': possible short grazing pass: below-threshold interval not provably invisible and crossing not resolvable at 1 us reporting resolution"}
  ]
}
```

> 反过来，若某站在某时段“未覆盖”、但**另一站在同一时段可证可见**，该时段
> 不进入 `unknown_intervals`，接力照常延续。

### `/relay-plan` 的 422 校验规则

沿用 `/passes` 的全部时间/坐标/轨道校验，此外：

- `stations` 为空 → 422；
- 任意两个 `stations[].id` 相同（或为纯空白）→ 422，例如：

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body"],
      "msg": "Value error, station ids must be unique; duplicate id(s): 'ULN'",
      "input": { "epoch": "2025-06-01T00:00:00Z", "...": "..." },
      "ctx": { "error": {} }
    }
  ]
}
```

- 轨道退化/非束缚、近地点穿地（`orbit_degenerate_or_invalid` /
  `orbit_impact_earth`）→ 422；
- 搜索区间相对 `epoch` 超出 ±10 年、跨度超过 7 天、时间顺序非法或不带
  时区 → 422；逐站经纬高/仰角越界、未知字段 → 422。

---

## 4. 算法说明

### 4.1 两体传播

由历元状态计算比机械能 ε = v²/2 − μ/r 与离心率矢量，要求 ε < 0、
e < 1。任意时刻 t = t₀ + Δt 用开普勒方程 M = E − e·sin E
（牛顿迭代，容差 1e-10，失败时二分回退），再在由 h、e 矢量构造的
近焦点坐标系中还原 r、v。圆轨道（e≈0）使用 r 方向作为面内基准，
无近点奇点。自测：一个周期后状态误差为 0，3 天内能量/角动量
相对漂移 < 1e-15。

### 4.2 坐标与时角

- GMST 使用 IAU 1982 模型（J2000 锚点 18h41m50.543s）。距 J2000 的
  日数直接由秒差计算——若先合成完整儒略日（≈2.46×10⁶）再相减，双精度
  会把时刻量化到约 40 µs，经测站杠杆臂（R站/斜距，近天顶时可达 12）
  放大为约 1e-6° 的仰角阶梯噪声，足以淹没或伪造擦边穿越；
- ECI→ECEF：`r_ECEF = Rz(−GMST)·r_ECI`，速度另减 ω×r 项；
- ENU 基由测站纬经度固定，仰角 `asin(ρ_up/|ρ|)`、
  方位角 `atan2(ρ_E, ρ_N)`；并给出仰角/方位角解析变化率供自适应用。

### 4.3 两级自适应扫描（不漏窗口）

1. **粗扫（几何门限）**：只算卫星—测站的**地心角** α。可见要求
   cos α ≥ R站/r星，因此用远地点构造最松门限 ψ_flag（地平角 + 8° 缓冲）。
   由近地点速度给出严格的地心角变化率上界
   Kα = v_p/r_p + ω_E，粗扫步长固定为 5°/Kα（LEO 约 40–75 s），
   **任何可能可见的几何都不可能被跨过**。标记点 ±一个粗步长合并为候选段。
2. **细扫（仰角门限）**：在候选段内按仰角距阈值的余量和全局 Lipschitz 界
   K1 = (v_p + ω_E·r_a)/(r_p − R站) 自适应取步（下限 0.02 s、上限 20 s）。
   **双端低于阈值的区间只有满足 K1·Δt ≤ 0.5·min(余量) 才算已覆盖**
   （变化率界证明曲线不可能触及阈值）；步长下限本身**不构成**覆盖依据。
3. **不可证明区间的递归细分**：不满足证明的双端低值区间被递归二分
   ——中点若高于阈值即暴露隐藏穿越（交回二分精确化），子区间若被证明
   则覆盖，直到 1 µs（UTC 时间戳的报告精度）。在此精度下仍不可证明的
   残段（峰值与门限之差小于数值分辨率的擦边接触）**登记进
   `uncovered_intervals` 并注明原因**，绝不静默跳过；细分求值设有全局
   预算（20 万次，正常 7 天 LEO 任务约用 3 万次），预算耗尽时剩余未证明
   区间同样如实登记。
4. **精确求交**：符号变化的相邻采样对做 45 次二分（时间精度 1e-8 s，
   端点仰角误差 < 1e-7°）。
5. **峰值**：以最佳细扫采样及其相邻采样为支架做 32 次黄金分割，
   且结果永不低于实际观测到的最大采样值。

保证的边界：只有峰值与阈值之差小于数值分辨率（仰角噪声 <~1e-7°，
见 4.2）、持续时间短到无法在 1 µs 精度内表示的擦边接触才**可能**无法
判定——此时区间与原因会出现在 `uncovered_intervals` 中，`status` 与
`coverage_fraction` 如实降级，而不会以步长下限为由宣称 `complete`；
其余窗口不漏。与独立的 0.25 s 均匀网格真值对比（1 天类 ISS 轨道）：
窗口数完全一致，升/落时刻误差 < 0.05 s，峰值仰角误差 < 0.001°。

### 4.4 半开区间与失败报告

- 搜索域为 **[start, end)**：恰在 `end` 发生的降落穿越按区间内处理
  （窗口闭合于该时刻）；恰在 `start` 已可见则标记 `start_open=true`，
  到 `end` 仍可见标记 `end_open=true`。穿越时刻按 UTC 微秒精度报告；
  距 `start` 不足 1 µs 的上升穿越与恰在边界上不可区分，按
  `start_open=true` 处理。
- 零时长切触（峰值恰好等于阈值）不作为窗口返回。
- 任何传播/求值/二分失败：受影响时段进入 `uncovered_intervals`
  （含起止时刻与原因），打开中的窗口标记 `truncated`，
  `status` 降级为 `partial`/`failed`，绝不静默丢弃。
- 双端低值且无法用变化率界证明不可见、细分到 1 µs 仍无法判定的
  擦边区间同样进入 `uncovered_intervals`（相邻残段合并为一条），
  `coverage_fraction = 1 − 未覆盖秒数/搜索跨度` 随之如实下降。

### 4.5 性能

典型开销（单机纯 Python）：7 天类 ISS 任务约 8300 次粗扫 + 6.7 万次
细扫求值，耗时约 7 秒；6 小时任务约 0.1–0.5 秒。高轨（GEO、闪电轨道）
更省或相当。病态擦边场景（峰值距阈值 <1e-6°）会触发递归细分，
单次请求额外求值以 20 万次为上限。`/relay-plan` 对每站各跑一次
`find_passes`，开销约为单站的站数倍（接力图本身仅在窗口数上做
O(N²) 的小规模松弛，可忽略）。

### 4.6 多站接力规划（relay.py）

输入只有 `find_passes` 的**已证**产物：每站的已证可见半开窗口与其
`uncovered_intervals`、站级 `status`。规划分三步：

1. **并窗**：同一站首尾相接（共享端点）或重叠的窗口无需切换，先合并为
   一段连续“已证可见跨度”；跨站仅端点相接**不**并（半开区间在该点只有
   一侧可见，构不成正重叠）。
2. **连通分量**：两段跨度当且仅当属于**不同站**且有**严格正时长重叠**时
   相连（一段跨度可同时桥接多个已有分量，触发合并）。每个连通分量即一段
   “存在某种无空洞接力”的时间范围。
3. **最少切换链（分量内 DAG）**：边 u→v 存在当且仅当站不同、v 起点不早于
   u、v 终点严格更晚、两窗正重叠。终点沿边严格增大，故按终点排序做一次
   拓扑松弛即可求最少切换数；标号为 `(切换数, 站 ID 序列)`，后者作为
   字典序裁决，保证同一切换数下结果确定。切换时刻取进入窗口的起点——
   它位于两窗正重叠区间内，切换瞬间双方都已证可见。

各分量取其最优链后，再在分量间按 **时长（1 µs 分辨率）→ 切换数更少 →
开始时刻更早 → 站 ID 序列字典序**选出搜索期内最长的连续接力时段。
`segments` 由相邻窗口起点切分，首尾恰好铺满该时段且互不重叠。

**未知时段**：把各站 `uncovered_intervals` 合并后，扣除“至少有一站
已证可见”的部分；剩下的才是“无人可证可见且有站未覆盖”的时段，进入
`unknown_intervals` 并合并原因。它既不计入接力，也不会被报成确定不可见
（确定不可见的时段什么都不报，正如 `/passes` 中已证无窗口即为
`complete`）。

**正确性边界**：接力的每一段都是某次 `find_passes` 的已证可见窗口，
因此整条链上每个时刻都有至少一站可证可见；切换只发生在正重叠内，
半开语义下覆盖连续无空洞。规划器不引入任何新的数值近似——所有可见性
结论都继承自第 4.3 节带 Lipschitz 证明的扫描。

---

## 5. 项目结构

```
app/
  main.py        # FastAPI 路由、422 映射、/passes 与 /relay-plan 响应组装
  schemas.py     # Pydantic 请求/响应模型与范围校验（含逐站列表/唯一 ID）
  mechanics.py   # WGS84 常数、两体开普勒传播
  geometry.py    # 大地坐标、GMST、ECI/ECEF、ENU、仰角/方位角
  passes.py      # 两级自适应扫描、二分穿越、黄金分割峰值
  relay.py       # 多站接力：已证窗口连通分量、最少切换 DAG、未知时段
  time_utils.py  # UTC/儒略日/GMST
examples/        # /passes 与 /relay-plan 的真实请求、成功响应与 422 响应
tests/           # pytest：/passes、relay 规划器单元测试、/relay-plan HTTP 测试
Dockerfile, docker-compose.yml, requirements.txt
```
