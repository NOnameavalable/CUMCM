# 2026 CUMCM B题 本地高保真 Mock 模拟器使用与原理指南 (`mock_simulator.py`)

## 1. 概述与设计定位

`mock_simulator.py` 是专为 **2026 年全国大学生数学建模竞赛 B 题《无线电干扰源的快速自动定位与清除》** 开发的本地高保真模拟器服务端与可视化战术态势控制台。

### 核心特性
1. **100% 协议与物理模型保真**：完全对齐国赛官方 HTTP 协议格式，严格遵循题目附件1、附件2对于全向干扰源、定向干扰源的物理特性建模。
2. **纯 Python 3 标准库实现**：无第三方库强依赖（仅使用 `http.server`, `json`, `math`, `random`, `subprocess`, `argparse` 等），开箱即用。
3. **严格虚拟时间计费状态机**：移动速度、频段切换、测向动作、激光与光学定位清除均按照题目规则严格计费推进虚拟时间（`virtual_time_s`）。
4. **原生幂等性支持**：完整实现请求 ID 缓存（`request_id`），网络抖动或客户端重发请求不会导致虚拟时间重复累加或状态紊乱。
5. **实时 WebUI 战术态势感知平台**：内置现代化 HTML5 Canvas 单页控制台，浏览器打开 `http://127.0.0.1:2026` 即可直观查看雷达网格、机器狗轨迹、干扰源分布、定向源 180° 发射扇面、±1.0° 误差测向锥及实时战报。
6. **防崩溃持久化日志（Crash-Resistant）**：每次动作原子化落盘（保存至 `./logs/` 目录），即便外部算法崩溃中断，已生成的对战数据与轨迹依然完整保存，支持通过 WebUI 随时逐帧复盘。
7. **一键外部算法进程调度**：WebUI 界面提供算法状态监控与“启动算法 / 停止算法”一键管理，自动检测并关联当前 Python 环境执行 `P3.py` 或 `p4.py`。
8. **内置自动化测试套件**：内置 `--test` 参数，可一键校验几何函数、物理覆盖判定、HTTP 四接口、状态机时间推进与幂等逻辑。

---

## 2. 快速上手

### 2.1 依赖环境
- **Python 版本**：Python 3.8 及以上（无需额外 `pip install` 任何第三方包）。
- **浏览器**：Chrome / Edge / Safari / Firefox 等现代浏览器。

### 2.2 启动模拟器

#### 方式一：常规交互模式（默认 P4 模式）
在终端中进入项目工作区，执行：
```bash
python3 mock_simulator.py
```
终端将输出：
```text
======================================================================
  2026 CUMCM B题 Mock 模拟器服务端 + 可视化 WebUI 已启动
  可视化控制台: http://127.0.0.1:2026/ (浏览器直接打开查看)
  运行模式: P4 (全向 + 定向混合干扰源)
  算法执行 Python: /path/to/python
  持久化日志: 已启用 (保存在 /.../CUMCM/logs)
  支持接口: POST /enter, /measure, /clear, /exit  |  GET /api/logs, /api/live
  按 Ctrl+C 可终止服务
======================================================================
```
此时在浏览器中打开 **[http://127.0.0.1:2026/](http://127.0.0.1:2026/)** 即可看到战术态势控制台。

#### 方式二：打印详细真值输出（推荐用于算法调试）
添加 `--verbose` 参数，模拟器将在终端实时打印机器狗坐标、当前真值（目标与狗的真实距离、真实方位角、观测方位角、清除判定等）：
```bash
python3 mock_simulator.py --verbose
```

#### 方式三：固定随机种子（用于实验算法对比复现）
```bash
python3 mock_simulator.py --mode p4 --seed 42 --verbose
```

#### 方式四：自动化自测验证
```bash
python3 mock_simulator.py --test
```
执行几何、辐射判定、状态机流程、日志持久化等多项断言，确保模拟器自身逻辑完全正确。

---

## 3. CLI 命令行参数速查

| 参数 | 类型 | 默认值 | 作用说明 |
| :--- | :---: | :---: | :--- |
| `--host` | `str` | `127.0.0.1` | HTTP 服务监听的主机 IP 地址 |
| `--port` | `int` | `2026` | HTTP 服务监听的端口号 |
| `--mode` | `choice` | `p4` | 模式选项：`p3`（仅全向源）或 `p4`（全向+定向混合源） |
| `--seed` | `int` | `None` | 伪随机数种子，用于生成可复现的算例 |
| `--targets` | `int` | `None` | 干扰源数量（缺省时在 $[10, 16]$ 随机选取） |
| `--arena-radius` | `float` | `1800.0` | 场地半径（单位：米） |
| `--dog-speed` | `float` | `5.0` | 机器狗移动速度（单位：m/s） |
| `--clear-radius` | `float` | `20.0` | 激光清除最大有效判定距离（单位：米） |
| `--near-radius` | `float` | `5.0` | 测向近距饱和判定距离（单位：米） |
| `--python-bin` | `str` | `None` | 指定用于执行算法脚本的 Python 解释器绝对路径（缺省自动检索） |
| `--headless` | `flag` | `False` | 无头后台静默运行模式 |
| `--verbose` | `flag` | `False` | 在终端打印包含 Ground-Truth（真值）的单步对比战报 |
| `--verbose-http` | `flag` | `False` | 打印底层 HTTP 原始访问日志（GET/POST 路径） |
| `--test` | `flag` | `False` | 运行内置自动化验证套件并退出 |

---

## 4. 模拟器工作原理与物理模型

### 4.1 场地与干扰源生成 (`Arena._init_targets`)

```
                 +Y (正北 90°)
                     |
                     |       * 目标 T (x_t, y_t)
                     |      /
                     |     /
                     |    /
                     |   / 示向角 alpha
  -X ----------------+---------------- +X (正东 0°)
    (正西 180°)      | 原点 O (0, 0)
                     |
                     |
                 -Y (正南 270°)
```

1. **场地空间**：以坐标原点 $(0, 0)$ 为中心、半径 $R = 1800\,\text{m}$ 的圆形区域。
2. **面积均匀采样**：
   $$r = R \cdot \sqrt{u},\quad \theta = 2\pi v,\quad u, v \sim U(0, 1)$$
   $$x = r \cos \theta,\quad y = r \sin \theta$$
   数学上严格保障场地内各微元面积内目标出现概率均等，避免向中心聚拢。
3. **干扰源数量与信道**：
   - 目标总数 $K \in [10, 16]$（整数）。
   - 信道集合：从 $\{1, 2, \dots, 20\}$ 中无重复随机抽取 $K$ 个信道。
   - 接收半径：每个目标 $i$ 的有效接收半径 $r_{\text{recv}, i} \sim U[1000, 1500]\,\text{m}$。
4. **模式特性**：
   - **P3 模式**：所有 $K$ 个目标均为**全向干扰源**。
   - **P4 模式**：混合模式。随机选取 $k_{\text{dir}} \in [\max(1, \lfloor K/3 \rfloor), \max(2, \lfloor 2K/3 \rfloor)]$ 个信道作为**定向干扰源**，其余为全向源。每个定向源分配一个发射主朝向 $\theta_{\text{dir}} \sim U[0^\circ, 360^\circ)$。

---

### 4.2 信号辐射与空间覆盖判定 (`Target.is_in_coverage`)

设机器狗当前坐标为 $(x_d, y_d)$，信道 $i$ 的目标坐标为 $(x_t, y_t)$，欧氏距离为：
$$d = \sqrt{(x_d - x_t)^2 + (y_d - y_t)^2}$$

1. **径向距离条件**：若 $d > r_{\text{recv}, i}$，机器狗处在信号极限范围外，无法接收信号。
2. **辐射角度条件（针对定向源）**：
   - 定向源具有前向 $180^\circ$ 辐射扇面，后半球存在盲区。
   - 计算从目标指向机器狗的方向角 $\phi$：
     $$\phi = \operatorname{atan2}(y_d - y_t, x_d - x_t) \quad (\text{转换到 } [0^\circ, 360^\circ))$$
   - 计算与发射主朝向 $\theta_{\text{dir}}$ 的绝对角差：
     $$\Delta \theta = |\operatorname{angle\_difference}(\phi, \theta_{\text{dir}})|$$
   - 仅当 $\Delta \theta \le 90^\circ$ 且 $d \le r_{\text{recv}, i}$ 时，机器狗才处于信号辐射有效覆盖区内；否则无法接收到信号。

---

### 4.3 测向观测与误差模型 (`Arena.measure`)

当机器狗在信道 $i$ 执行测向时：
1. **未覆盖或目标已清除**：
   返回 `measure_result = "no_signal"`。
2. **近距饱和**：
   若处于覆盖区且 $d \le 5.0\,\text{m}$，测向机产生过载饱和，返回 `measure_result = "near"`，此时**不返回角度数值**。
3. **正常测向输出**：
   若处于覆盖区且 $d > 5.0\,\text{m}$，返回 `measure_result = "direction"`：
   - 计算机器狗指向目标的真实方位角：
     $$\alpha_{\text{true}} = \operatorname{atan2}(y_t - y_d, x_t - x_d) \quad (\text{归一化至 } [0^\circ, 360^\circ))$$
   - 叠加测量均匀随机噪声 $\epsilon \sim U[-1.0^\circ, +1.0^\circ]$：
     $$\alpha_{\text{obs}} = (\alpha_{\text{true}} + \epsilon) \bmod 360^\circ$$
   - 返回 `svd_deg = round(alpha_obs, 2)`。

---

### 4.4 激光清除机制与朝向无关准则 (`Arena.clear`)

根据题目附件规定：
1. 机器狗搭载光学精确定位与高能激光清除载荷，其有效杀伤探测半径为 $R_{\text{clear}} = 20.0\,\text{m}$。
2. **清除判定与主朝向无关**：哪怕目标是定向源且机器狗正处于其后半球盲区，只要几何距离 $d \le 20.0\,\text{m}$，光学扫描即可精确定位并使用激光将其摧毁。
3. 清除结果：
   - 若 $d \le 20.0\,\text{m}$ 且该信道目标未被清除：标记目标为已清除，返回 `clear_result = "success"`。
   - 若 $d > 20.0\,\text{m}$ 或目标已被清除：返回 `clear_result = "no_target_in_range"`。

---

### 4.5 严格虚拟时间状态机（计费规则）

仿真系统维护严格递增的高精度浮点数 `virtual_time_s`，每次指令执行时按以下顺序累加耗时：

$$\Delta T = T_{\text{move}} + T_{\text{switch}} + T_{\text{action}}$$

| 耗时类型 | 计算规则 | 说明 |
| :--- | :--- | :--- |
| **移动耗时 $T_{\text{move}}$** | $\Delta s / 5.0$ 秒 | 机器狗以恒定速率 $5.0\,\text{m/s}$ 从旧坐标直线匀速移动至新坐标 $(\Delta s = \sqrt{\Delta x^2 + \Delta y^2})$ |
| **切频耗时 $T_{\text{switch}}$** | 若新信道 $\ne$ 当前信道则为 $1.0$ 秒；否则为 $0.0$ 秒 | **仅在 `/measure` 时触发**；`/clear` 指令不占用也不切换测向机频段 |
| **动作耗时 $T_{\text{action}}$** | • `/measure`：固定 $5.0$ 秒<br>• `/clear` 成功：固定 $5.0$ 秒（光学定位 3.0s + 激光清除 2.0s）<br>• `/clear` 失败：固定 $3.0$ 秒（光学搜索 3.0s 未发现目标） | 严格对齐题目时间参数设定 |

---

### 4.6 请求幂等性机制

为应对网络重试或客户端异常重复发送：
- 模拟器内置 `cached_responses` 字典，以客户端传入的 `request_id` 为键。
- 若接收到已处理过的 `request_id`，模拟器直接返回上一次的缓存结果，**绝不重复移动、绝不重复累加虚拟时间、绝不重复变更目标状态**。

---

## 5. 官方评测 HTTP 协议接口规范

外部算法（如 `simulator.SimulatorClient`、`P3.py`、`p4.py`）通过标准 JSON-RPC 风格的 HTTP POST 协议与模拟器交互。

### 5.1 `POST /enter` (入场初始化)
机器狗进入场地，启动计时。

- **请求 Payload**:
  ```json
  {
    "request_id": "uuid-enter-01",
    "arena_id": "default",
    "robot_id": "dog-alpha"
  }
  ```
- **成功响应**:
  ```json
  {
    "accepted": true,
    "real_timestamp_ms": 1726180000000,
    "virtual_time_s": 0.0,
    "max_virtual_duration_s": 360000,
    "max_real_duration_s": 1200,
    "remaining_real_duration_s": 1200.0
  }
  ```

### 5.2 `POST /measure` (移动并测向)
指定新坐标与接收信道。机器狗先移动到该坐标，再切换到目标信道进行测向。

- **请求 Payload**:
  ```json
  {
    "request_id": "uuid-m-01",
    "arena_id": "default",
    "robot_id": "dog-alpha",
    "position": { "x": 300.0, "y": 450.0 },
    "channel": 3
  }
  ```
- **成功响应示例 1（测得方向）**:
  ```json
  {
    "accepted": true,
    "real_timestamp_ms": 1726180010000,
    "virtual_time_s": 113.2,
    "measure_result": "direction",
    "svd_deg": 68.45
  }
  ```
- **成功响应示例 2（近距饱和）**:
  ```json
  {
    "accepted": true,
    "real_timestamp_ms": 1726180020000,
    "virtual_time_s": 180.5,
    "measure_result": "near"
  }
  ```
- **成功响应示例 3（无信号）**:
  ```json
  {
    "accepted": true,
    "real_timestamp_ms": 1726180030000,
    "virtual_time_s": 235.0,
    "measure_result": "no_signal"
  }
  ```

### 5.3 `POST /clear` (移动并执行激光清除)
指定新坐标与目标信道。机器狗移动到该坐标，启动光学精确定位与激光清除载荷。

- **请求 Payload**:
  ```json
  {
    "request_id": "uuid-c-01",
    "arena_id": "default",
    "robot_id": "dog-alpha",
    "position": { "x": 312.0, "y": 456.0 },
    "channel": 3
  }
  ```
- **成功响应（范围内清除成功）**:
  ```json
  {
    "accepted": true,
    "real_timestamp_ms": 1726180040000,
    "virtual_time_s": 243.0,
    "clear_result": "success"
  }
  ```
- **响应（超出范围清除未命中）**:
  ```json
  {
    "accepted": true,
    "real_timestamp_ms": 1726180050000,
    "virtual_time_s": 241.0,
    "clear_result": "no_target_in_range"
  }
  ```

### 5.4 `POST /exit` (离场结算)
算法任务完成或超时，结束仿真，模拟器完成结算并打印战报。

- **请求 Payload**:
  ```json
  {
    "request_id": "uuid-exit-01",
    "arena_id": "default",
    "robot_id": "dog-alpha"
  }
  ```
- **成功响应**:
  ```json
  {
    "accepted": true,
    "real_timestamp_ms": 1726180060000,
    "virtual_time_s": 1286.4,
    "exit_reason": "user_exit"
  }
  ```

---

## 6. 控制台与管理 REST API

除官方通信接口外，`mock_simulator.py` 还提供了供前端 Web 控制台和自动化脚本调用的 REST 接口：

| HTTP 方法 | 接口路径 | 说明 |
| :--- | :--- | :--- |
| `GET` | `/` 或 `/index.html` | 返回可视化 WebUI 单页应用 |
| `GET` | `/api/live` | 获取当前对战的实时帧数据（机器狗位置、目标状态、最新测向锥等） |
| `GET` | `/api/logs` | 列出 `./logs/` 目录下所有已落盘的历史会话元数据 |
| `GET` | `/api/log?file=xxx` | 读取指定历史对战文件（如 `session_20260913_xxxx.json`）全部轨迹帧 |
| `GET/POST` | `/api/set_mode?mode=p3\|p4` | 动态切换模式并重新生成算例目标 |
| `GET/POST` | `/api/reset` | 停止当前运行算法，清空状态并重新生成算例（重置新局） |
| `GET` | `/api/algo_status` | 检查后台算法子进程运行状态（运行中/退出码/错误信息） |
| `GET/POST` | `/api/start` | 根据当前模式（P3/P4）在后台一键调用对应算法脚本执行 |
| `GET/POST` | `/api/stop` | 强行终止后台正在运行的算法子进程 |
| `GET` | `/status` | 导出当前模拟器内部完整真值数据（调试对比专用） |

---

## 7. WebUI 战术态势控制台功能详解

在浏览器访问 `http://127.0.0.1:2026/`，控制台包含以下功能模块：

```
+-------------------------------------------------------------------------+
| [CUMCM-2026] 战术态势控制台    [模式切换: P3 / P4] [重置新局]           |
+------------------------------------+------------------------------------+
|                                    | 仪表盘与战报统计                   |
|                                    |  • 虚拟时间 / 真实耗时             |
|                                    |  • 清除进度 (如 14/14, 100%)       |
|            Canvas 雷达主图         |  • 移动路程 / 移动耗时             |
|                                    |  • 测向次数 (方向/近距/无信号)     |
|   • 1800m 场地边界圈               |  • 切频次数与耗时                  |
|   • 干扰源分布与状态               |------------------------------------|
|     (绿色:已清除 / 红色:全向未清除 | 算法进程调度                       |
|      紫色扇形:定向源 180° 主朝向)  |  • 状态: 空闲 / 运行中             |
|   • 机器狗实时位置与运动折线轨迹   |  • [🚀 启动算法] [⏹ 停止算法]     |
|   • 示向度黄色误差锥 (±1.0°)       |------------------------------------|
|                                    | 历史录像回放 (Timeline Player)     |
|                                    |  • 下拉选择历史对战文件            |
|                                    |  • [▶ 播放] [⏸ 暂停] [⏭ 步进]   |
|                                    |  • 轨迹帧滑动条逐帧审查            |
+------------------------------------+------------------------------------+
```

### 1. Canvas 交互地图
- **平移与缩放**：支持鼠标左键拖拽平移画布，滚轮自由无级缩放。
- **干扰源渲染**：
  - **绿色实心圆**：已被成功清除的目标。
  - **红色空心圆**：未清除的全向源。
  - **紫色扇面**：未清除的定向源，直观绘出发射主朝向与 $180^\circ$ 辐射扇面，直观展示盲区。
- **机器狗轨迹**：蓝色高亮折线标出移动顺序与关键途径点。
- **测向误差锥**：动态展示示向度中心线与 $\pm 1.0^\circ$ 的半透明黄色观测锥形区。

### 2. 算法一键管理
- 点击右侧面板中的 **“启动算法”**，模拟器服务端会在后台使用合适环境的 Python 自动执行工作区对应的求解主程序（如 `P3.py` 或 `p4.py`），并将输出实时落盘至 `logs/runtime/algo_runner.log`。
- 点击 **“停止算法”** 可随时中止正在运行的外部进程。

### 3. 历史对战录像与单步复盘
- 下拉框自动列出 `logs/` 目录下全部对战记录。
- 支持自由拖动进度条，以 1 步为步长复现机器狗每一步的移动、测向与激光清除动作。

---

## 8. 与算法配合调测的最佳实践

### 典型调试工作流

#### 步骤 1：终端启动模拟器并开启详细日志
```bash
python3 mock_simulator.py --mode p4 --verbose
```

#### 步骤 2：浏览器打开监控页面
打开浏览器访问：`http://127.0.0.1:2026/`

#### 步骤 3：运行第四题（或第三题）求解程序
在另一个终端窗口中执行算法客户端（以第四题 `p4.py` 为例）：
```bash
python3 p4.py --run --base-url http://127.0.0.1:2026
```
*(或者直接在 WebUI 控制台上点击“启动算法”按钮)*。

#### 步骤 4：观察实时对战
- 页面将实时以 150ms 间隔平滑更新机器狗的位置移动、各信道的锁定、定向源盲区探针二分收敛以及最终清除过程。
- 任务执行完毕后，控制台与终端会同时输出最终战报。

---

## 9. 常见问题排查 (FAQ)

### Q1: 运行算法时报错 `URLError: connection refused` 或超时？
1. 确认 `mock_simulator.py` 服务是否已在后台运行并监听指定端口（默认 2026）。
2. 若开启了代理软件（Clash / Surge / 系统全局代理），系统可能会将 `127.0.0.1` 路由至代理从而连接失败。`mock_simulator.py` 已内置 `no_proxy` 屏蔽本地回环地址；若使用独立脚本，请在终端执行：
   ```bash
   export no_proxy=127.0.0.1,localhost
   ```

### Q2: 提示端口已被占用 (`OSError: Address already in use`)？
说明上一次运行的模拟器进程尚未完全退出，或端口 2026 被占用。可以通过指定其他端口启动：
```bash
python3 mock_simulator.py --port 2027
```
算法端启动时带上对应参数：
```bash
python3 p4.py --run --base-url http://127.0.0.1:2027
```

### Q3: Web 页面上点击“启动算法”提示未找到 numpy 或找不到环境？
模拟器内置了智能 Python 探测机制（依次扫描激活的 Conda 环境、`~/.conda/environments.txt` 中的 `cumcm2026-b` 环境及 PATH）。如果你的特定环境安装在特殊路径，可以使用 `--python-bin` 参数显式指定：
```bash
python3 mock_simulator.py --python-bin /Users/your_name/miniforge3/envs/cumcm2026-b/bin/python
```
