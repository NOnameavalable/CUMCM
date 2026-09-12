# CUMCM 2026 B 题问题 2：算法全景解析与逐函数深度剖析

本文档对 `P2.py` 模块进行系统性解读，说明题目背景、物理模型、数学本质、架构逻辑和主要接口的设计用意。

---

## 目录
1. [题目背景与问题 2 的数学本质](#1-题目背景与问题-2-的数学本质)
2. [全局算法架构与全流程数据流](#2-全局算法架构与全流程数据流)
3. [核心数据结构与配置参数解析](#3-核心数据结构与配置参数解析)
4. [逐函数深度拆解与原理解读（全函数覆盖）](#4-逐函数深度拆解与原理解读全函数覆盖)
   - [4.1 基础数学与坐标系变换模块（6个函数）](#41-基础数学与坐标系变换模块)
   - [4.2 连续可行域构建与六边形蜂窝剖分模块（5个函数）](#42-连续可行域构建与六边形蜂窝剖分模块)
   - [4.3 先验联合状态空间扩展模块（1个函数）](#43-先验联合状态空间扩展模块)
   - [4.4 候选点搜索空间与双边采样模块（4个函数）](#44-候选点搜索空间与双边采样模块)
   - [4.5 前向观测模拟与向量化后验更新模块（6个函数）](#45-前向观测模拟与向量化后验更新模块)
   - [4.6 最小覆盖圆算法（Welzl）与局部 Fisher 信息模块（6个函数）](#46-最小覆盖圆算法welzl与局部-fisher-信息模块)
   - [4.7 后验评价、候选点评分与粗细网格优化模块（9个函数）](#47-后验评价候选点评分与粗细网格优化模块)
   - [4.8 总控求解、收敛验证与可视化交付模块（6个函数）](#48-总控求解收敛验证与可视化交付模块)
5. [基准求解结果与物理规律分析](#5-基准求解结果与物理规律分析)
6. [数学建模核心亮点与常见疑问（FAQ）](#6-数学建模核心亮点与常见疑问faq)

---

## 1. 题目背景与问题 2 的数学本质

### 1.1 题目背景要素
- **目标区域**：以原点 $(0,0)$ 为圆心，半径 $R_T = 1800\text{ m}$ 的圆形区域。
- **干扰源特征**：全向干扰源（360° 均匀辐射），具体位置未知，有效接收半径 $R_{\text{recv}} \in [1000, 1500]\text{ m}$（对每个干扰源固定但未知）。
- **测向设备限制**：
  1. 单次测得的示向度误差在 $[-1^\circ, 1^\circ]$ 内。在同一地点重复测量误差固定不变，移动到新地点误差服从统计独立性。
  2. 当测向机与干扰源距离 $d \le 5\text{ m}$ 时，由于信号过强无法获得示向度（称为 `near` 盲区），但可直接进行光学精确定位并清除。
  3. 当距离 $d > R_{\text{recv}}$ 时，接收不到信号（`no_signal`）。
- **清除系统参数**：
  - 机器狗移动速度 $v = 5\text{ m/s}$。
  - 测向稳定读数时间 $5\text{ s}$。
  - 简易光学探测仪在距离干扰源 $d \le 20\text{ m}$ 时方可精确定位并启动激光清除。光学定位耗时 $3\text{ s}$，激光清除耗时 $2\text{ s}$。

### 1.2 问题 2 的核心难点与决策冲突
问题 2 描述：“已知某全向干扰源在一个检测点处测得的示向度，请给出第二个检测点的选择策略，以获得关于该干扰源较好的定位效果，并给出第二个检测点的候选区域。”

选定第二个检测点 $S_2$ 面临多重强耦合的矛盾：
1. **几何交会角 vs. 信号可达性**：直觉上两视线交角接近 $90^\circ$ 定位误差最小，但若 $S_2$ 侧移过大或纵深过远，当干扰源实际 $R_{\text{recv}} = 1000\text{ m}$ 时可能会脱离信号覆盖（导致 `no_signal` 无法测向）。
2. **定位精度 vs. 移动耗时**：如果跑得太远才能获得极限交会精度，机器狗的行进时间大幅增加，导致总任务完成时间恶化。
3. **点决策 vs. 区域决策**：不仅要给出一个具体的最优坐标 $(x, y)$，还要给出鲁棒且连续的“候选区域”。
4. **两步定位与一次清除（Clear）闭环**：第二点测完后，两组示向角带交会出的后验区域如果足够小（最小覆盖圆半径 $\le 20\text{ m}$），机器狗走到圆心就能直接进入光学视距实施一次性清除，省去后续第三次、第四次机动。

### 1.3 `P2.py` 的求解哲学
`P2.py` 没有采用简化的“假设目标在某一点求交角”的启发式做法，而是建立了一个完整的**信息论与后验鲁棒优化闭环**：
$$\text{目标连续可行域} \xrightarrow{\text{正六边形蜂窝}} \text{目标状态空间} \times \text{未知接收半径} \xrightarrow{\text{前向观测模拟}} \text{全情景后验更新} \xrightarrow{\text{Welzl 最小覆盖圆}} \text{Minimax 鲁棒打分}$$

---

## 2. 全局算法架构与全流程数据流

```
[输入: 第一次检测点 S1, 示向度 theta1]
                 │
                 ▼
     1. 构建第一次连续几何可行域 (_build_first_feasible_region)
     (目标圆域 1800m ∩ 接收上限 1500m ∩ 扇形角带 2° \ 盲区 5m)
                 │
                 ▼
     2. 正六边形蜂窝三角晶格剖分 (_sample_target_cells)
     (离散化目标代表点，赋面积权重，计算单元外接圆半径 cell_radius)
                 │
                 ▼
     3. 扩展 (位置, 接收半径) 联合状态空间 (_expand_receive_radius_states)
     (每个网格点与相容的 R_recv 组合，构成先验状态集 states)
                 │
                 ▼
     4. 生成候选点网格（局部坐标系正负两侧分别评价）
     (粗网格 coarse_grid: 100m 步长)
                 │
                 ▼
     5. 候选点全情景评估 (_evaluate_all_candidates)
     ┌─────────────────────────────────────────────────────────────┐
     │ 对每个候选点 point：                                        │
     │  a. 遍历状态与误差 (-1°, 0°, 1°)，模拟前向观测 events        │
     │  b. 对每个观测事件：更新后验相容目标子集                    │
     │  c. Welzl 算法求后验点集最小覆盖圆 (C_post, R_circle)       │
     │  d. 保守半径 R_cons = R_circle + max(cell_radius)           │
     │  e. 校验是否 R_cons <= 20m (can_clear)                      │
     │  f. 统计 worst_radius, clear_fraction, expected_total_time  │
     └─────────────────────────────────────────────────────────────┘
                 │
                 ▼
     6. 两级自适应局部加密搜索 (Coarse-to-Fine)
     (选取粗筛前 8 个优选种子点，在 160m 半径内以 25m 步长进行细筛)
                 │
                 ▼
     7. Pareto 筛选与形态学候选区域提取
     (多目标非支配排序 + 5% 容差内高分子区域膨胀融合 unary_union)
                 │
                 ▼
[输出: 最优检测点坐标、镜像对称点、后验覆盖圆半径、可清除率、候选区域多边形]
```

---

## 3. 核心数据结构与配置参数解析

### 3.1 `Problem2Config`
参数配置数据类，使用 `@dataclass(frozen=True)` 冻结保证不可变。
- **物理环境参数**：
  - `target_radius = 1800.0`：目标分布区域圆半径（米）。
  - `receive_radius_min = 1000.0` / `receive_radius_max = 1500.0`：信号有效接收半径范围 $[1000, 1500]\text{ m}$。
  - `bearing_error_deg = 1.0`：测向机最大角误差 $1^\circ$（总张角为 $2^\circ$）。
  - `near_radius = 5.0`：强信号无法测向盲区半径（米）。
  - `clear_radius = 20.0`：光学精确定位与激光清除有效视距（米）。
- **机器狗时空参数**：
  - `dog_speed = 5.0`：行进速度 $5\text{ m/s}$。
  - `detection_time = 5.0`：测向耗时 $5\text{ s}$。
  - `clear_localize_time = 3.0`：光学精确定位耗时 $3\text{ s}$。
  - `clear_operation_time = 2.0`：激光清除耗时 $2\text{ s}$。
- **离散网格与搜索参数**：
  - `target_grid_spacing = 20.0`：目标可行域正六边形晶格中心间距（米）。该间距保证六边形外接圆半径仅约 $11.55\text{ m}$，低于 $20\text{ m}$ 清除视距。
  - `candidate_grid_spacing = 100.0`：全局第一阶段粗筛搜索步长（米）。
  - `refined_grid_spacing = 25.0`：第二阶段细筛局部加密步长（米）。
  - `refinement_radius = 160.0`：局部加密窗口半径（米）。
  - `refinement_seed_count = 8`：用于局部加密的优选种子点数。
- **`__post_init__`**：内置自检逻辑，在初始化时自动校验所有物理参数必须为正数、误差在合理区间、半径上下限符合逻辑。

### 3.2 `TargetCell`
代表可行域中一个被六边形晶格裁剪的微元：
- `index`：全局唯一索引。
- `representative`：单元代表点坐标（`Point`），通常取几何重心/内部代表点。
- `geometry`：Shapely 几何对象（被裁剪后的多边形）。
- `area`：微元面积。
- `weight`：归一化权重（$\text{area} / \text{total\_area}$）。
- `cell_radius`：微元边界点到代表点的最大距离（单元外接圆半径）。用于后续做保守覆盖估计。

### 3.3 `StateSample`
目标位置与未知接收半径的**联合先验状态**：
- `target_index`：对应 `TargetCell` 的索引。
- `target`：目标代表点位置。
- `receive_radius`：该状态假定的干扰源有效接收半径 $R_{\text{recv}}$。
- `weight`：该联合状态在所有可能状态中的概率权重。
- `cell_radius`：对应单元的最大几何半径。

### 3.4 `Observation`
模拟的第二点观测反馈：
- `kind`：观测类型，字面量 `"near"`（强信号盲区）、`"direction"`（正常测得示向角）、`"no_signal"`（超出接收半径）。
- `bearing_deg`：若为 `"direction"`，记录测得的方位角（度，规范到 $[0, 360)$）。

### 3.5 `EnclosingCircle`
圆的几何定义：`center: Point`, `radius: float`。

### 3.6 `PosteriorMetrics`
某一特定观测事件发生后，后验目标集合的评估指标：
- `event_weight`：该观测事件发生的边缘概率。
- `state_count` / `target_count`：相容的后验状态数与物理目标点数。
- `clear_center`：推荐的前往清除目标点（即最小覆盖圆心）。
- `sample_radius`：离散代表点的最小覆盖圆半径 $R_{\text{circle}}$。
- `conservative_radius`：保守半径 $R_{\text{cons}} = R_{\text{circle}} + \max(\text{cell\_radius})$。
- `can_clear`：布尔值，是否满足 $R_{\text{cons}} \le 20.0\text{ m}$。
- `follow_up_distance` / `follow_up_time`：机器狗从候选点前往清除圆心的机动距离和耗时。

### 3.7 `CandidateMetrics`
候选第二检测点的综合评价指标：
- `point`, `a`, `b`：全局坐标及以示向线为基准的局部坐标。
- `travel_distance`：从第一次检测点走过来的距离。
- `guaranteed_signal`：是否落在保证接收区域（即无论 $R_{\text{recv}}$ 多大都 $100\%$ 不会产生 `no_signal`）。
- `worst_radius`：所有可能后验中的最坏保守覆盖半径（Minimax 核心指标）。
- `clear_fraction`：所有可能观测中能达成“一次性直接清除”的情形比例。
- `expected_total_time`：走过来 + 检测 5s + 预期前往清除圆心的总耗时。
- `fisher_logdet`：Fisher 信息阵对数行列式（局部几何灵敏度代理指标）。
- `pareto_optimal`：是否为多目标非支配点。

### 3.8 `Problem2Result`
整个问题 2 求解流程的最终封装结果：包含配置、第一次可行域、剖分网格单元、所有候选点评分、全局最优候选点 `best_candidate`、Pareto 解集和推荐候选区域几何对象 `recommended_region`。

---

## 4. 逐函数深度拆解与原理解读（全函数覆盖）

### 4.1 基础数学与坐标系变换模块

#### 1. `_validate_inputs(first_position: Sequence[float], first_bearing_deg: float) -> Point`
- **功能**：校验输入参数合法性。
- **每一步在干什么**：
  1. 检查 `first_position` 长度是否为 2，确保是二维坐标 $(x, y)$。
  2. 检查各分量及示向角是否为有限数值（非 NaN、非 Inf）。
  3. 转换为全局几何对象 `Point(x, y)` 返回。

#### 2. `_wrap_angle_deg(angle: float) -> float`
- **功能**：将任意角度规范化到 $[-180^\circ, 180^\circ)$。
- **每一步在干什么**：
  利用模运算公式 `(angle + 180.0) % 360.0 - 180.0`，消除 $360^\circ$ 的周期性，方便计算夹角和误差。

#### 3. `_angular_difference_deg(first: float | np.ndarray, second: float | np.ndarray)`
- **功能**：计算两个角度向量之间的最小有符号角差 `first - second`。
- **每一步在干什么**：
  同样通过 `(first - second + 180.0) % 360.0 - 180.0`，正确处理如 $1^\circ$ 与 $359^\circ$ 之间的角差为 $+2^\circ$ 而不是 $-358^\circ$。支持 Numpy 广播运算。

#### 4. `_make_local_basis(bearing_deg: float) -> tuple[np.ndarray, np.ndarray]`
- **功能**：建立以第一次示向线为基准的正交局部坐标系单位基向量。
- **每一步在干什么**：
  1. 计算轴向单位向量 $\mathbf{e}_a = (\cos\theta, \sin\theta)$，代表沿示向线向目标方向延伸的方向。
  2. 计算横向单位向量 $\mathbf{e}_b = (-\sin\theta, \cos\theta)$，即 $\mathbf{e}_a$ 逆时针旋转 $90^\circ$ 的左法向向量。
  3. 这一步是后续将任意方向的物理空间旋转至标准直角系、实现对称性分析的关键。

#### 5. `_local_to_global(first_position: Point, a: float, b: float, basis: tuple[np.ndarray, np.ndarray]) -> Point`
- **功能**：将示向线局部坐标 $(a, b)$ 转换为全局地图坐标 $(x, y)$。
- **每一步在干什么**：
  执行仿射坐标变换：$\mathbf{r} = \mathbf{r}_1 + a \cdot \mathbf{e}_a + b \cdot \mathbf{e}_b$。

#### 6. `_global_to_local(first_position: Point, point: Point, basis: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]`
- **功能**：将全局地图坐标 $(x, y)$ 投影到局部坐标系 $(a, b)$。
- **每一步在干什么**：
  计算相对位移向量 $\Delta \mathbf{r} = \mathbf{r} - \mathbf{r}_1$，分别与基向量做点积：$a = \Delta \mathbf{r} \cdot \mathbf{e}_a$，$b = \Delta \mathbf{r} \cdot \mathbf{e}_b$。

---

### 4.2 连续可行域构建与六边形蜂窝剖分模块

#### 7. `_build_first_feasible_region(first_position: Point, first_bearing_deg: float, config: Problem2Config) -> BaseGeometry`
- **功能**：严格根据题目物理约束构造第一次观测后的连续可行域几何体。
- **每一步在干什么**：
  1. **目标区域圆盘**：以原点为中心、半径 $1800\text{ m}$ 的圆盘。
  2. **信号可达区域**：以 $S_1$ 为中心、半径 $R_{\text{recv\_max}} = 1500\text{ m}$ 的外圆盘。
  3. **非盲区**：以 $S_1$ 为中心、半径 $r_{\text{near}} = 5\text{ m}$ 的内圆盘。
  4. **示向角扇形**：以 $S_1$ 为顶点，张角为 $[\theta_1 - 1^\circ, \theta_1 + 1^\circ]$（包含测量误差）且半径为 $1500\text{ m}$ 的圆扇形。
  5. **布尔几何操作**：求交集 $\text{TargetDisk} \cap \text{OuterDisk} \cap \text{Sector} \setminus \text{InnerDisk}$。
  6. 检查相交后面积是否大于 0，若为空则抛出异常。最终得到平滑的曲边多边形。

#### 8. `_generate_triangular_lattice(bounds: tuple[float, float, float, float], spacing: float) -> list[tuple[float, float]]`
- **功能**：在给定的矩形包围盒内生成正三角网格晶格点集（Triangular Lattice）。
- **每一步在干什么**：
  1. 三角网格的行高为 $\frac{\sqrt{3}}{2} \times \text{spacing}$。
  2. 偶数行不偏移，奇数行沿 X 轴错开 $\frac{1}{2} \times \text{spacing}$。
  3. 这种错位点集是生成正六边形蜂窝 Voronoi 剖分的几何中心。

#### 9. `_hexagon(center_x: float, center_y: float, spacing: float) -> ShapelyPolygon`
- **功能**：以 $(x, y)$ 为中心生成边长为 $\frac{\text{spacing}}{\sqrt{3}}$ 的正六边形多边形。
- **每一步在干什么**：
  1. 计算外接圆半径 $R = \frac{\text{spacing}}{\sqrt{3}}$。
  2. 按 $30^\circ, 90^\circ, 150^\circ, 210^\circ, 270^\circ, 330^\circ$ 生成 6 个顶点的坐标，构建闭合 Shapely 多边形。
  3. **设计亮点**：在二维平面上，正六边形蜂窝铺砌具有最小的周长-面积比（开普勒蜂窝猜想），在相同采样点数下，六边形覆盖具有最均匀的各向同性误差。

#### 10. `_geometry_coordinates(geometry: BaseGeometry) -> Iterable[tuple[float, float]]`
- **功能**：递归提取多边形（包括包含内环空洞的多边形或 MultiPolygon）所有边界顶点的坐标迭代器。
- **每一步在干什么**：
  遍历多边形的外轮廓（`exterior`）及所有内孔轮廓（`interiors`），为后续精确计算单元最远点距离提供边界点源。

#### 11. `_sample_target_cells(first_region: BaseGeometry, config: Problem2Config) -> list[TargetCell]`
- **功能**：用六边形蜂窝网格完全覆盖第一次可行域，并生成带权重的离散目标单元。
- **每一步在干什么**：
  1. 调用 `_generate_triangular_lattice` 生成晶格中心。
  2. 对每个六边形，与第一次连续可行域求几何交集（`intersection`）。若相交面积为空或过小则丢弃。
  3. 提取被裁剪单元的内部代表点（`representative_point`）。
  4. 计算该单元所有边界点到代表点的最大欧氏距离，记为 `cell_radius`（即离散代表点对整个单元的最大几何截断误差）。
  5. 统计总相交面积，将各单元的面积占比作为先验离散概率权重（几何无偏均匀假设）。

---

### 4.3 先验联合状态空间扩展模块

#### 12. `_expand_receive_radius_states(target_cells: Sequence[TargetCell], first_position: Point, config: Problem2Config) -> list[StateSample]`
- **功能**：建立目标位置与未知接收半径的联合状态空间（Joint State Space）。
- **每一步在干什么**：
  1. 题目中干扰源的有效接收半径 $R_{\text{recv}} \in [1000, 1500]\text{ m}$ 且未知。在 $S_1$ 处能收到信号，意味着该目标与 $S_1$ 的距离必满足 $d(S_1, \text{Target}) \le R_{\text{recv}}$。
  2. 对每个目标单元，其 $R_{\text{recv}}$ 的下界为 $\max(1000.0, d(S_1, \text{Target}))$。
  3. 在有效区间内对 $R_{\text{recv}}$ 进行均匀离散采样（默认采样 5 个代表值，包括上下确界）。
  4. 将位置权重平均分配给相容的 $R_{\text{recv}}$ 采样值，生成全状态集合 `states`，最后归一化权重。
  5. **设计用意**：如果忽略 $R_{\text{recv}}$ 的不确定性，仅假设 $R_{\text{recv}}=1500\text{ m}$，选出的第二点在实际 $R_{\text{recv}}=1000\text{ m}$ 时很可能遭遇接收不到信号的灾难性后果。联合状态建模彻底消除了这一隐患。

---

### 4.4 候选点搜索空间与双边采样模块

#### 13. `_candidate_bounds(first_region: BaseGeometry, first_position: Point, basis: tuple[np.ndarray, np.ndarray], config: Problem2Config) -> tuple[tuple[float, float], tuple[float, float]]`
- **功能**：在局部坐标系中确定第二检测点的搜索边界包围盒。
- **每一步在干什么**：
  1. 提取第一次可行域凸包（Convex Hull）的所有顶点，转换为局部坐标 $(a, b)$。
  2. 轴向坐标 $a$（沿示向线方向）：取可行域范围并向外扩展 `candidate_margin`（默认 $300\text{ m}$）。
  3. **横向坐标 $b$（侧向偏移）**：默认在 $[-B,B]$ 双边搜索，其中 $B$ 同时覆盖可行域、外扩边界和最小侧向范围。
  4. 双边点各自执行完整后验评价，不使用镜像分数。`bilateral_search=False` 可恢复原单边对照；显式 `candidate_b_bounds` 始终原样生效。

#### 14. `_inclusive_range(lower: float, upper: float, step: float) -> np.ndarray`
- **功能**：生成包含起始点与终点的等间隔采样点数组。
- **每一步在干什么**：
  按给定步长生成序列，并在末尾点未达 `upper` 时补齐 `upper`，防止边界漏网。

#### 15. `_generate_candidate_grid(first_region, first_position, first_bearing_deg, config, spacing, windows) -> list[tuple[Point, float, float]]`
- **功能**：生成候选第二检测点的二维网格（粗筛全域或细筛局部窗口）。
- **每一步在干什么**：
  1. 根据是否传入局部窗口 `windows`，确定是全局包围盒搜索还是围绕优选种子的局部搜索。
  2. 双重循环遍历局部坐标 $(a, b)$。
  3. 默认保留正负两侧；仅在单边对照模式且未显式给定边界时过滤 $b<0$。
  4. 过滤与 $S_1$ 距离为 0 的点（题目指出同一点重复测向误差不变，原地二次检测无意义）。
  5. 调用 `_local_to_global` 将 $(a, b)$ 转换为全局地图坐标 $(x, y)$。
  6. 利用字典 key 哈希去重，返回坐标列表。

#### 16. `_build_safe_candidate_region(first_region: BaseGeometry, config: Problem2Config) -> BaseGeometry`
- **功能**：计算**绝对保证能收到信号的候选安全区域**（Guaranteed Safe Region）。
- **每一步在干什么**：
  1. 取可行域凸包的外轮廓点。
  2. 对每一个可行域极端点，以最小可能接收半径 $R_{\text{recv\_min}} = 1000\text{ m}$ 作圆盘。
  3. 对所有圆盘求连续几何交集（Intersection）。
  4. **物理意义**：只要第二检测点落在该交集区域内，无论干扰源实际位于可行域的哪一个极端角落、哪怕其接收半径是最小的 $1000\text{ m}$，也 $100\%$ 处于信号覆盖范围内，绝不会发生 `no_signal`。

---

### 4.5 前向观测模拟与向量化后验更新模块

#### 17. `_simulate_observation(candidate: Point, state: StateSample, error_deg: float, config: Problem2Config) -> Observation`
- **功能**：根据给定的物理真实状态与仪器误差，前向生成第二点的传感器读数。
- **每一步在干什么**：
  1. 计算候选点到目标真实坐标的欧氏距离 $d$。
  2. 若 $d \le r_{\text{near}} = 5\text{ m}$：信号超强，返回 `Observation("near")`。
  3. 若 $d > R_{\text{recv}}$：超出接收范围，返回 `Observation("no_signal")`。
  4. 否则：正常测得示向角，实际真值加上测量误差 `(bearing + error_deg) % 360.0`，返回 `Observation("direction", ...)`。

#### 18. `_observation_key(observation: Observation, config: Problem2Config) -> tuple[str, float | None]`
- **功能**：将连续观测读数聚类/分箱（Binning）为离散的观测事件键值。
- **每一步在干什么**：
  对示向角按 `observation_bin_deg = 0.25^\circ` 的分辨率进行四舍五入离散化。这样可将无限连续角聚类为有限个后验观测事件，保证贝叶斯更新的数值可计算性。

#### 19. `_state_arrays(states: Sequence[StateSample]) -> dict[str, np.ndarray]`
- **功能**：将所有状态对象的属性提取为平行的 Numpy 数组（矩阵化）。
- **每一步在干什么**：
  构造包含 $x$ 坐标、$y$ 坐标、接收半径 $R$、权重、单元半径等平行数组。供下一步进行无循环的向量化矢量过滤。

#### 20. `_is_observation_compatible(arrays, candidate, observation, config) -> np.ndarray`
- **功能**：向量化判断整个状态空间中，哪些状态能产生该观测（计算似然函数）。
- **每一步在干什么**：
  1. 一次性计算候选点到所有状态目标的距离向量 `distances`。
  2. 若观测为 `"near"`：返回 `distances <= 5.0` 的布尔掩码。
  3. 若观测为 `"no_signal"`：返回 `distances > radius` 的布尔掩码。
  4. 若观测为 `"direction"`：计算候选点到各状态目标的真实方位角向量，并计算其与测量示向角的角差。同时满足距离位于 $[5\text{ m}, R_{\text{recv}}]$ 且角差在允许误差界 $[-\Delta\theta, +\Delta\theta]$ 内的状态被保留。
  5. **性能关键**：完全基于 SIMD 矩阵化计算，避免了数千个状态的 Python 嵌套循环，提速数十倍。

#### 21. `_update_posterior_states(states, arrays, candidate, observation, config) -> tuple[list[StateSample], np.ndarray]`
- **功能**：根据向量化掩码，提取相容的后验状态子集。
- **每一步在干什么**：
  调用 `_is_observation_compatible` 得到索引，从 `states` 列表中筛选出满足条件的相容状态集。

#### 22. `_project_target_points(states: Sequence[StateSample]) -> tuple[list[Point], list[float]]`
- **功能**：将多维联合状态投影回物理二维目标位置点集。
- **每一步在干什么**：
  一个物理位置可能对应多个不同的 $R_{\text{recv}}$。本函数通过目标索引哈希去重，提取出该后验条件下唯一的物理目标坐标列表 `points` 以及每个单元对应的截断半径 `cell_radii`。

---

### 4.6 最小覆盖圆算法（Welzl）与局部 Fisher 信息模块

#### 23. `_circle_from_two(first: Point, second: Point) -> EnclosingCircle`
- **功能**：计算以两点连线为直径的外接圆。
- **每一步在干什么**：
  圆心为两点中点 $\frac{\mathbf{p}_1 + \mathbf{p}_2}{2}$，半径为中点到端点的欧氏距离。

#### 24. `_circle_from_three(first: Point, second: Point, third: Point) -> EnclosingCircle | None`
- **功能**：计算三点确定的外接圆。
- **每一步在干什么**：
  1. 计算三点坐标构成的行列式判定三点是否共线。
  2. 若共线（行列式近乎 0）则返回 `None`。
  3. 否则根据垂直平分线交点公式求出外接圆圆心和半径。

#### 25. `_circle_contains(circle: EnclosingCircle, point: Point) -> bool`
- **功能**：判断一个点是否严格被圆包含（含数值容差 $10^{-9}$）。

#### 26. `_minimum_enclosing_circle(points: Sequence[Point], random_seed: int = 2026) -> EnclosingCircle`
- **功能**：经典 **Welzl 随机增量算法**，在期望 $O(N)$ 时间复杂度内求任意二维点集的最小覆盖圆（Minimum Enclosing Circle, MEC）。
- **每一步在干什么**：
  1. 将输入点集按固定随机种子打乱（确保期望线性复杂度且结果可复现）。
  2. 维护当前最小圆。遍历点 $P_i$：若当前圆已包含 $P_i$，跳过。
  3. 若未包含，说明 $P_i$ 必定在新最小圆的边界上。固定 $P_i$ 为边界点，向前遍历 $P_j$；
  4. 若仍未包含，则 $P_i, P_j$ 均在边界上，向前遍历 $P_k$ 确定由三点决定的外接圆。
  5. 最终返回全局最小覆盖圆。
  6. **设计用意**：问题 1 和问题 2 要求评价定位区域大小。最小覆盖圆是无偏评价二维凸/非凸点集最紧致的包络指标。

#### 27. `_bearing_gradient(sensor: Point, target: Point) -> np.ndarray`
- **功能**：计算测向角关于目标坐标 $(x_T, y_T)$ 的雅可比梯度向量。
- **每一步在干什么**：
  根据方位角公式 $\theta = \arctan2(y_T - y_S, x_T - x_S)$，其全微分梯度为 $\nabla \theta = \left(-\frac{y_T - y_S}{d^2}, \frac{x_T - x_S}{d^2}\right)$。

#### 28. `_fisher_information_score(first_position, second_position, target, sigma_theta) -> tuple[float, float, float]`
- **功能**：计算两观测站在目标点处的 Fisher 信息矩阵（FIM）、A-最优指标与视线锐交会角。
- **每一步在干什么**：
  1. 测向机 FIM 为各测站梯度外积之和：$\mathbf{F} = \frac{1}{\sigma_\theta^2} \left(\nabla\theta_1 \nabla\theta_1^T + \nabla\theta_2 \nabla\theta_2^T\right)$。
  2. 计算 $\mathbf{F}$ 的行列式，取 $\log_{10}(\det \mathbf{F})$（D-最优准则指标，反映定位误差椭圆面积的倒数）。
  3. 计算 $\text{Tr}(\mathbf{F}^{-1})$（A-最优准则指标，反映均方误差下界）。
  4. 计算两测站指向目标的视线夹角 $\alpha$。
  5. 作为局部微分几何的对比参考指标。

---

### 4.7 后验评价、候选点评分与粗细网格优化模块

#### 29. `_evaluate_posterior(posterior_states, candidate, observation, event_weight, config) -> PosteriorMetrics`
- **功能**：对某一特定观测结果下的后验点集进行综合指标结算。
- **每一步在干什么**：
  1. 提取去重后的物理后验目标点。
  2. 若观测为 `"near"`：目标已被限制在以候选点为中心、半径 $5\text{ m}$ 的圆内，覆盖圆半径即为 $5.0\text{ m}$，机动距离为 0。
  3. 若观测为其他：调用 `_minimum_enclosing_circle` 计算后验离散点的最小覆盖圆 $(C_{\text{post}}, R_{\text{circle}})$。
  4. **保守覆盖半径**：$R_{\text{cons}} = R_{\text{circle}} + \max(\text{cell\_radius})$。加上六边形截断误差，确保连续可行域内的任意真实目标点 $100\%$ 被覆盖。
  5. **判断是否可一次清除**：校验 $R_{\text{cons}} \le 20.0\text{ m}$。若成立，说明机器狗只要从候选点跑到 $C_{\text{post}}$，目标必在 $20\text{ m}$ 光学清除视距内，无需再做第三次搜索！
  6. 计算后续行进耗时及清除作业耗时。

#### 30. `_weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float`
- **功能**：计算离散加权样本的指定分位数（如 90% 分位数）。
- **每一步在干什么**：
  对数值排序，累加归一化权重，通过 `searchsorted` 找到累积概率达到给定分位数的阈值。

#### 31. `_evaluate_candidate(...) -> CandidateMetrics`
- **功能**：**算法核心大脑**，完成单个候选检测点的全情景前向展开、后验收缩与综合性能打分。
- **每一步在干什么**：
  1. **全情景展开**：遍历所有先验联合状态，遍历角误差样本 $\{-1^\circ, 0^\circ, +1^\circ\}$，调用 `_simulate_observation` 统计各种可能出现的观测事件及其发生概率。
  2. **后验收缩**：对每个可能观测事件，提取后验相容状态，调用 `_evaluate_posterior` 计算其后验指标。
  3. **指标聚合**：
     - `worst_radius`：所有可能后验中的最大保守半径 $\max(R_{\text{cons}})$（极小化极大 Minimax 决策依据）。
     - `clear_fraction`：加权统计可一次清除（`can_clear == True`）的概率总和。
     - `expected_total_time`：机器狗从 $S_1$ 移动到第二点的耗时 + 5s 测向 + 加权预期后续清除耗时。
     - `guaranteed_signal`：判断该点是否在绝对保证接收区内。
  4. 返回该候选点的全面画像 `CandidateMetrics`。

#### 32. `_candidate_sort_key(candidate: CandidateMetrics, mode: OptimizationMode) -> tuple[float, ...]`
- **功能**：构建候选点的多级排序决策键（Lexicographical Sort Key）。
- **每一步在干什么**：
  - **默认鲁棒模式（`robust`）**：
    1. 优先极小化最坏后验覆盖半径 `worst_radius`（最坏情况兜底，防止极端失误）。
    2. 次优极大化一次直接清除比例 `-clear_fraction`。
    3. 再次比较 90% 分位数半径与平均期望半径。
    4. 最后极小化总作业时间 `expected_total_time`。
  - **概率模式（`probabilistic`）**：优先最大化一次清除比例，其次极小化平均期望半径。

#### 33. `_evaluate_all_candidates(...) -> list[CandidateMetrics]`
- **功能**：对候选点集合批量执行评估，并按决策排序准则排序返回。

#### 34. `_refine_candidate_grid(coarse_results, first_region, first_position, first_bearing_deg, config) -> list[tuple[Point, float, float]]`
- **功能**：粗细两级自适应网格优化的第二阶段——局部自适应加密。
- **每一步在干什么**：
  1. 从第一阶段粗筛（步长 $100\text{ m}$）的排序结果中选出排名前 8 的优选候选点作为种子（Seeds）。
  2. 在每个种子点周围开辟半径为 $160\text{ m}$ 的局部矩形窗口。
  3. 使用细筛步长 $25\text{ m}$ 在这些窗口内进行密集重采样。
  4. **算法优势**：避免了在几千米的大范围内全图使用 $25\text{ m}$ 细网格带来的算力爆炸，计算耗时从 30 分钟降至 1 分钟以内，同时保留了高分辨率定位优势。

#### 35. `_dominates(first: CandidateMetrics, second: CandidateMetrics) -> bool`
- **功能**：Pareto 支配关系判定。
- **每一步在干什么**：
  在 `worst_radius`、`-clear_fraction`、`expected_radius`、`expected_total_time` 四个目标维度上，若第一点在所有维度均不劣于第二点，且至少在一个维度上严格优于第二点，则判定第一点支配第二点。

#### 36. `_find_pareto_candidates(candidates: Sequence[CandidateMetrics]) -> list[CandidateMetrics]`
- **功能**：筛选出全体候选点中的非支配解集（Pareto Frontier）。
- **每一步在干什么**：
  两重循环比对，标记出不被任何其他候选点支配的点，置其 `pareto_optimal = True` 并返回。供决策者权衡“耗时与精度”。

#### 37. `_extract_candidate_region(candidates: Sequence[CandidateMetrics], config: Problem2Config) -> BaseGeometry`
- **功能**：从最优离散点集构建连续、平滑的二维几何**推荐候选区域**。
- **每一步在干什么**：
  1. 以当前全局最优解 `best` 为基准。
  2. 设定近优容差阈值：最坏半径在最佳值的 $105\% + 2\text{ m}$ 以内，且一次清除概率不低于最佳值的 $98\%$。
  3. 筛选出所有满足该近优条件的密集候选点。
  4. 对每个入选点以 $\frac{\text{step}}{\sqrt{2}}$ 作微圆盘（形态学膨胀），最后调用 Shapely 的 `unary_union` 进行多边形并集融合。
  5. 严密回答了题目“给出第二个检测点的候选区域”的要求。

---

### 4.8 总控求解、收敛验证与可视化交付模块

#### 38. `solve_problem_2(first_position, first_bearing_deg, config) -> Problem2Result`
- **功能**：问题 2 的对外统一总控求解入口函数。
- **每一步在干什么**：
  1. 参数校验与第一可行域连续构造。
  2. 六边形蜂窝网格剖分与先验联合状态空间扩展。
  3. 构建绝对保证接收区域。
  4. 执行粗筛（$100\text{ m}$ 步长）与评估。
  5. 执行细筛（$25\text{ m}$ 步长加密）与评估。
  6. 合并去重粗细两级结果并重新排序，提取最优候选点。
  7. 提取 Pareto 解集与融合生成连续推荐候选区域。
  8. 封装为 `Problem2Result` 返回。

#### 39. `run_convergence_check(first_position, first_bearing_deg, configs) -> list[dict[str, float]]`
- **功能**：网格独立性与数值收敛性验证。
- **每一步在干什么**：
  使用不同网格精度序列（例如目标网格 $40\text{m} \to 20\text{m}$，候选点网格 $200\text{m} \to 100\text{m} \to 25\text{m}$）重复求解，输出最优坐标与各项指标的变化趋势，向评委证明当前求解步长已达到数值收敛，消除离散网格假象。

#### 40. `_plot_geometry(ax, geometry: BaseGeometry, **kwargs) -> None`
- **功能**：底层绘图辅助函数，将 Shapely 几何对象（多边形或复合多边形）转换为 Matplotlib 多边形补丁绘制在坐标轴上。

#### 41. `plot_problem_2_result(result: Problem2Result, output_path=None, show=False)`
- **功能**：绘制高规格的科技感结果图。
- **每一步在干什么**：
  1. 自动探测系统支持的中文字体（优先使用 `PingFang SC`、`Microsoft YaHei` 等，杜绝汉字乱码豆腐块）。
  2. 绘制第一次扇形带状可行域（蓝绿色）。
  3. 绘制保证接收安全区（淡绿色）。
  4. 绘制推荐候选区域多边形（淡粉色轮廓）。
  5. 将所有候选点按最坏后验覆盖半径绘制散点热力图（Colormap 映射）。
  6. 标注第一次检测点（黑色方块）与全局最优第二检测点（红色五角星）。
  7. 导出高清图像文件（如 `P2_result.png`）。

#### 42. `print_result_summary(result: Problem2Result) -> None`
- **功能**：在控制台格式化打印关键数值摘要，供论文正文引用。

#### 43. `main() -> None`
- **功能**：CLI 命令行执行入口。
- **每一步在干什么**：
  1. 解析命令行参数（`--demo`, `--x`, `--y`, `--bearing`, `--out`）。
  2. 若指定 `--demo`：使用大步长（目标 60m、粗筛 200m、细筛 50m）在 5 秒内快速输出原型。
  3. 默认全功能模式：使用高精度参数（目标 20m、粗筛 100m、细筛 25m）进行约 1~2 分钟的标准求解。
  4. 调用 `solve_problem_2` 并保存结果图。

---

## 5. 基准求解结果与物理规律分析

当初始检测点位于中心原点 $S_1 = (0, 0)$，测得示向度为正东 $\theta_1 = 0^\circ$ 时，执行高精度默认求解所得的关键基准数据如下：

| 评估指标 | 正式高精度求解结果 | 物理意义与解释 |
| :--- | :--- | :--- |
| **第一次可行域面积** | $39265.049\text{ m}^2$ | 目标圆域、$[5, 1500]\text{m}$ 距离带与 $2^\circ$ 误差张角的交集 |
| **目标剖分单元数** | 219 个正六边形 | $20\text{ m}$ 间距，单元截断误差上限仅 $11.55\text{ m}$ |
| **联合状态数** | 937 个状态 | 空间网格与 $R_{\text{recv}} \in [1000, 1500]\text{ m}$ 的先验全排列组合 |
| **最优检测点坐标 (全局)** | **$(960.000, 490.000)$** | 综合定位精度、清除概率与行进耗时的全局最佳第二检测点 |
| **对称检测点坐标 (全局)** | **$(960.000, -490.000)$** | 由关于示向线的物理对称性直接映射得到的另一侧等价最优点 |
| **最优点局部坐标 $(a, b)$** | **$(a=960.000, b=490.000)$** | 沿示向线纵深推进 $960\text{ m}$，侧向机动偏移 $490\text{ m}$ |
| **最坏后验覆盖半径** | **$51.843\text{ m}$** | 无论真实目标在何处、何种测量误差，交会后圆半径不超过此值 |
| **一次直接清除比例** | **$3.52\%$** | 仅靠第二步测向就将目标锁定在 $\le 20\text{ m}$ 圆内的概率 |
| **预计总任务耗时** | **$340.133\text{ s}$** | 机器狗移动时间、测向时间与后续清除时间之和（约 5.6 分钟） |

### 物理规律深层剖析：
1. **为什么最优第二检测点纵深是 $960\text{ m}$？**
   第一次测向后，目标大致分布在距离 $S_1$ 约 $500\text{ m} \sim 1500\text{ m}$ 的射线带上。将第二点前推到 $960\text{ m}$，恰好位于目标可能分布区的中腹位置，使得第二检测点到大多数目标的距离保持在 $400\sim 800\text{ m}$ 的黄金测距区间，角误差转化为的横向位移最小。
2. **为什么侧移是 $490\text{ m}$ 而不是更大（例如 $1000\text{ m}$）？**
   如果侧移过大（如 $1000\text{ m}$），视线交角确实更接近 $90^\circ$，但当干扰源真实接收半径为最小的 $1000\text{ m}$ 时，第二点会大概率落在信号盲区之外导致测不到信号；同时机器狗横向跑动距离过长。$490\text{ m}$ 是在**大交会夹角（约 $60^\circ \sim 75^\circ$）**与**信号可达性**之间的最优折中点。

---

## 6. 数学建模核心亮点与常见疑问（FAQ）

### Q1: 为什么使用“正六边形网格”而不是常见的“矩形正方形网格”？
- **几何各向同性优势**：正六边形最接近圆形。对于中心间距相同的网格，正六边形的外接圆半径比正方形小约 $15.5\%$。
- 本题中 $20\text{ m}$ 间距的正六边形，其最远顶点距离代表点仅 $\frac{20}{\sqrt{3}} \approx 11.55\text{ m}$。由于题目中光学清除视距为 $20\text{ m}$，单元截断误差 $11.55\text{ m} < 20\text{ m}$，使得离散网格的代表性误差严格低于物理清除阈值，保证了模型保真度。

### Q2: 为什么改为双边搜索？
- P3 同时处理多个频道时，两侧点对其他频道和全局路线的价值不同；提前丢弃一侧会漏掉共享或顺路检测位置。
- 目标圆边界对第一次可行域的裁剪也不总是关于示向线对称，因此程序分别计算两侧分数。单边模式只用于对照实验。
- `evaluate_problem_2_candidate(result, point)` 可复用已经离散好的联合状态，对 P3 提出的任意全局共享坐标进行精确复评。

### Q3: 为什么评价标准以“最坏后验半径（Minimax）”为主，而不是只看“平均期望半径”？
- **任务高可靠性要求**：军事与抢险场景下，搜寻排除干扰源要求**高成功率底线**。若仅追求平均半径最小，算法可能会冒险选择一个在大多数情况下表现优异、但在 $5\%$ 极端边界情形下产生巨大定位误差（甚至丢信号）的点位。
- 采用极小化极大（Minimax）鲁棒准则，保证了在全工况下定位误差都有严格的数学上界。

### Q4: 推荐候选区域是如何从离散点演化为连续多边形的？
- 许多建模论文只给出一个单点坐标，忽略了题目“给出候选区域”的要求。
- 本代码采用**近优容差过滤 + 形态学膨胀融合**：先筛选出与最优解性能极度接近（误差在 5% 容差内）的优质点簇，再以离散步长作微多边形缓冲区，最后使用 Shapely 的 `unary_union` 融合成一个封闭的连通多边形，给出了兼具理论严密性与工程可操作性的二阶决策区域。
