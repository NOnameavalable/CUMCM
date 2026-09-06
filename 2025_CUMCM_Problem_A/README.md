# 2025 CUMCM Problem A 🚀

本仓库整理了我们对 **2025 年全国大学生数学建模竞赛 A 题** 的完整解题材料：论文、MATLAB 代码、结果表和论文插图。

项目主题是：在来袭导弹场景下，规划多架无人机的烟幕干扰弹投放策略，使真目标获得尽可能长的有效遮蔽时间。

![FY1-FY5 trajectory](<paper/FY1到FY5投放烟幕干扰弹的运动轨迹图.png>)

## 项目亮点

- 🧮 建立无人机、烟幕弹、烟幕云团和导弹的三维运动模型
- 🎯 使用圆柱目标离散采样判断“完全遮蔽”
- 🔍 用二分法精修遮蔽时间区间
- 🐦 用粒子群优化算法（PSO）求解连续决策变量
- 🧬 用 PSO-DE 融合算法提升高维搜索稳定性
- 🔗 用匈牙利算法处理无人机-导弹任务分配

## 五个问题的结果速览

| 问题 | 场景 | 方法 | 关键结果 |
| --- | --- | --- | --- |
| 问题一 | FY1 固定策略干扰 M1 | 运动学 + 离散采样 + 二分法 | 圆柱目标有效遮蔽 `1.39 s` |
| 问题二 | FY1 单弹优化 | PSO | 最优遮蔽 `4.64 s` |
| 问题三 | FY1 三弹优化 | PSO-DE + 边界采样 | 总遮蔽 `6.402 s` |
| 问题四 | FY1-FY3 各一弹干扰 M1 | PSO-DE | 总遮蔽 `13.36 s` |
| 问题五 | FY1-FY5、M1-M3 全局策略 | 匈牙利算法 + 分组优化 | 全局遮蔽 `20.5 s` |

## Visual Overview

### PSO convergence

![PSO convergence](<paper/PSO各参数收敛曲线.png>)

### Three-UAV shielding timeline

![FY1 FY2 FY3 shielding](<paper/FY1，FY2，FY3遮蔽时间.png>)

### Multi-missile shielding duration

![M1 M2 M3 shielding](<paper/对M1，M2，M3的遮蔽时长.png>)

## 仓库结构

```text
.
├── paper/
│   ├── 国赛A题.pdf          # 最终论文
│   ├── 国赛A题.tex          # LaTeX 源码
│   ├── cumcmthesis.cls      # CUMCM 模板类文件
│   └── *.png                # 论文插图
├── code/
│   ├── wenti_1.m            # 问题一主程序
│   ├── wenti_2.m            # 问题二主程序
│   ├── wenti_3.m            # 问题三主程序
│   ├── wenti_4.m            # 问题四主程序
│   ├── wenti_5.m            # 问题五主程序
│   └── *_*.m                # 辅助函数
└── results/
    ├── result1.xlsx         # 问题三结果
    ├── result2.xlsx         # 问题四结果
    └── result3.xlsx         # 问题五结果
```

## 如何运行

建议使用 MATLAB，并把当前工作目录切换到 `code/`：

```matlab
cd code
run("wenti_1.m")
run("wenti_2.m")
run("wenti_3.m")
run("wenti_4.m")
run("wenti_5.m")
```

其中：

- `wenti_1.m` 到 `wenti_5.m` 是五个问题的主入口
- 其他 `.m` 文件是遮蔽判据、采样、区间求交、变量解码等辅助函数
- `results/` 中的 Excel 文件是最终提交结果表

## 论文

- [`paper/国赛A题.pdf`](<paper/国赛A题.pdf>)：最终论文
- [`paper/国赛A题.tex`](<paper/国赛A题.tex>)：论文 LaTeX 源码

如需重新编译论文，请在 `paper/` 目录下编译 `国赛A题.tex`。

## 方法一句话

我们把烟幕遮蔽问题拆成“几何判定 + 运动学建模 + 启发式优化 + 任务分配”四层：先判断某时刻云团是否完全遮蔽真目标，再把遮蔽时长作为优化目标，最后用 PSO/PSO-DE 和匈牙利算法给出可执行的投放策略。
