# 问题一 LaTeX 初稿

依据仓库《问题一章节写作提纲.md》生成第 5 章，包含 5.1—5.6 六节、四幅图及两张数据表。省略可选单点扇区图和未运行的离散敏感性实验，图编号连续为 5-1—5-4。

- `main.tex`：独立编译入口，采用 UTF-8、ctexart 和 Fandol 字体。
- `chapter.tex`：正文，可在调整相对路径和导言区后并入完整论文。
- `tables.tex`、`numbers.tex`：程序生成的表格及数值宏。
- `results.json`：原始精度计算结果、输入数据及日志行号。
- `generate_results.py`：从仓库 P3_run_log.jsonl 频道 4 重算数值和算例图；需要 numpy、shapely、matplotlib 以及仓库 P1.py、utils.py。
- `figures/`：原有交会图 PDF 的副本、新增 TikZ 原理图和程序生成的算例 PDF/PNG。

在此目录运行两次 `xelatex -interaction=nonstopmode -halt-on-error main.tex`，或运行 `latexmk -xelatex main.tex`。需安装 ctex、Fandol、TikZ、amsmath、booktabs、geometry、hyperref 等标准宏包。整个 problem1 文件夹可上传 Overleaf，编译器选 XeLaTeX。

当前环境没有可用 Linux TeX 引擎；附带 Windows Tectonic 也因 WSL 互操作错误无法运行。因此尚未完成全文编译和分页检查，不能将源文件检查视为编译通过。诊断见 compile-status.txt 与 latex-status.json。算例脚本已经实际运行，算例图已检查；现有交会图沿用仓库成图，两个新增 TikZ 图待编译检查。

复现数值（从仓库根目录运行）：

```bash
/home/bingy/anaconda3/envs/cumcm2026-b/bin/python paper/problem1/generate_results.py
```

本初稿未修改算法源代码，也未假定日志属于正式评测。正文明确保留问题一默认不使用接收半径和近场盲区的建模口径。
