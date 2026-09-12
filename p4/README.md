# 第四题控制入口

在仓库根目录运行 `python p4.py` 可离线查看 17 个覆盖站坐标。
实际执行命令与 P3 相同，只更换脚本名称：

```bash
python -u p4.py --run --robot-id 2026001 --base-url http://127.0.0.1:2026 --log P4_run_log.jsonl
```

点位直接取自 `p4/verify_coverage.py` 的 `generate_stations()`，编号依次为
原点、8 个外圈点、8 个内圈点。外圈角度为 45° × i，内圈为 45° × i + 22.5°。
外圈半径为 R/cos(22.5°)，内圈半径为
R cos(22.5°) − sqrt(r² − (R sin(22.5°))²)。没有将外圈检测点裁掉或投影回场内。

`Problem4Controller` 继承 P3，唯一的控制策略差异是覆盖点列表。
任务池、开放 TSP、P2 模板缓存、共享二测、P1 定位、清除、异常处理、退出条件
和运行统计均复用 P3；初始任务为 17 × 20 = 340 项。访问顺序仍由 TSP 决定，
不是按脚本列出点位的顺序固定访问。默认日志名称为 `P4_run_log.jsonl`。
