# 三连通硬件架构上表面码的通用逻辑操作及其线路实现

**Universal logical operations and their circuit implementations for surface codes on trivalent hardware architectures**

本仓库提供线路生成、采样、统计重绘和距离核验所需的代码与小型数据。完整冻结线路、逐分片统计和故障证据见数据附件。

```bash
git clone --branch master --recurse-submodules https://github.com/QEC-FPGA/trivalent-logical-operations.git
cd trivalent-logical-operations
python -m pip install -r requirements-plot.txt
python tools/check_results.py
python reproduce_figures.py
```

重绘图 9–12 不需要重新采样。输出位于 `reproduced_figures/`。

| 内容 | 用途 |
| --- | --- |
| `data/` | 275 行论文计数、H 分片、S 报告、精确批次关系及注入线路定义 |
| `src/h/`、`src/s/`、`src/s_lowp/` | H/S 线路生成与两个冻结 S 采样协议 |
| `src/surgery/` | ZZ/XX/XZ 联合测量线路及译码输入编译 |
| `src/strict_search/` | 完整故障签名、精确约束搜索和分区枚举 |
| `tools/` | 数据核验、故障重放、独立采样与距离搜索入口 |
| `third_party/midout` | 固定提交的 [midout_3-CX](https://github.com/QEC-FPGA/midout_3-CX) 构造依赖 |

完整命令见 [复现说明](docs/REPRODUCIBILITY.md)，附件用途与数据限制见 [数据说明](docs/DATA.md)。[AGENTS.md](AGENTS.md) 是自动化复现的简短入口；所有读者均可按同一套命令操作。

含噪数值数据涵盖 H、S 和联合测量组件，不代表完整 CNOT 或魔态蒸馏装置的性能。距离结论限于指定线路、边界和故障模型。原创贡献按 MIT 提供，第三方边界及必要归属见 [LICENSE.md](LICENSE.md)。
