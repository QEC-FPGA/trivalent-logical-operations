# 复现说明

以下命令均从论文仓库根目录执行。冻结的 S 采样入口要求 Python 3.12；已验证环境为 Python 3.12.14、Stim 1.16.0、PyMatching 2.3.1、NumPy 2.4.4。绘图依赖单独列出，完整依赖清单还包含搜索所需的 OR-Tools。

## 获取固定版本依赖

```bash
git clone --branch master --recurse-submodules https://github.com/QEC-FPGA/trivalent-logical-operations.git
cd trivalent-logical-operations
```

已有克隆可执行 `git submodule update --init --recursive`。`third_party/midout`
固定到 `midout_3-CX` 的一个确切提交；不要用 `--remote` 更新到未经论文验证的版本。
只从计数重绘不需要加载 midout，晶格手术线路生成需要该依赖。

从 Zenodo 代码 ZIP 解压的目录没有 Git 元数据。若需生成晶格手术线路，
可在解压后的仓库根目录获取同一固定依赖：

```bash
git clone --no-checkout https://github.com/QEC-FPGA/midout_3-CX.git third_party/midout
git -C third_party/midout checkout --detach ef3bb5ef1f70a40280c60370d7d3071cfdfdbc06
```

代码 ZIP 的 `MANIFEST.json` 同时记录该依赖 URL 和提交号；子模块源码
不包含在论文仓库的 ZIP 中。


## 从已有计数重绘

```bash
python -m pip install -r requirements-plot.txt
python tools/check_results.py
python reproduce_figures.py
```

输出位于 `reproduced_figures/`，包括 H、S 主线、等待时间比较和晶格手术图。绘图不需要大型数据附件。

错误率按整条测试线路的失败数除以样本数，不除以轮数。绝对率区间采用 95% Wilson；比值使用两组 97.5% Clopper–Pearson 区间组合。不同图可能复用同一批采样，不可把各图计数相加作为独立样本总数。

## 读取和核验冻结产物

按 [DATA.md](DATA.md) 选取附件；下载位置可自行指定。解压后成员位于 `bundles/`。每个附件都有成员级 SHA256 清单；压缩文件的校验和见 `release-assets.json`。

```bash
python tools/check_results.py --assets-dir /path/to/downloads --members
python tools/check_s_reports.py /path/to/downloads/fig10-11-records.tar.gz
python tools/check_surgery_reports.py /path/to/downloads/fig12-confirmation.tar.gz
tar -xzf /path/to/downloads/distance-evidence.tar.gz
python tools/check_distance_results.py
```

`check_distance_results.py` 核对归档搜索的分区覆盖、模型和证据哈希，**不重新完成穷举搜索**。可选的大型 pair-index 附件用于复用计算缓存，不是重绘或检查已保存结果的必需下载。

## 线路生成和见证检查

```bash
python -m pip install -r requirements.txt
python src/h/generate_one.py --kind h_flow --distance 5 --basis X -o out/H_d5_X.stim
python src/s/run.py check
python src/s/run.py list
python src/s_lowp/run.py check
python src/surgery/generate.py --d 5 --p 0.001 --measurement ZZ --out out/surgery_d5_ZZ
python tools/replay_witnesses.py
python tools/check_d3_certificate.py
```

H 的示例命令生成理想线路；用于论文计数的 SI1000 含噪线路与 DEM 以图 9 附件中的冻结文件为准。S 的 `run.py` 提供 `prepare/pilot/budget/confirm/report`；各参数组在对应 `protocol.json` 中定义。低错误率材料也提供冻结协议和生成代码，其精确输入文件覆盖范围见 [DATA.md](DATA.md)。不要用新计划覆盖原报告中的计划身份。冻结源码还保留历史 `mechanism` 子命令；它依赖未提供的 `mechanism_seeds/`，不属于本包支持的复现入口。故障见证使用上面的独立检查工具。

晶格手术生成器使用附带的 d=5 参考线路和修改版 midout，许可边界见 [LICENSE.md](../LICENSE.md)。生成线路不自动等于通过噪声模型分解或获得距离证明。

## 重新采样

选择附件中的一对线路/DEM，通过论文表或附件的 `stats.csv` 对照其哈希。以下为新增独立采样示例；不替换归档计数。

```bash
python tools/sample_frozen.py --circuit path/to/circuit.stim --dem path/to/matching.dem --shots 100000 --seed 12345 --out out/new_sample.json
```

程序使用相关 PyMatching，记录输入哈希、随机种子、批大小、错误数及多可观测量残差直方图。随机算法、线程、批大小和依赖版本会影响逐次样本；复核既有精确计数应聚合原始分片。完整规模的低错误率采样计算量很大。

## 重新搜索

```bash
python -m pip install -r requirements.txt
python tools/search_distance.py --case d5_m2-3_W3_XY --cap 4 --seconds 900 --workers 2 --out out/search_d5_XY
```

此入口从已解压的模型运行新的完整签名约束搜索，输出到新目录。`UNKNOWN` 表示预算内未完成，不构成下界。四个 d=5、7 条件的归档结果和索引可直接复用；原始分区枚举、完整签名模型、物理故障实现和算法代码一并提供。新的 CP-SAT 运行不保证在给定时间内重现归档使用的分区算法耗时。

W3 的权重 11、12、14 见证只证明相应线路的距离上界。严格距离和附录 C 的局部移动诊断有各自的线路与边界，不能跨线路混用故障指令偏移。

## 冻结源码与可选缓存

两套 S 协议及距离搜索保留各自原有的源码布局，以维持生成行为和采样记录中的代码摘要。请从各自入口运行，不移动其内部依赖文件。

完整故障证据解压后位于 `bundles/distance-evidence/`。若需要复用单独下载的配对索引，仅解压数组，避免覆盖核心成员清单：

```bash
tar -xzf /path/to/downloads/strict-distance-pair-indexes.tar.gz --strip-components=2 --wildcards -C bundles/distance-evidence "*/results/*/pair_index.npy"
```

计数检查、重绘与证据重放通常是桌面级任务；完整低错误率采样及距离穷举需要远大于这些检查的计算预算。
