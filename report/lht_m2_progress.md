# lht：M2 开发与验收进展

日期：2026-10-22
关联任务：[#19 M2 实现](https://github.com/guts-yang/BeautyCam/issues/19)、[#20 单测与直方图](https://github.com/guts-yang/BeautyCam/issues/20)
状态：冻结配置内的自适应 gamma / linear 已实现，40 张隔离评测与直方图已完成；CLAHE 配置扩展待共同评审，#19 尚未完整完成。

## 本轮结果

| 项目 | 结果 | 要求/范围 |
| --- | --- | --- |
| 固定集有效样本 | 40/40 | 无剔除 |
| 真实欠曝样本 | 1 张，ID 6277 | 输入皮肤 GT 平均 Y < 90 |
| 该图皮肤平均 Y | 88.535792 → 106.190245 | 同一 GT 区域 |
| 相对提亮 | 19.9405% | ≥ 15%，达标 |
| 处理后过曝比例 | 0.0000% | Y > 250 比例 ≤ 1%，达标 |
| 非集成回归 | 183 passed，1 deselected | 新增 M2 40 项测试 |

全部 40 张的平均相对提亮为 3.1153%，该数混合了欠曝与正常曝光图，不能用来替代欠曝子集结果。真实欠曝子集只有 1 张，因此上述达标不能证明广泛欠曝场景的泛化效果。

样本 5118 的输入皮肤区域已经有 5.6690% 像素 Y>250，输出保持相同。本模块在输入超过配置过曝预算时安全跳过，不宣称已修复原有过曝。全部 40 张中，输出过曝比例超 1% 的也只有该图；未删除它或用全图宏平均掩盖它。

## 实现范围与方法

本轮将 exposure.py 从返回副本的占位函数替换为实际曝光处理，保持冻结签名：

```python
process(image_bgr, skin_mask, cfg) -> np.ndarray
```

- 读取 BGR uint8 图像、同尺寸 uint8 0/255 皮肤掩码和既有 YAML；返回新数组，不修改输入图、掩码或配置。
- 在 YCrCb 的 Y 上计算皮肤区域均值。目标均值为 min(mean + gain × (target_y − mean), mean × max_gain)。
- gamma 模式按输入与目标均值计算自适应指数，再乘现有 gamma 参数；gamma=1 表示无额外修正，指数上限为 1，使该模块只提亮。非均匀图像的真实均值变化由评测测量，不把曲线公式当作实际达成值。
- linear 模式按目标/输入均值计算线性倍率。
- 两种曲线均限制单像素提升倍率，纯黑保留为黑；超过 target_y、gain=0 或 max_gain=1 时逐字节返回副本。
- 只回写掩码内亮度发生变化的像素；掩码外严格不变，避免未处理处的色彩往返取整漂移。高饱和颜色在转回 BGR 时仍可能发生色域裁剪。
- 在实际输出 BGR 重算 Y 后检查 overexposure_y 与比例预算；如超预算，以二分搜索减弱整次提升，数值求解精度为 1/256 强度。不依赖中间浮点 Y 代替最终过曝检查。
- 空掩码与 None 掩码均安全跳过。输入已超预算时原样返回；这是保护策略，不能保证对任何任意输入都修复既有过曝。

YAML 仅补充已有键的含义注释，未改参数值或 schema：target_y=150、gain=0.3、max_gain=1.5、overexposure_y=250、overexposure_ratio_limit=0.01、method=gamma、gamma=1.0。模块没有文件读写、全局可变状态、其他算法模块导入或新增依赖。

## CLAHE 尚待完成的原因与具体提案

Issue #19 要求 CLAHE + 自适应伽马，但冻结 exposure.yaml 没有 CLAHE 的 clipLimit 和 tileGridSize。

[arch.md 7.1](../docs/baseline/arch.md) 规定“可调阈值、强度、核大小、色温等不得硬编码”，且“改 schema、接口或跨模块语义时须走 change-request”。因此没有把这两个调参值藏进源码或悄悄新增生效配置。

已准备 [CLAHE 配置变更建议](lht_m2_clahe_schema_proposal.md)，明确建议新增的三个键、候选初值、兼容行为、写入边界及后续测试；待确认后接入。该文档仍是本地提案，没有冒称已获得共同评审或已发布远程 Issue。

summary.json 明确记录 implementation_scope 和 full_issue_19_acceptance=false。当前可说“gamma/linear 分支与该欠曝样本的定量项完成”，不能说“#19 全部完成”。提交使用 Refs，不使用 Closes。

## 评测协议与材料

1. 完整使用 assets/evaluation_manifest.csv 的 40 张；清单 SHA-256 为 d9df0cff2d661b0e7f94a763aa932fe6779068a461f904c634d4c79f326cf5a9。
2. 使用 skin GT，512×512 最近邻对齐至原图 1024×1024；不使用 M1 预测掩码代替 GT。
3. 输入皮肤 GT 平均 Y<90 才判为欠曝，不按未经确认的 scene 占位值筛选。本轮 6277 的 scene 是 blemish，5118 的 scene 是 lowlight，仍严格按实际均值筛选。
4. 输入与输出均使用 BGR→YCrCb 的 8-bit Y；相对提亮 = (mean_after−mean_before)/max(mean_before,1)，过曝比例 = count(Y_after>250)/GT 像素数。
5. 均值、过曝比例及 256 个亮度桶计数使用同一张图、同一皮肤 GT。直方图像素计数可由 CSV 复核，绘图坐标线性、使用相同纵轴。
6. 报告全部样本和欠曝子集；空 GT 标为无效，无欠曝样本时不得以空集合逻辑宣称通过。本工具要求每张真实欠曝图均满足两个门槛，并记录宏平均/中位数。
7. 没有合成暗化图片来充当真实欠曝样本，也未改变固定清单或评分阈值。
8. 评测检查输出 shape/dtype、新数组、图像/掩码不变性及掩码外逐字节一致。

最终材料目录：results/m2_validation_20261022_r2/。

- summary.json：有效参数、评测范围、宏平均、中位数、样本数、清单/算法哈希、运行环境及 CLAHE 待完成状态。
- samples.csv：全部 40 张的均值、相对提亮、前后过曝比例与欠曝标记。
- histograms.csv：每图 256 个桶，共 10,240 行，含前后原始像素计数。
- outputs/：40 张原尺寸处理结果。
- figures/：40 张“原图/输出/同一皮肤 GT 前后直方图”图，1200×950。
- README.md：材料与验收范围说明。

首次 results/m2_validation_20261022/ 与修正图表文字间距后的 r2 独立复跑使用相同算法、配置与清单，数值结果一致。原图和衍生图均仅保留本地被忽略目录。

## 本地复现

在 BeautyCam 仓库根目录执行：

```powershell
.\.venv\Scripts\pytest.exe -m "not integration" -q
.\.venv\Scripts\python.exe tools/evaluate_m2.py --data-root data/input/subset
```

工具默认创建新的 results/m2_<run>/ 目录，禁止覆盖已有输出。CLI 退出码仅表示欠曝定量项，不能当作整个 #19 完成的凭据。

测试覆盖实际目标提亮、两种曲线、各参数生效、过曝预算在实际 BGR 上的约束、原有过曝跳过、空掩码、只读/非连续数组、极小图像、非法输入/配置、统计边界 Y=90/250/251、GT 最近邻对齐、直方图计数及无欠曝/空 GT 不得虚假通过。全部测试使用内存合成数据和临时 fixture，不依赖真实数据集或可选绘图库。

本地环境为 Python 3.12.8、OpenCV 4.10.0、NumPy 1.26.4；Python 3.10 与远程 CI 待后续验证。未修改 A 的 core/、io/、pipeline.yaml、requirements.txt、.github/ 或 C 的源码。

## B 板块余项

- M1：IoU 0.446276，Recall 0.961867；IoU 未达标，继续改进。
- M2：当前分支定量项达标；CLAHE 接入待配置评审，随后需要同集对照评测。
- M3：定量达标；两名非实现者的边缘与痘印 ROI 检查待完成。
- M4：定量达标；两名非实现者的 10 张高反差样例检查待完成。
- 远程 PR、CI 和统一联调尚未完成；不把本地提交等同于已合入 main。
