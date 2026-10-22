# lht：M2 开发与验收进展

日期：2026-10-22
关联任务：[#19 M2 实现](https://github.com/guts-yang/BeautyCam/issues/19)、[#20 单测与直方图](https://github.com/guts-yang/BeautyCam/issues/20)
状态：CLAHE + 自适应 gamma、过曝保护、单测和固定集隔离评测已完成；远程 PR、CI 和统一联调尚未完成。

## 本轮结果

用户明确确认 B 模块与配置由本人负责，并授权直接修改。因此本轮在 B 自有 exposure.yaml 中增加 CLAHE 参数并完成接入，不再等待配置确认。该授权不等于已完成远程评审，未改写公共架构文档或其他成员文件。

同一固定 40 张的开启/关闭对照：

| 项目 | CLAHE 关闭 | CLAHE 开启（最终 YAML） | 要求 |
| --- | ---: | ---: | --- |
| 有效样本 | 40/40 | 40/40 | 无剔除 |
| 真实欠曝样本 | 1 张，6277 | 同一张 | 输入皮肤 GT 平均 Y < 90 |
| 该图平均 Y | 88.535792 → 106.190245 | 88.535792 → 106.689032 | 同一 GT 区域 |
| 相对提亮 | 19.9405% | 20.5038% | ≥ 15%，达标 |
| 该图输出 Y>250 比例 | 0.0000% | 0.0000% | ≤ 1%，达标 |
| 本地 CLAHE+gamma 要求 | 本次关闭，不满足该算法组合 | 满足 | Issue #19 |

真实欠曝子集仅 1 张，不能将本轮门槛达标称为广泛欠曝场景的泛化验证。全部 40 张开启 CLAHE 后的平均相对提亮为 3.4825%，该统计混合正常曝光图，不能替代欠曝子集成绩。

样本 5118 输入皮肤区域已经有 5.6690% 像素 Y>250，输出保持相同。全部 40 张中输出过曝比例超 1% 的也只有该图；未删去它或宣称已修复既有过曝。

## 实现与配置

冻结的 process(image_bgr, skin_mask, cfg) 签名和新建 BGR uint8 返回值不变。

1. 在 YCrCb 的 Y 上做 CLAHE，按既有 gain 混合局部对比度结果；低于原亮度的候选值不用于暗化。亮度全平坦时跳过局部均衡，纯黑保持为黑。
2. 目标均值为 min(mean + gain × (target_y − mean), mean × max_gain)，由输入皮肤区域统计得到。
3. gamma 模式根据 CLAHE 混合后的亮度均值与目标均值计算自适应指数，再乘既有 gamma 参数；指数上限为 1。linear 模式保留为可选曲线。
4. 用 max_gain 限制单像素亮度提升，只回写掩码内亮度实际变化的像素。掩码外保持逐字节不变。
5. 在实际输出 BGR 重算 Y 后检查过曝预算；超预算时二分减弱整次提升，强度求解精度为 1/256。输入已超预算时安全跳过，不能据此声称修复既有过曝。
6. 空掩码、None 掩码、gain=0、max_gain=1、皮肤均值已达到 target_y 时返回副本，不修改输入图、掩码或配置。
7. 8-bit 亮度只有 256 个桶，clip_limit≥256 时按等价的不裁剪上界处理，避免数值溢出。网格数上限按当前图像像素尺寸收敛，支持极小图。
8. 高饱和像素转回 BGR 时仍可能发生色域裁剪；所有正式统计基于最终 BGR，未用中间浮点值替代结果。

最终配置：

```yaml
target_y: 150
gain: 0.3
max_gain: 1.5
overexposure_y: 250
overexposure_ratio_limit: 0.01
method: gamma
gamma: 1.0
clahe:
  enabled: true
  clip_limit: 2.0
  tile_grid_size: [8, 8]
```

clahe 缺失或 enabled=false 时兼容旧曲线。真实 40 张关闭输出与前一版 results/m2_validation_20261022_r2/ 的输出逐像素一致；开启结果逐图独立重算后也全部一致。

CLAHE 参数定义参考：[OpenCV 4.10 官方文档](https://docs.opencv.org/4.10.0/d6/db6/classcv_1_1CLAHE.html)。网格表示横向/纵向分块数量，不能误写成单格像素尺寸。

## 评测协议

- 完整使用 assets/evaluation_manifest.csv 的同一 40 张，SHA-256 为 d9df0cff2d661b0e7f94a763aa932fe6779068a461f904c634d4c79f326cf5a9。
- 使用 skin GT，512×512 最近邻对齐至 1024×1024 原图；不使用 M1 预测误差参与隔离计分。
- 实际输入皮肤 GT 平均 Y<90 才属于欠曝，不按未经确认的 scene 占位值筛选。
- 相对提亮 = (mean_after−mean_before)/max(mean_before,1)；过曝比例 = count(Y_after>250)/GT 像素数。
- 均值、过曝比例及 256 桶计数始终使用同一图、同一皮肤 GT。
- 报告全部样本及欠曝子集；空 GT 无效、无欠曝样本时不能虚假通过。所有真实欠曝图均须满足两个门槛，并保留宏平均/中位数。
- 不合成暗图充当真实欠曝成绩，不更改 GT、固定清单或评分标准。

## 复现与材料

在仓库根目录执行：

```powershell
.\.venv\Scripts\pytest.exe -m "not integration" -q
# 最终 YAML：CLAHE 开启
.\.venv\Scripts\python.exe tools/evaluate_m2.py --data-root data/input/subset
# 只覆盖本次开关作消融，不回写 YAML
.\.venv\Scripts\python.exe tools/evaluate_m2.py --no-clahe
```

结果默认写入新目录，禁止覆盖已有证据。CLI 退出码表示本次定量项；summary.json 的 local_acceptance 同时检查 CLAHE+gamma 与定量门槛，full_issue_19_acceptance 仍为 false 并明确说明远程 PR/CI 待完成。

本轮目录：

- results/m2_clahe_off_20261022/：CLAHE 关闭的同集消融。
- results/m2_clahe_on_20261022/：最终开启配置，作为本轮正式本地材料。

两目录各有 summary.json、40 行 samples.csv、10,240 行 histograms.csv、40 张原尺寸 outputs/ 与 40 张前后直方图 figures/。开启目录额外有 compatibility_check.json，记录 40 张旧版兼容、40 张开启复算及直方图区域计数一致。图表已目视检查，人物与衍生图只保留本地被忽略目录。

旧的 results/m2_validation_20261022*/ 保留首轮 gamma-only 基线，不再作为当前功能范围说明。

## 测试与边界

最新非集成回归为 **214 passed，1 deselected**，其中 M2 算法/评测测试共 60 项。验证实际局部对比度变化、clip/grid 参数作用、关闭开关兼容性、只读/非连续输入、嵌套配置不变性、掩码外不变、CLAHE 开启/关闭的实际 BGR 过曝预算、极小图/超大网格/极端数值，以及统计边界与不可评估规则。

本地环境 Python 3.12.8、OpenCV 4.10.0、NumPy 1.26.4；Python 3.10 和远程 CI 尚未验证。

本轮只修改 B 的源码、模块 YAML、对应测试、离线工具与报告。未修改 A 的 core/、io/、config/pipeline.yaml、requirements.txt、.github/ 或 C 的文件；没有新增依赖、文件读写型算法或模块间导入。

M2 的 CLAHE 接入已完成，不再列为等待配置确认。后续 [M1 正式实现](lht_m1_final.md) 也已通过本地定量门槛。余项为速度优化、M3/M4 的两名非实现者人工检查，以及远程 PR/CI 与统一联调；远程状态以当前 PR 为准。
