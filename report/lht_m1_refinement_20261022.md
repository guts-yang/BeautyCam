# lht：M1 误检归因与空间分割试验

日期：2026-10-22
关联任务：[#17](https://github.com/guts-yang/BeautyCam/issues/17)、[#18](https://github.com/guts-yang/BeautyCam/issues/18)
状态：完成本地误检诊断，空间候选仅供开发验证；正式 M1 算法和生效配置未替换，未完整验收。

## 先看结论

当前正式 M1 的 40 张宏平均 IoU 仍为 0.446276，Recall 为 0.961867，Precision 为 0.462241。主要问题是误检多，不能仅靠追求高召回解决。

本轮不是再次盲目收紧颜色阈值，而是新增可复现的逐标签归因和连通域能力诊断，并研究不使用预训练模型的 GrabCut 空间分割。额外参数全部放在显式实验 JSON 内，没有悄悄增加 config/skin_detect.yaml 的 schema。

## 误检来自哪里

同一固定清单共 40 张，原尺寸预测，标签/GT 最近邻对齐。下表按全部误检像素汇总，属于像素加权构成比例，不是逐图宏平均：

| GT 类别 | 误检像素数 | 占全部误检 |
| --- | ---: | ---: |
| hair 头发 | 8,352,563 | 57.1294% |
| background 背景 | 2,177,857 | 14.8960% |
| neck 颈部 | 1,707,574 | 11.6794% |
| nose 鼻子 | 831,028 | 5.6840% |
| l_ear 左耳 | 218,381 | 1.4937% |

全部误检为 14,620,422 像素。头发与背景合计约 72.03%，因此空间与外观分离比简单删除孤立噪点更值得继续。

40 张均验证 skin_masks 与最近邻对齐后的合并标签 == 1 完全一致。鼻子、颈部和耳朵在日常语义中也属于皮肤，但本项目已冻结的 skin GT 将它们独立标注；本轮不更改 GT、不把这些类别加进正类来抬高成绩。

标签顺序与本项目 prepare_data.py 一致，来源参考：[CelebAMask-HQ 官方合并脚本](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py)、[官方数据说明](https://github.com/switchablenorms/CelebAMask-HQ)。没有使用该仓库的训练代码或预训练权重。

## 为什么只筛选连通域不够

以当前正式预测的每个完整 8 连通域为单位，仅允许保留或删除整块区域，即使用 GT 帮忙作出最有利选择，40 张最大可达 IoU 的宏平均也只有 **0.459815**。

这是受限方法族的离线上界：它只说明“在当前固定预测上挑整块连通域”不能达到 0.75，不代表所有传统图像算法的上限，也不是可部署算法成绩。

计算方法：对每个连通域求 TP_i、FP_i，整图皮肤 GT 面积为 G。选中集合的 IoU = sum(TP_i) / (G + sum(FP_i))。按 TP_i/FP_i 降序排序，再比较所有前缀即可得到最优值；FP=0 的真阳性连通域优先。单测另对所有连通域子集穷举，独立验证此优化结果。GT 只传入诊断函数，绝不传入 detect_skin。

## 传统空间分割候选

步骤：

1. 将长边缩到不超过显式 working_size，本轮为 512；保留纵横比。
2. 按原 M1 色度阈值生成候选颜色区域。
3. 用显式归一化中心与椭圆半径定义可能人脸范围和确定前景种子；候选只接收图像、颜色配置和实验选项。
4. 以颜色候选与种子初始化 GrabCut 的前景/背景状态，在每张输入图上估计 GMM 并作图割，不使用任何预训练模型。
5. 将图割前景与原颜色候选相交，最近邻放大到原图尺寸后计分。
6. 无足够种子时保留该样本、记为全零预测及明确状态，不能跳过困难样本。

首轮 4 组探索中，最好候选的 IoU=0.700969、Recall=0.924484，已明显改善，但仍未达到 IoU≥0.75。后续 8 组候选的参数见 m1_spatial_candidates_20261022.json。两轮均是同一固定集上的开发试验，没有独立留出验证，不作为泛化成绩。

后续 8 组的完整结果如下，均为 40 个有效样本，均无种子失败：

| 候选 | IoU 宏平均 | Recall 宏平均 |
| --- | ---: | ---: |
| pilot_best | 0.700969 | 0.924484 |
| shorter_roi | 0.653724 | 0.841978 |
| lower_center | 0.697031 | 0.909553 |
| narrower_roi | 0.677583 | 0.871398 |
| shorter_seed | 0.699980 | 0.921517 |
| higher_center | 0.644997 | 0.835544 |
| tight_seed_color | 0.698952 | 0.918489 |
| short_roi_tight_seed | 0.653296 | 0.838240 |

pilot_best 与首轮同参数结果一致。最佳候选的 Precision 为 0.745357，全部误检从 14,620,422 减至 3,461,076 像素。其剩余误检中，头发占 37.22%、鼻子占 24.00%，左右眉分别约占 4.97%。这说明消除大块头发/背景后，鼻子和五官等与肤色相近的独立 GT 类别成为更突出的难点。

在最佳候选上再作整连通域最优选择，上界也仅为 0.702131，因此后续不能指望只增加“保留最大块”补齐 0.75 的差距。

OpenCV 算法接口参考：[GrabCut 官方文档](https://docs.opencv.org/4.10.0/d8/d83/tutorial_py_grabcut.html)。当前实验 CLI 在每次候选前重置 OpenCV RNG；[OpenCV 4.10 实现](https://github.com/opencv/opencv/blob/4.10.0/modules/imgproc/src/grabcut.cpp)的初始化使用 k-means。正式接入前需要解决可复现初始化与模块不维护全局状态之间的约束，不能把实验代码原样搬进正式模块。

归一化椭圆是对居中肖像的空间先验，不能直接声称支持任意构图或多人照片。正式生效所需的空间参数尚未纳入冻结 schema。

## 复现

在 BeautyCam 仓库根目录执行：

```powershell
.\.venv\Scripts\pytest.exe -m "not integration" -q
.\.venv\Scripts\python.exe tools/analyze_m1_errors.py
.\.venv\Scripts\python.exe tools/experiment_m1_spatial.py --candidates report/m1_spatial_candidates_20261022.json
# 上一条结果的实际目录替换 <run>；审计保存的候选，不重新生成预测
.\.venv\Scripts\python.exe tools/analyze_m1_errors.py --predictions results/<run>/predictions/pilot_best
```

两个工具默认创建全新结果目录，禁止覆盖已有输出。缺少固定清单中的任何文件时失败，不静默缩小数据集；GT 不进入候选预测函数。

误检材料：results/m1_error_audit_20261022/。

- summary.json：正式 M1 指标、受限连通域上界、误检来源、清单/算法哈希及环境。
- samples.csv：40 张 TP/FP/FN、IoU、Recall、Precision、连通域数和离线上界。
- label_errors.csv：每图 19 类计数，共 760 行。
- fp_sources.csv：汇总误检构成。
- figures/：40 张原图、GT、正式预测及 TP/FP/FN 错误面板。

空间试验材料：results/m1_spatial_candidates_20261022/。逐图分数、所有候选汇总和原尺寸预测均保留在本地；相关人像与派生图不提交到 Git。

- candidate_scores.csv：8 组候选的全部宏平均。
- samples.csv：320 条候选/样本记录，不丢弃失败或低分样本。
- predictions/：320 张原尺寸二值预测。
- summary.json：显式参数、实验源码哈希、环境与计分边界。

最佳候选的独立错误审计在 results/m1_spatial_best_audit_20261022/，含 40 张错误面板、760 行逐类计数和候选数组哈希；该目录标为 saved candidate masks，算法哈希留空，不冒称由当前正式 M1 输出。

全部使用同一 assets/evaluation_manifest.csv，SHA-256 为 d9df0cff2d661b0e7f94a763aa932fe6779068a461f904c634d4c79f326cf5a9。scene 仍是未经人工确认的占位标签，不据此宣称真实场景分层。

## 回归与写入边界

194 passed，1 deselected。新增 11 项测试，包含连通域上界与穷举一致、8 连通语义、空 GT、标签一致性、最近邻对齐、缺文件不缩集、空间预测无 GT 接口、只读输入不变性、颜色背景排除、确定种子失败状态和候选参数校验。

本地验证 Python 3.12.8、OpenCV 4.10.0、NumPy 1.26.4；Python 3.10 和远程 CI 尚未在本轮验证。

未修改正式 M1 源码、config/skin_detect.yaml、A/C 源码、公共接口、依赖或评分阈值。#17/#18 仍使用 Refs，不使用 Closes。后续用户已授权 B 自主扩展并接入 CLAHE，最新状态见 [M2 报告](lht_m2_progress.md)。
