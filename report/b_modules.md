# B 板块实现与验证

2026-11-17 整理。负责人 qixing1227，开发分支 lht，对应 Issue #17–#24。[PR #52](https://github.com/guts-yang/BeautyCam/pull/52)

## 实现范围

B 负责 M1 肤色检测、M2 曝光改善、M3 保边平滑和 M4 频域增强，交付内容包括模块实现、独立 YAML 参数、算法与评测测试、正式评测脚本及报告图表。主流程执行顺序为 M5 → M1 → M2 → M3 → M4 → M6，M7 负责评价。平台入口由 A 维护，M5–M7 由 C 维护。

| 模块 | 输入与输出 | 实现文件 | 参数文件 |
| --- | --- | --- | --- |
| M1 | `detect_skin(image_bgr, cfg)` 返回同尺寸二维 `uint8` 掩码，值为 0/255 | [skin_detect.py](../src/beauty/modules/skin_detect.py) | [skin_detect.yaml](../config/skin_detect.yaml) |
| M2 | `process(image_bgr, skin_mask, cfg)` 返回同尺寸 BGR `uint8` 图像 | [exposure.py](../src/beauty/modules/exposure.py) | [exposure.yaml](../config/exposure.yaml) |
| M3 | 同上 | [smooth.py](../src/beauty/modules/smooth.py) | [smooth.yaml](../config/smooth.yaml) |
| M4 | 同上，另提供只读频率诊断接口 | [freq_enhance.py](../src/beauty/modules/freq_enhance.py) | [freq_enhance.yaml](../config/freq_enhance.yaml) |

处理函数接收内存数组，保持输入图、掩码和配置只读。M2–M4 只在传入掩码内合成结果，空掩码返回图像副本。跨模块调用由主流程负责。

## M1 肤色检测

先在 YCrCb 中用 Cr 133–173、Cb 77–127 生成颜色候选，再以传统 Haar 级联检测人脸和眼睛。默认启用 `face_locator`，定位可靠时，根据实际人脸框设置图割种子和搜索域，随人脸位置与大小变化。灰度和 BGR 分支使用确定性 GMM 初始化、GrabCut 分割与投票，随后恢复主体附近的连续皮肤，并沿图像中的发际和下颌边缘限制选区。

五官保护结合眼部检测框、眉毛暗结构、局部唇色及已分出的嘴部空洞。鼻子保留在美颜作用域中。最后执行形态学开闭运算和小连通域过滤。可靠定位缺失时回退至颜色候选。

- `spatial.enabled: false` 为默认值。该项控制固定整图构图模式，只有显式开启时才采用整图 `center/roi_radius/seed_radius`。
- `face_locator.enabled: true` 独立控制 Haar 路径。`face_locator.segmentation` 中的 `center_offset/roi_scale/seed_scale` 相对实际人脸框定义。
- `features.dark/lips/eyes/brows/mouth_holes` 集中管理五官保护阈值与几何比例，并检查类型、数值范围、核尺寸及上下界关系。
- 旧配置省略可选项时沿用代码回退值。正式运行和评测加载仓库 YAML，以取得当前完整参数。

Haar 是经本项目授权采用的传统预训练分类器。三个模型由已锁定的 `opencv-python-headless==4.10.0.84` 随包提供，YAML 仅记录资源名。通过 `importlib.resources.files("cv2.data")` 读取依赖包内的 XML，再用 OpenCV 内存接口加载，支持 Windows 中文安装路径。模型加载只访问依赖包资源，用户图像通过数组传入。原始许可另见 [Haar 资源许可](lht_haar_models_LICENSE.txt)。

GMM 的 `1×65` 数组保存权重、均值和协方差，逆协方差由 OpenCV 构造器另外缓存。测试用各向异性高斯验证 `GC_EVAL_FREEZE_MODEL` 的读取方式及模型不变性。[OpenCV 4.10 源码](https://github.com/opencv/opencv/blob/4.10.0/modules/imgproc/src/grabcut.cpp#L85-L104)

## M2 曝光改善

在皮肤区域估计平滑亮度基底，执行 CLAHE 与自适应 Gamma，再把平滑曝光增量加回原图亮度，以保留细纹。指数形式为 `255 × (Y/255)^exponent`，指数小于 1 时提亮。目标均值由当前均值、`target_y`、`gain` 和 `max_gain` 共同限制。

外轮廓与眼唇孔洞分别羽化，减少掩码边缘的亮度接缝。生成候选 BGR 后重新计算亮度并检查过曝预算，必要时二分减弱提升强度。输入已经超过预算时跳过提升。当前主要参数为 `target_y=150`、`gain=0.3`、`max_gain=1.5`，CLAHE 网格为 8×8，`clip_limit=2.0`。

## M3 保边平滑

使用双边滤波平滑皮肤，并保留颜色差异较大的边缘。局部中值与原亮度的暗残差经 Otsu 阈值产生候选，只修补皮肤内部的小型暗结构。按掩码内到边缘的距离控制融合权重，减轻五官边界的光晕。

当前参数为 `strength=0.5`、`bilateral_d=9`、`sigma_color=75`、`sigma_space=75`，边界采用 `BORDER_REFLECT_101`。暗点也可能是雀斑、阴影或正常纹理，真实痘印效果需要人工确认。

## M4 频域增强与诊断

亮度经过偶反射延拓后，使用 NumPy `rfft2/irfft2` 和高斯传递函数分离低频与有符号高频，再按增益合成并在掩码边界渐变融合。低通函数为 `H(fx, fy) = exp(-2π²σ²(fx²+fy²))`，高频为原亮度减低频。

当前 `lowpass_sigma=3.0`、`lowpass_gain=1.0`、`highpass_gain=1.1`，用于轻量细节回注。`get_frequency_diagnostics(image_bgr, cfg)` 返回 `low_frequency`、`high_frequency` 两张 BGR `uint8` 可视化。高频图以 128 表示零残差，诊断显示的裁剪不参与主处理。测试和评测检查调用诊断前后的主输出相同。

## 验证协议与结果

固定使用 [evaluation_manifest.csv](../assets/evaluation_manifest.csv) 的完整 40 张。清单 SHA256 为 `d9df0cff2d661b0e7f94a763aa932fe6779068a461f904c634d4c79f326cf5a9`。原图为 1024×1024，标签为 512×512，评分时使用最近邻对齐。

M1 按原 `skin_masks`（语义类 1）计算逐图 IoU/Recall 后取宏平均。鼻子是语义类 2，因此另报类 1+2 辅助指标。M2–M4 隔离评测使用同一 GT 皮肤区域，组合评测则使用 M1 的实际预测掩码。GT 不传入实际美颜算法作检测依据。

下表来自当前算法与有效配置对应的最近一次正式测量。本轮整理文档和工具，四个算法文件及参数值保持一致，沿用这些结果。

| 项目 | 测量结果 | 定量要求 |
| --- | ---: | --- |
| M1 原协议 IoU 宏平均 | 0.752956 | ≥ 0.75 |
| M1 Recall 宏平均 | 0.903888 | ≥ 0.80 |
| M1 后处理 IoU 变化 | -0.000170 | ≥ -0.02 |
| M2 真实欠曝提亮 | 20.7970% | ≥ 15% |
| M2 欠曝图输出 `Y>250` 比例 | 0 | ≤ 1% |
| M3 局部方差下降率宏平均 | 25.9499% | 20%–40% |
| M3 五官 EPI | 0.998719 | ≥ 0.80 |
| M4 高频保留率 | 1.068823 | 0.70–1.30 |
| M4 五官 EPI | 0.999453 | ≥ 0.80 |

M2 将输入 GT 区平均 Y<90 的图定义为真实欠曝，本批只有 6277 一张，评分前后使用同一 GT 区域。M3 方差采用 Y 通道 7×7 窗口，在 11×11 椭圆核腐蚀后的皮肤区域取中心。M4 高频保留率为皮肤内部高频 RMS 的处理后/处理前比值。五官 EPI 使用 Sobel 梯度的逐像素较小值之和除以较大值之和，空区域和零梯度单独记录无效原因。

M1 含鼻子辅助 IoU 为 0.819332，鼻子覆盖宏平均为 99.59%。正式与实际组合的 40 张预测掩码逐像素一致。本轮本地 340 项非集成测试通过，1 项未选，远程结果见本 PR 最新 CI。

| 本地证据 | 目录（仓库根目录下） | 主要材料 |
| --- | --- | --- |
| M1 正式评测 | `results/m1_review_fixes_20261116/` | 逐图 IoU/Recall、开关对照、预测/GT/叠加图 |
| M2 隔离评测 | `results/m2_illumination_inner_blend_20261030/` | 提亮/过曝统计、原尺寸输出、直方图 |
| M3 隔离评测 | `results/m3_validation_20261017/` | 方差/EPI、三联图、ROI 候选与人工表 |
| M4 隔离评测 | `results/m4_detail_20261030/` | 高频/EPI、低频/高频诊断、频谱图 |
| 实际 M1–M4 组合 | `results/b_composite_review_fixes_20261116/` | 全部 40 张预测掩码和真实输出 |
| B 四阶段计时 | `results/b_timing_review_fixes_20261116/` | 逐图耗时、环境、源码和配置摘要 |
| 人工评审包 | `results/b_manual_review_20261030/` | 原图、处理图、两名评价者的空表 |

这些目录属于本地结果。仓库仅提交代码、清单和文档，评测者运行下述命令可重新生成图表。每次运行使用新的输出目录，保留已有测量。

## 复现方式

先按 [README 快速开始](../README.md#快速开始) 安装依赖，并准备 40 张工作子集。在仓库根目录运行。

```powershell
python -m pytest -m "not integration" -q
python tools/inspect_haar_models.py
python tools/evaluate_m1.py --output results/m1_recheck
python tools/evaluate_m2.py --output results/m2_recheck
python tools/evaluate_m3.py --output results/m3_recheck
python tools/evaluate_m4.py --output results/m4_recheck
python tools/evaluate_b_composite.py --output results/b_composite_recheck
python tools/benchmark_b.py --output results/b_timing_recheck
```

各 `evaluate_m*.py` 对定量门槛返回退出码。`evaluate_b_composite.py` 检查实际组合输出与掩码边界，`benchmark_b.py` 保留真实耗时。两者的成功退出表示执行完成。它们分别验证组合行为和性能，不能替代隔离门槛或人工评分。

M3 产出 `figures/` 三联图、`visual_review.csv`、`roi_candidates.csv` 和 `roi_review.csv`。M4 产出 `figures/`、`diagnostics/`、`spectra/` 与 `visual_review.csv`。组织离线人工材料时运行。

```powershell
python tools/prepare_b_review.py --m3-run results/m3_recheck --m4-run results/m4_recheck --output results/b_review_recheck
```

打开生成目录的 `index.html`，按 100% 比例查看原图和处理图，再由两位非实现者分别填表。Windows 可使用 `.\.venv\Scripts\python.exe` 替换命令中的 `python`。

## 效果边界与待验收项

- 全部 40 张参与了开发校准，缺少独立留出集。清单里的 `scene` 尚未人工核验。真实欠曝只覆盖一张图。
- 遮挡、强高光和 Haar 检测失败时仍可能漏选或误选，固定集过线余量有限。
- B 四阶段在 512×512 输入下，平均 1.906638 秒，P95 为 2.902468 秒，25/40 张在 2 秒内。模型资源加载计入 M1，计时排除图像读写、缩放、M5–M7 和 UI，整套程序性能仍待 A 验收。
- 当前 CLI 写图层在 Windows 下受中文输出路径限制，运行时通过 `--output-root` 指定可写的 ASCII 路径，示例见 README。
- M3 固定 10 张边缘检查至少 9/10 张需由两位非实现者均判定通过。真实痘印需先确认至少 10 个 ROI、覆盖至少 5 张图，再评分；至少 80% ROI 需两人均判定可见度下降 ≥1 分，且自然度和五官保护均 ≥4 分。
- M4 固定 10 张高反差样例至少 9/10 张需由两位非实现者均判定无光晕、无双边缘。评分材料已准备，人工评分仍待完成。
- A 负责 PR 复审、最终联调和合并。
