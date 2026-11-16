# M1 评审修正记录

2026-11-16，qixing1227 / lht，[PR #52](https://github.com/guts-yang/BeautyCam/pull/52)。对应组长五条评审评论，回复与Resolve在推送及CI核验后执行。

## 协方差核查

已核对[OpenCV 4.10 GrabCut实现](https://github.com/opencv/opencv/blob/4.10.0/modules/imgproc/src/grabcut.cpp#L85-L104)。GMM构造器从1×65数组读取权重、均值和协方差，并另外计算inverseCovs与covDeterms缓存。GC_EVAL_FREEZE_MODEL仍执行这个构造流程。本项目写入协方差的布局符合此版本实现，保留原计算。增加各向异性高斯回归，正确协方差得到预期前景，将同一区域误换成逆矩阵则得到背景，并验证冻结模式保持模型数组字节。

## 偏位人像和开关

固定整图构图模式spatial.enabled已默认关闭，显式开启时继续采用原参数。默认face_locator.enabled=true，使用实际检测框定义局部图割种子和搜索域，参数位于face_locator.segmentation。center_offset、roi_scale、seed_scale均相对当前人脸框，随人脸移动。定位不可靠时使用颜色候选回退，不运行固定居中图割。增加偏上偏左皮肤保留、种子平移和开关独立性用例。冻结返回值仍为0/255掩码。

## 可调五官参数

features.dark、lips、eyes、brows、mouth_holes及dilate_kernel包含组长指出的暗结构阈值、搜索带、连通域几何、唇色阈值和分组距离。调用链贯穿初始分割、连续恢复及Haar后处理。缺省项保留原默认值，配置会验证映射、数值、比例、核尺寸和范围关系。增加修改阈值实际影响像素、YAML与默认值相同和无效参数报错用例。

## 模型来源和中文路径

删除配置中的压缩XML与原打包脚本，改用[opencv-python-headless现有包资源](https://github.com/opencv/opencv-python#usage)。当前skin_detect.yaml为4942字节，models仅保存三个包内资源名。官方直接路径API在本机中文安装目录下返回空级联，因此通过importlib.resources.files("cv2.data")读取包内XML，再使用FileStorage MEMORY及CascadeClassifier.read加载。模型原始SHA256和许可与此前封装字节一致。tools/inspect_haar_models.py可核对资源、版本和哈希。

此资源读取按本轮模型加载评审要求实施，是对原先“模块内部不读写文件”约束中模型加载部分的特定调整。模块继续通过内存接收用户图像，模型加载仅访问已安装依赖包。模型对象为调用局部，输入和配置保持只读。

配置注释现明确说明默认Haar开启、固定构图关闭，以及两个独立开关的含义。相应说明已在模块文档和许可来源说明中更新。

## 正式默认配置复测

| 完整40张 | 本次 |
| --- | --- |
| 原类1 IoU宏平均 | 0.752956443 |
| 原类1 Recall宏平均 | 0.903887828 |
| 后处理IoU变化 | -0.000170130 |
| 含鼻子辅助IoU（类1+2） | 0.819332405 |
| 鼻子覆盖宏平均 | 99.5941% |
| 眼眉选中比例 | 22.1348% |
| 口唇/口腔选中比例 | 22.6915% |

原协议三项通过。上次IoU 0.757008，本次变化-0.004052，这是默认分割初始化随人脸框变化后的结果。40张均参与开发，缺少独立留出集，遮挡和强高光仍有失败案例。原标注、清单和门槛沿用，类1+2指标单列。正式与实际M1–M4组合的全部40张掩码逐像素一致，源码、配置和清单哈希一致。351项非集成测试通过，1项未选。

512输入B四阶段独立计时，平均1.906638秒，中位数1.838386秒，P95 2.902468秒，最大3.421857秒，25/40张≤2秒。包资源读取、模型构建和检测计入M1，排除图像I/O、缩放、M5–M7和UI。完整程序性能仍待A验收。

M2–M4实现及配置沿用原隔离证据，M3/M4人工评分仍为空。核对121个材料哈希和820个评分格。当前实际预览results/b_showcase_20261022/revision_8/及evaluation40/。原始结果位于results/m1_review_fixes_20261116/、b_composite_review_fixes_20261116/和b_timing_review_fixes_20261116/。

```powershell
pytest -m "not integration" -q
python tools/inspect_haar_models.py
python tools/evaluate_m1.py --output results/新的目录
python tools/evaluate_b_composite.py --output results/新的组合目录
python tools/benchmark_b.py --output results/新的计时目录
```
