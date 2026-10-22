# M2 CLAHE 配置变更建议（待共同评审）

日期：2026-10-22
关联任务：#19、#20
状态：本地提案；尚未发布远程 change-request，也未修改生效配置的 schema。

用户反馈（2026-10-22）：组长尚未确认，先保留提案；本次不接入 CLAHE 或新增生效配置键。

## 原因与现有约束

[#19](https://github.com/guts-yang/BeautyCam/issues/19) 要求“CLAHE + 自适应伽马”；当前 config/exposure.yaml 只提供 gamma / linear 和目标亮度、提升强度、过曝预算，没有 CLAHE 的裁剪限制或网格尺寸。

[arch.md 7.1](../docs/baseline/arch.md) 明确规定：“可调阈值、强度、核大小、色温等不得硬编码。”以及“改 schema、接口或跨模块语义时须走 change-request。”

因此没有把 CLAHE 的 clipLimit / tileGridSize 藏成源码常量，也没有把 gamma / gain 等现有键重解释成 CLAHE 参数。当前已完成可独立运行的自适应 gamma / linear 分支，CLAHE 部分待确认以下扩展。

## 建议的最小 YAML 增量

仅在 B 负责的 config/exposure.yaml 末尾增加：

```yaml
# 局部对比度增强；method 仍只表示后续 gamma / linear 曲线
clahe:
  enabled: true
  clip_limit: 2.0
  tile_grid_size: [8, 8]
```

| 键 | 类型/范围 | 含义 |
| --- | --- | --- |
| clahe.enabled | bool | 是否在曲线前执行 CLAHE |
| clahe.clip_limit | 有限正数 | OpenCV CLAHE 直方图裁剪限制 |
| clahe.tile_grid_size | 长度为 2 的正整数序列 | 横向/纵向网格数，不是单个格子的像素尺寸 |

上述初始值是待实测的候选，不是已经证明的最终参数。OpenCV 参数定义参考：[CLAHE 4.10 官方文档](https://docs.opencv.org/4.10.0/d6/db6/classcv_1_1CLAHE.html)。

## 接入行为

- 冻结的 process(image_bgr, skin_mask, cfg) 签名和 BGR uint8 返回值不变。
- method 保持 gamma / linear；不添加 pipeline 开关，不修改 A、C 文件或公共依赖。
- 在 Y 亮度上先做 CLAHE，以现有 gain 混合局部对比度结果，再用自适应曲线靠近目标；只回写皮肤掩码内。
- 继续用 max_gain 限制提升，并在实际输出 BGR 重算 Y 后检查过曝预算。
- 整个 clahe 块缺失或 enabled=false 时，逐字节复现现有 gamma / linear 分支，以兼容旧配置。
- 不改变 PRD 的欠曝定义、GT 区域、评分阈值或固定 40 张清单。

## 确认后需要完成的验证

1. 增加开关关闭的兼容测试、CLAHE 局部对比度实际变化测试、网格/裁剪参数生效与非法参数测试。
2. 重跑非集成回归及相同 40 张隔离评测；保留关闭/开启两组逐图分数、直方图和输入不变性检查。
3. 如候选参数导致提亮或过曝指标失败，只调整 B 的上述参数，不能放宽评分标准或剔除样本。
4. 全部材料通过后才将 #19 标为实现完成；PR 由 A 统一评审与合并。

现有 40 张里仅 1 张属于真实欠曝。若后续补充样本，须先固定、审阅新清单并单独报告，不能把合成暗化结果混入本轮真实欠曝成绩。
