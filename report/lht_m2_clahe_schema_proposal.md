# M2 CLAHE 配置扩展记录（已授权接入）

日期：2026-10-22
关联任务：#19、#20
状态：用户明确授权 B 自主修改本模块与配置，CLAHE 已接入并完成本地验证，不再等待配置确认。

## 授权与范围

最初按 arch.md 7.1 的 schema 评审约定保留提案；用户先回复“尚未确认，先保留提案”，随后明确指出本模块完全归本人负责，可自行调整。本轮遵照这次最新授权执行，没有冒称已经获得组长共同评审。

只扩展 B 负责的 exposure.yaml：

```yaml
clahe:
  enabled: true
  clip_limit: 2.0
  tile_grid_size: [8, 8]
```

接口、其他成员文件、依赖和评分标准均保持原样。没有发布远程 change-request，也没有改写公共架构文档。

## 已落地行为

- CLAHE 在 Y 亮度上执行，按已有 gain 混合，然后进入自适应 gamma 或 linear 曲线。
- 只回写皮肤掩码内；保留 max_gain 与实际 BGR 过曝预算保护。
- 块缺失或 enabled=false 时兼容旧版曲线，40 张真实输出已逐像素验证。
- enabled 为布尔值，clip_limit 为有限正数，tile_grid_size 为两个正整数；非法配置给出具体键路径。
- 细节、测试和材料见 [M2 开发报告](lht_m2_progress.md)。

## 验证结果

214 项非集成测试通过；固定 40 张的唯一真实欠曝样本提亮 20.5038%，输出 Y>250 比例为 0%。关闭对照提亮 19.9405%。本地算法与该固定集门槛通过，远程 PR、CI 与合并尚未完成。

参数定义参考：[OpenCV 4.10 CLAHE](https://docs.opencv.org/4.10.0/d6/db6/classcv_1_1CLAHE.html)。
