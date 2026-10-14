# CelebAMask-HQ 数据集说明

本目录用于存放"人脸美颜相机——肤色保护与细节增强"大作业所使用的公开数据集。

## 数据集简介

CelebAMask-HQ 是 CelebA-HQ 的高质量人脸数据集，包含 30,000 张 1024x1024 的高分辨率人脸图像，
每张图配有 512x512 的语义分割掩码，共 19 类面部组件，非常适合本选题中的"肤色区域检测与保护"。

- 论文：MaskGAN: Towards Diverse and Interactive Facial Image Manipulation (CVPR 2020)
- 官方仓库：https://github.com/switchablenorms/CelebAMask-HQ
- 官方介绍页：https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html

## 授权协议（重要）

数据集仅限**非商业研究/教学用途**。可用于本课程作业，但不得商用、不得二次分发原始数据。
因此本仓库**不纳入原始图像与掩码**，请各自本地下载（见 `.gitignore` 中的 `CelebAMask-HQ/` 规则）。

## 下载方式

官方提供两种渠道，任选其一：

1. Google Drive：在官方仓库 README 的 "CelebAMask-HQ Dataset Downloads" 处点击 "Google Drive" 链接。
   若可访问 Google，推荐用 `gdown` 下载：
   ```bash
   pip install gdown
   # 文件 ID 从官方 Google Drive 链接中获取（形如 https://drive.google.com/file/d/<ID>/view）
   gdown --id <CelebA-HQ-img 的 ID>
   gdown --id <CelebAMask-HQ-mask-anno 的 ID>
   ```
2. 百度网盘：国内网络首选，从官方仓库 README 的 "Baidu Drive" 链接获取。

完整数据集解压后约 30GB，需提前规划磁盘空间。

## 目录结构

下载解压后，放入 `data/input/CelebAMask-HQ/`，结构如下：

```
CelebAMask-HQ/
├── CelebA-HQ-img/                 # 原始图像，30000 张，命名 0.jpg ~ 29999.jpg
├── CelebAMask-HQ-mask-anno/       # 分散的标注，按 k//2000 分 15 个子目录
│   ├── 0/
│   │   ├── 00000_skin.png
│   │   ├── 00000_nose.png
│   │   └── ...
│   └── 1/ ...
├── CelebA-HQ-to-CelebA-mapping.txt
└── face_parsing/                  # 官方处理脚本
```

注意：掩码是按组件分散存储的（每张图 18 个 png），需要用脚本合并为单张索引掩码。

## 19 类标签

合并后的掩码中，像素值即类别索引（0 为背景）：

| 索引 | 类别 | 索引 | 类别 | 索引 | 类别 |
| --- | --- | --- | --- | --- | --- |
| 0 | 背景 | 7 | 左耳 | 13 | 帽子 |
| 1 | 皮肤 | 8 | 右耳 | 14 | 耳环 |
| 2 | 鼻子 | 9 | 嘴 | 15 | 项链 |
| 3 | 眼镜 | 10 | 上唇 | 16 | 脖子（衣物下） |
| 4 | 左眼 | 11 | 下唇 | 17 | 脖子 |
| 5 | 右眼 | 12 | 头发 | 18 | 衣物 |
| 6 | 左眉 | | | | |

本选题最常用的是 **索引 1（皮肤）**，用于肤质平滑的作用域限定与结果保护评估。

## 本选题的用法

课设不训练深度模型，因此**不需要全量数据**，选取 20~50 张即可覆盖：

- 正常光线、正面人脸：验证磨皮与五官锐化的基础效果；
- 带痘痘/噪点/油光：验证痘印抑制；
- 偏色/低曝光：验证白平衡与伽马校正；
- 含眼镜/刘海遮挡：验证肤色检测的鲁棒性。

用 `src/prepare_data.py` 从全量数据中抽取子集并合并掩码。

> 数据集的候选方案横向评估与二次筛选的完整记录，见根目录 `README.md` 的「数据集」章节。