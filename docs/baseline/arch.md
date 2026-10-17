# arch.md —— 架构约束基线（焕颜相机 BeautyCam）

> 状态：**已确认（v1.0 基线锁定）**。接口及目录方案已冻结。
> 版本：v1.0 ｜ 适用项目：焕颜相机 BeautyCam（《数字图像处理》课程设计，2026）
> 文档语言：中文
> 关联基线：[prd.md](./prd.md)、[design.md](./design.md)

---

## 1. 用途与系统边界

本文件规定技术栈、项目结构、模块职责、接口、数据契约、测试与本地运行约束。系统是本地 Python 图像处理课程项目，提供 CLI 和本地 Streamlit 演示，不提供生产环境服务端。

课程算法模块为 M1–M6，评价模块为 M7。算法仅使用传统数字图像处理，不引入深度学习框架或预训练模型。

## 2. 技术栈与版本约束

| 层 | 选型 | 约束 |
| --- | --- | --- |
| 语言 | Python | 最低 3.10；本地与 CI 用 3.10.x，云端部署用 3.12（Streamlit Cloud 可选下限），试装后确认 |
| 图像处理 | OpenCV | 使用 `opencv-python-headless`；不得同时安装 `opencv-python` 与 headless 发行包 |
| 数值计算 | NumPy | 直接依赖精确 pin |
| 演示 UI | Streamlit | 直接依赖精确 pin，本地启动 |
| 图表 | Matplotlib | 直接依赖精确 pin |
| 配置 | PyYAML | 直接依赖精确 pin |
| 图像指标 | scikit-image | 用于 SSIM、Lab/CIEDE2000 等传统指标，不使用其深度学习功能；直接依赖精确 pin |
| 测试 | pytest | 直接依赖精确 pin |

所有直接依赖最终均须在 `requirements.txt` 写为 `package==version`。具体版本号须在干净 venv 试装、CLI/UI 启动和测试通过后锁定；不得猜版本或在未验证环境下宣称可安装。

## 3. 项目结构（v1.0 冻结）

```text
BeautyCam/
├── .github/
│   └── workflows/
│       └── tests.yml                  # PR 单元测试 CI
├── .streamlit/
│   └── config.toml                    # Streamlit 主题映射
├── src/
│   ├── main.py                        # CLI 薄入口，只调用 core
│   ├── app.py                         # Streamlit 薄入口，只调用 core
│   ├── prepare_data.py                # 课程数据预处理工具
│   └── beauty/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py              # 加载/合并/校验配置
│       │   ├── contracts.py           # core 与入口间的结果类型
│       │   └── pipeline.py             # 唯一流水线编排点
│       ├── io/
│       │   ├── __init__.py
│       │   ├── reader.py               # 图像/GT 读取与规范化
│       │   └── writer.py               # 图像、CSV 和结果持久化
│       ├── modules/
│       │   ├── __init__.py
│       │   ├── skin_detect.py          # M1 (B)
│       │   ├── exposure.py             # M2 (B)
│       │   ├── smooth.py                # M3 (B)
│       │   ├── freq_enhance.py         # M4 (B)
│       │   ├── restore.py               # M5 (C)
│       │   └── color.py                 # M6 (C)
│       └── metrics/
│           ├── __init__.py
│           └── metrics.py               # M7 (C)，纯计算
├── tests/
│   ├── unit/                            # 合成数据，不依赖外部数据集
│   └── integration/                     # 本地真实数据集评测
├── config/
│   ├── pipeline.yaml                    # 顺序、阶段开关、路径和运行模式（A）
│   ├── skin_detect.yaml                 # M1 参数（B）
│   ├── exposure.yaml                    # M2 参数（B）
│   ├── smooth.yaml                      # M3 参数（B）
│   ├── freq_enhance.yaml                # M4 参数（B）
│   ├── restore.yaml                     # M5 参数（C）
│   ├── color.yaml                       # M6 参数（C）
│   └── metrics.yaml                     # M7 参数/评测协议（C，协议变化须共同评审）
├── assets/
│   ├── test-images/                        # 仓库现有课程测试图；新增图像须核对许可
│   ├── evaluation_manifest.csv           # 固定评测样本 ID/场景清单，不含图像像素
│   └── blemish_roi_manifest.csv          # R2.3 若确认，记录匿名样本 ROI 坐标
├── data/
│   ├── input/                           # 本地输入，不提交受限数据
│   └── output/                          # 运行产物，默认忽略
├── results/                             # 每次运行的指标、失败清单、图表
├── docs/                                # 课程资料和项目文档
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

目录边界：

- `core` 是唯一流程编排和配置合并层；`io` 只做输入/输出；`modules` 和 `metrics` 只做纯计算。
- `modules` 之间不互相依赖；`metrics` 不 import 图像处理模块。
- `.streamlit/config.toml` 只承载 UI 主题；算法参数不得放到 UI 配置。

## 4. 架构分层与职责

### 4.1 允许的依赖方向

```text
CLI / Streamlit UI
        ↓
core.pipeline + core.config
   ↓          ↓          ↓
modules M1–M6  metrics M7  io
```

- `main.py`、`app.py` 只调用 `core.pipeline` 和必要的配置/输入适配，不直接调用任何算法模块。
- `core.pipeline` 是唯一能够按流程调用 M1–M6 与 M7 的层。
- `modules` 彼此禁止 import；`metrics` 不 import `modules`；算法模块不依赖 UI、CLI、core 状态或文件系统。
- `io` 负责读写，不承载算法，也不被算法模块调用。
- 图像数组在进程内由 `core.pipeline` 显式传递；磁盘只作持久化边界，不作为模块之间的通信总线。

### 4.2 模块职责与文件所有权

| 责任域 | 源码/配置 | owner 与写入边界 |
| --- | --- | --- |
| M1 肤色检测 | `modules/skin_detect.py`、`config/skin_detect.yaml` | B 独占 |
| M2 曝光改善 | `modules/exposure.py`、`config/exposure.yaml` | B 独占 |
| M3 空间域平滑 | `modules/smooth.py`、`config/smooth.yaml` | B 独占 |
| M4 频域增强 | `modules/freq_enhance.py`、`config/freq_enhance.yaml` | B 独占 |
| M5 运动模糊复原 | `modules/restore.py`、`config/restore.yaml` | C 独占 |
| M6 色彩/白平衡 | `modules/color.py`、`config/color.yaml` | C 独占 |
| M7 评价计算 | `metrics/metrics.py`、`config/metrics.yaml` | C 独占；评测协议变化须 A/B/C 评审 |
| 流程/配置 | `core/`、`config/pipeline.yaml`、CLI、UI | A 独占 |
| I/O | `io/` | A 独占 |
| 测试 | `tests/unit/` 对应模块测试 | 各模块 owner；A 维护集成/CI 测试 |
| 依赖/CI | `requirements.txt`、`.github/workflows/tests.yml` | A 提 PR；至少一名其他成员评审 |

模块配置文件让各 owner 可以独立开发。模块参数 schema 在并行开发前共同评审；模块 owner 只修改自己的配置文件。改公共接口、`pipeline.yaml`、评测协议或依赖集合时先提变更 Issue，评审后由 A 合并，避免共享文件冲突。

## 5. 接口与数据契约（v1.0 冻结）

### 5.1 按职责定义接口

M1 产出掩码，M5 做可选复原，M2/M3/M4/M6 做图像变换，M7 产出指标；不强制七个模块返回同一种对象。以下签名定义输入/输出形状，不是实现代码：

```python
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def detect_skin(image_bgr: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """M1：BGR uint8 HxWx3 -> 同尺寸 uint8 HxW 掩码，取值 0/255。"""

def process(image_bgr: np.ndarray, skin_mask: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """M2/M3/M4/M6：图像与掩码 -> 新的 BGR uint8 图像。"""

def get_frequency_diagnostics(image_bgr: np.ndarray, cfg: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    """M4 可选诊断：返回 low_frequency/high_frequency 两张同尺寸 BGR uint8 可视化图。"""

def restore_motion_blur(image_bgr: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """M5：BGR uint8 图像 -> 新的 BGR uint8 图像；不依赖 M1。"""

def compute_image_metrics(stage_input_bgr: np.ndarray | None,
                          stage_output_bgr: np.ndarray | None,
                          regions: Mapping[str, np.ndarray],
                          reference_bgr: np.ndarray | None,
                          predicted_skin_mask: np.ndarray | None,
                          cfg: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    """M7：对可用输入计算指标；返回纯数据记录，不做文件 I/O。"""

def summarize_survey(responses: Sequence[Mapping[str, Any]],
                     cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """M7：匿名问卷记录 -> 汇总统计；不读写文件。"""
```

类型约束：`np.ndarray` 仅作接口说明。图像为 OpenCV BGR、`uint8`、`H×W×3`；掩码为同尺寸 `uint8`、仅取 0/255。`cfg` 为模块配置键到标量/列表的映射，不包含文件句柄、UI 状态或模块对象。

指标记录采用普通映射，不定义跨模块自有类。每个 `metric_name` 对应记录包含：`value`（有限数值或空值）、`unit`、`valid`（布尔值）、`not_applicable_reason`（有效时为空；无效时说明原因）。`valid=false` 时 `value` 必须为空，不得写 NaN/Infinity；`valid=true` 时必须是有限数值。

接口约束：

- `regions` 是 GT 区域名到同尺寸 0/255 掩码的映射，候选键为 `skin_gt`、`face_feature_gt`；缺失区域代表相应指标不可评估，不得用预测掩码冒充 GT。
- `predicted_skin_mask` 仅用于 R1 预测掩码指标；无预测掩码时对应指标不可评估。
- `reference_bgr` 只在已知清晰/中性参考样本存在时提供；普通上传图无可信参考时不计算 PSNR、SSIM 或 ΔE2000 等参考型指标。
- `stage_input_bgr` / `stage_output_bgr` 用于阶段前后指标；M1 可只提供 `predicted_skin_mask`，不要求将掩码伪装成图像结果。
- 图像变换不得原地修改输入；必须返回新数组并满足 shape/dtype 契约。
- M1 全零掩码是合法结果；由 pipeline 给出状态。M2/M3/M4/M6 在空掩码上不对整图执行肤色限定处理，按配置安全跳过。
- M2/M3/M4/M6 只读取传入图、掩码和各自配置，不读写文件、不维护全局状态。
- M7 只接收数组/评测区域和配置，返回纯数据记录；CSV 由 `io/writer.py` 写入。
- `PipelineResult` 可在 `core/contracts.py` 定义，供 UI/CLI 查看 `run_id`、`config_id`、最终图、可选阶段图、掩码、指标记录和运行状态；算法模块不得依赖该类型或用它互相传状态。

### 5.2 评测区域与数据对齐

- **数据获取最小化（顶层约束）**：所有数据集脚本（下载、解压、预处理、抽样）只获取工作所需的最小样本子集，禁止下载完整数据集或全量抽取。本课题不训练深度模型，工作子集为 20~50 张；`prepare_data.py fetch` 以 HTTP Range 按需抽取，不下载整包。个别样本缺标注时按跳过处理，不得因此改为全量拉取。确需全量数据须走 `change-request`。
- `src/prepare_data.py` 生成的子集目录为 `images/`、`masks/`、`skin_masks/`；GT 标签图按最近邻插值对齐到被测图尺寸。
- CelebAMask-HQ 合并标签图中的皮肤标签索引为 1。五官 EPI 的候选标签为 `eye_g/l_eye/r_eye/l_brow/r_brow/mouth/u_lip/l_lip`，对应当前脚本的一基索引 `{3,4,5,6,7,10,11,12}`。
- 固定评测集清单版本管理在 `assets/evaluation_manifest.csv`，至少记录 `sample_id,scene`；路径由数据根目录和 ID 依照固定规则构造，若写路径则只能是项目相对路径。清单不得包含机器绝对路径或图像像素。
- 当前 `prepare_data.py` 按随机种子抽样，但枚举文件未先排序、没有分层抽样，也不生成 manifest。若基线确认 40 张固定/分层评测集，后续需通过独立 Issue 修订预处理脚本并人工核验清单；完成前不得声称已分层或完全可复现。
- 若 R2.3 确认，`assets/blemish_roi_manifest.csv` 记录 `sample_id,x,y,width,height`；坐标以 1024×1024 原图左上角为原点的像素坐标，ROI 必须完全位于 skin GT 内。匿名评分记录单独保存于被忽略的本地评测产物中；ROI 清单不代替算法掩码，也不进入推理接口。
- 仓库现有 `assets/test-images/` 保留为课程测试素材；新增图像须核对许可。CelebAMask-HQ 原始图/掩码、派生子集、用户上传肖像和运行产物不得提交；CI 使用合成样例。

## 6. 流水线与执行规则

唯一执行编排点为 `core/pipeline.py`：

```text
输入 BGR 图
  → M5 运动模糊复原（pipeline.yaml 中 restore_enabled=true 时执行；默认 false）
  → M1 肤色检测
  → M2 曝光改善
  → M3 空间域平滑
  → M4 频率域增强（正式 evaluation profile 中 diagnostics.frequency_enabled=true，并另调用 get_frequency_diagnostics）
  → M6 色彩/白平衡
  → M7 阶段评价（仅计算所需图像、GT 和参考均可用的指标）
  → io 写入每次运行的最终图、可选阶段图、指标/失败状态
```

- `pipeline.yaml` 固定上述运行顺序和阶段开关。完整课程 profile 中 M1/M2/M3/M4/M6 必须开启；M5 默认关闭，但必须在其模块隔离合成模糊评测中显式开启。其他阶段关闭只允许用于消融/演示，并在结果中标为非完整运行。
- 运行模式分为 `preview`、`process`、`evaluation`：`preview` 只在本地内存处理并返回结果，不创建持久化运行目录；`process` 由 CLI 运行并保存结果图/状态；`evaluation` 使用固定 `assets/evaluation_manifest.csv`、本地 GT/参考图并保存指标。UI 用户显式导出时才把预览结果交给 I/O 层保存。
- `evaluation.profile` 取 `module_isolated` 或 `end_to_end`。`module_isolated` 需提供 `target_module` 且只启用被测模块，PRD 中模块阈值只以此 profile 验收；M1 使用输入图评预测掩码，M2/M3/M4/M6 接收 skin GT 掩码，M5 使用清晰参考合成的运动模糊输入，M6 使用记录了 RGB 通道增益的偏色输入。`end_to_end` 按标准顺序运行完整流程并单独报告，不替代任何模块隔离阈值。
- M4 诊断开关 `diagnostics.frequency_enabled` 位于 `pipeline.yaml`，普通 `process` 默认 false、`evaluation` 的 M4 隔离 profile 默认 true；关闭时不生成频率分解图，且该运行不满足 R3.1 的诊断产物验收。开启时，pipeline 将 M4 输入图与 M4 配置传给 `get_frequency_diagnostics`，再由 I/O 层写入本次运行的 `diagnostics/frequency/low_frequency.png` 与 `high_frequency.png`。诊断接口只返回可视化数组，不做文件操作，不改变 M4 主处理输出。
- 模块算法参数放各自 YAML；阶段启用状态只放 `pipeline.yaml`。参数覆盖优先级为代码默认值 < YAML 配置 < 本次 CLI/UI 明确覆盖；UI 覆盖只作用于当前 run，不回写 YAML。
- 普通用户上传图没有 GT 时仍可运行美化；UI 不显示没有参考依据的客观质量分数，应说明哪些指标不可评估。
- M7 在 `evaluation` 模式中按阶段接收该阶段输入/输出、适用预测掩码、GT 区域和可信参考图；M1 可计算 IoU/Recall，M2/M3/M4/M5/M6 计算各自适用指标。禁用阶段或缺失必要参考时，生成 `valid=false`、空 `value` 和原因；`preview` 模式不生成评测记录。
- 无皮肤掩码时，M2/M3/M4/M6 不得把整图误当皮肤处理；pipeline 返回可读状态，不崩溃。
- 单张运行遇到错误时返回失败状态且不把部分阶段结果伪装成最终结果；批处理将错误写入本次运行失败清单后继续下一个样本。
- M7 每次运行只做纯计算；pipeline/io 统一写 CSV 与图像文件。

## 7. 配置、路径与产物

### 7.1 配置

- 用户可调参数位于 `config/`；模块算法参数各自放在模块专属 YAML；`pipeline.yaml` 仅放固定顺序、阶段开关、运行模式、路径与 I/O 选项。
- `core/config.py` 合并配置并校验必需键、类型、范围；报错指出配置文件和键路径。
- 数学常量可定义为有名源码常量并说明用途；可调阈值、强度、核大小、色温等不得硬编码。
- 模块 schema 在并行开发前共同确认；模块 owner 只改自己的模块 YAML。改 schema、接口或跨模块语义时须走 `change-request`。

### 7.2 路径与运行标识

- 默认路径以仓库根目录为基准，不依赖用户当前工作目录；CLI/UI 可接受用户指定输入/输出路径。
- 禁止提交机器绝对路径、用户目录路径和凭据。
- 每次 CLI 正式运行或评测运行分配唯一 `run_id`，格式为 UTC 时间戳加随机短后缀；创建运行目录时采用原子方式，若发生碰撞则重新生成，不静默覆盖已有结果。纯 `preview` 不创建持久化运行目录。
- 评测 `sample_id` 取自固定 manifest；普通 CLI 批次按输入根目录下排序后的相对路径分配稳定序号 ID，输出记录中不得包含输入根目录绝对路径。UI preview 不产生持久化 sample ID；显式导出时再分配。
- `config_id` 是本次合并后有效处理配置（算法参数、阶段开关及本次 UI/CLI 处理参数）按键排序、规范化序列化后的 SHA-256 前 12 位十六进制摘要；排除输入/输出路径、用户目录和凭据，避免泄露个人路径。
- 基线确认后的实现需更新 `.gitignore`，排除 CelebAMask-HQ 原始数据、派生子集、`data/output/` 运行产物、`results/` 运行记录和临时文件。若需归档报告图，须复制到课程报告资产目录并核查许可，不直接提交原始数据。

### 7.3 产物路径与 CSV 契约

- 每个样本的最终图：`data/output/<run_id>/samples/<sample_id>/final.png`；启用阶段图时存放在该样本目录的 `stages/<stage>.png`。
- 可选 M4 诊断图存放在 `data/output/<run_id>/samples/<sample_id>/diagnostics/frequency/low_frequency.png` 与 `high_frequency.png`；不启用诊断时不生成。
- 运行记录：`process`/`evaluation` CLI 运行在 `results/<run_id>/` 下写入 `failures.csv`；仅 `evaluation` 模式写 `metrics.csv`。一次批处理共享一个 `run_id`，每个样本各有 `sample_id`；每次运行使用新目录，不跨运行追加。`preview` 不创建该目录，用户导出时只保存图像产物。
- `metrics.csv` 固定字段：`run_id,sample_id,stage,metric_name,value,unit,valid,not_applicable_reason,config_id`。每个样本/阶段/指标一行；无效值的 `value` 留空，`valid=false` 时填写原因。
- 匿名问卷汇总写入 `results/survey/<survey_run_id>/summary.csv`，只保存题目/统计量/分布/有效样本数，不保存原始答卷或身份信息；原始答卷如需临时读取，只能放在被 `.gitignore` 排除的本地路径。
- `failures.csv` 固定字段：`run_id,input_relative_path,stage,error_type,error_message`；路径相对于项目根目录或输入根目录，不含本机绝对路径。
- CSV 使用 UTF-8、LF 换行；有效浮点值以十进制格式保留 6 位；字段顺序固定并由 writer 测试校验。

## 8. 测试与 CI

- 测试框架为 pytest。`pytest.ini` 至少固定 `testpaths=tests`、`pythonpath=src`，并声明 `integration` marker，保证从仓库根目录执行时可以导入 `beauty` 包。
- 每个模块至少有一个正向用例和一个边界用例；图像变换模块额外检查输入不变性及图像 dtype/shape，M1 检查掩码 dtype/shape/取值，M7 检查指标记录字段、有效/无效值规则，避免对非图像模块套用图像尺寸断言。
- `tests/unit/` 和 PR CI 只使用内存生成的合成图、掩码和 CSV fixture，不依赖 CelebAMask-HQ 或开发者本地文件。
- `tests/integration/` 使用本地数据集和固定 `assets/evaluation_manifest.csv`，标记为 `integration`；仅在配置好数据根目录后执行 `pytest -m integration`。缺少数据时给出清晰前置条件，不静默伪装为通过。
- GitHub Actions 在 Ubuntu runner、Python 3.10.x 上安装已 pin 依赖并执行 `pytest -m "not integration"`；CI 不下载受限数据、不访问需要鉴权的资源。CI 失败不得通过跳过或删除测试规避。
- 每个 PR 必须通过 CI 并关联 Issue；模块 PR 仅改本模块、模块配置和测试。公共契约/流水线/CI 变更由 A 负责并经至少一名其他成员评审。
- A 负责主流程单图、空掩码和批处理失败隔离集成测试；测试门禁适用于所有优先级。

## 9. 本地运行与交付

- 本地环境：Python 3.10.x venv；不要求容器或云环境。云端部署（Streamlit Community Cloud）使用 Python 3.12，当前 pin 依赖在 3.12 有完整预编译 wheel，已验证可安装。
- CLI：`python src/main.py --help`，按帮助运行单图或批处理。
- UI：本地入口 `streamlit run src/app.py` 作为主运行方式；课程演示允许部署到 Streamlit Community Cloud（绑定 main 分支，push 自动同步），上传原图只在本地/云端进程内存处理，不自动复制持久化，该约束不因部署形态改变；本地运行保留为离线兜底。
- 环境安装：`python -m pip install -r requirements.txt`。
- README 必须说明环境安装、数据准备、单图运行、批量运行、UI 启动、集成测试命令及常见错误处理。
- 提交源码和课程允许提交的文档/结果；不提交可执行二进制、数据集、派生图像或含个人身份信息的问卷原始数据。
- 数据仅用于课程教学/研究，遵守数据集许可，不二次分发。

## 10. 基线确认结论（v1.0）

以下条目已按预设值锁定为基线，开发中如需调整走 `change-request`。

- [x] OpenCV 采用 `opencv-python-headless` 发行包。
- [x] Python 3.10.x（本地与 CI）+ 3.12（云端部署）双版本约束，pin 依赖在两版本均有预编译 wheel。
- [x] 所有直接依赖精确 pin，具体版本在干净环境试装后补入。
- [x] 分层目录与按模块独立 YAML 配置。
- [x] M5 默认关闭，合成模糊评测时显式启用。
- [x] M1/M5/M7 差异化接口及模块纯计算约束（按第 5 节）。
- [x] 运行目录、CSV 长表字段与 12 位配置摘要（按第 7 节）。
- [x] 课程“可执行程序”按可运行 Python 源码解释；若教师要求打包二进制再走变更。
- [x] UI 上传图不自动持久化，结果仅在显式保存/导出时保留。

变更记录：

- 2026-10-17 change-request #1（A 发起）：UI 部署约束放宽——允许部署到 Streamlit Community Cloud 用于课程演示，上传图仅进程内存处理、不持久化的原约束不变；合并前须经至少一名其他成员评审。
- 2026-10-17 change-request #2（A 发起）：云端部署 Python 版本约束——Streamlit Cloud 平台已不提供 3.10/3.11 选项，云端部署环境定为 Python 3.12（平台可选下限）；本地与 CI 仍用 3.10.x，CI 增加双版本矩阵守护。pin 依赖已在 3.12 wheel 检验通过，代码无 API 兼容性问题。