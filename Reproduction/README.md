# SonicSurface 论文复现

论文：**Generating Airborne Ultrasonic Amplitude Patterns Using an Open Hardware Phased Array**
（Morales, Ezcurdia, Irisarri, Andrade, Marzo — *Applied Sciences* 2021, 11(7), 2981）
[DOI: 10.3390/app11072981](https://doi.org/10.3390/app11072981) · 开放获取 CC BY 4.0

## 目录内容

| 文件 | 说明 |
|---|---|
| `论文中文翻译.md` | 论文全文中文翻译（含图表注、通信协议表） |
| `sonicsurface_algorithm.py` | 论文第 4 节迭代算法的纯 Python 移植（角谱法正/反向传播 + 阵列离散化约束） |
| `reproduce_fig9_sweep.py` | 复现论文图 9 / 第 5.2 节：发射器尺寸、相位分辨率、幅度分辨率的参数扫描 |
| `python_result.png` | 笑脸图案复现结果（目标 vs 重建） |
| `fig9_sweep.png` | 图 9 复现曲线（MSE 随三个分辨率参数的变化） |
| `fig9_sweep_results.json` | 扫描的原始 MSE 数据 |
| `smiley.png` | 测试图案（复制自 `Algorithms/EmissionPhasesAmpsForTargetSlice/`） |

原始 MATLAB 代码在 `../Algorithms/EmissionPhasesAmpsForTargetSlice/`，已用本机 MATLAB R2025b 验证可运行。

## 算法原理（论文第 4 节）

迭代修改版 Gerchberg–Saxton 算法，每次迭代四步：

1. **目标面约束**：将目标幅度"盖"到当前目标场上，保留其相位
2. **背向传播**：角谱法（FOCUS 库 `fftasa`）把目标场反传回发射面
3. **发射面约束**：下采样到换能器网格 → 幅度/相位量化（如相位 32 级）→ 最近邻上采样 → 乘圆形发射器掩模
4. **正向传播**：把发射面正传到目标面

循环 50 次后，从发射面中心提取每个换能器的幅度与相位。

Python 移植与 MATLAB 原版的差异：`imresize` 用 `scipy.ndimage.zoom` 近似（MATLAB 下采样默认带抗混叠）；倏逝波用复数开方处理（与 MATLAB 的复数 `sqrt` 行为一致）。

## 运行方式

```bash
# 1. 笑脸图案复现（对应 MATLAB Test.m）
python sonicsurface_algorithm.py

# 2. 图 9 参数扫描（约 1–3 分钟）
python reproduce_fig9_sweep.py

# 3. 原始 MATLAB 版本
cd ../Algorithms/EmissionPhasesAmpsForTargetSlice && matlab -batch Test
```

依赖：`numpy`、`scipy`、`matplotlib`、`Pillow`（MATLAB 版另需 Image Processing Toolbox）。

## 复现结果

### 笑脸图案（16 cm 距离，50 次迭代，发射器 5 mm，相位 32 级，无幅度调制）

| 实现 | MSE |
|---|---|
| 原始 MATLAB（`Test.m`，R2025b 实测） | **0.0731** |
| Python 移植（`sonicsurface_algorithm.py`） | **0.0586** |

两者同一量级，重建图案均清晰可辨（见 `python_result.png` / `../Algorithms/EmissionPhasesAmpsForTargetSlice/matlab_result.png`）。

### 图 9 甜点分析（见 `fig9_sweep.png`，原始数据 `fig9_sweep_results.json`）

论文结论 → 复现实测（4 个图案：smiley / A / circle / curve，MSE 越低越好）：

1. **发射器越小 MSE 越低，但小于 λ/4（≈2.2 mm）后收益消失** — ✓ 复现成立。
   circle: 10 mm→0.0483，6 mm→0.0352，2 mm→0.0319，之后不再改善；curve 同样在 2–3 mm 进入平台。
   （smiley/A 单次迭代存在收敛随机性，趋势一致但曲线略抖动。）
2. **相位分辨率约 8 级/周期后进入平台期** — ✓ 复现成立。
   smiley: 2 级 0.1487 → 8 级 0.0916 → 16 级 0.0828 → 32 级 0.0921（平台/略回升）；其余图案一致。
3. **幅度调制即使只有 4 级也能把 MSE 降低约一半** — ⚠️ 部分复现。
   4 级幅度相对无幅度基线的改善为：A 0.0713→0.0491（−31%）、smiley 0.0921→0.0752（−18%）、
   circle 0.0483→0.0444（−8%）、curve 0.0279→0.0289（持平）。方向一致但幅度弱于论文的"减半"；
   论文使用的是巴西国旗、鸽子等不同图案。另注意 2 级幅度（0/0.5 两档）反而常比无调制更差。

已知伪影：128×128 图案（circle/curve）在 1 mm 发射器处为 nan —— 发射器小于 1 像素导致圆形掩模
覆盖不到任何像素（`mask` 全零），属网格分辨率限制，非物理结果，图中该点留空。

## 硬件部分（未复现，需要实物）

论文的硬件（16×16 换能器阵列、FPGA 固件、ESP32 控制器、PCB）无法软件复现。仓库中相关资源：

- `Electronics/` — PCB Gerber 与 BOM（SMD 版 16×16、10 mm 换能器）
- `Firmware/FPGA primary/` — FPGA 固件（VHDL 源码 + Quartus 工程）
- `Firmware/ESP32 controller/` — ESP32 无 PC 独立控制固件（含 C 版聚焦/IBP 算法）
- `ControlSoftware/Python/` — 上位机驱动（聚焦、涡旋、双阱、多点 IBP，需串口连接实物）
- 组装教程：<https://www.instructables.com/SonicSurface-Phased-array-for-Levitation-Mid-air-T/>
