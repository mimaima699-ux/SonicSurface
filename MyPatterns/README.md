# MyPatterns — 自制 SonicSurface 图案脚本

不依赖 Ultraino AcousticFieldSim(Java/NetBeans GUI)的独立控制脚本:
**算 256 个相位 →(可选)声场仿真预览 → 串口发给 FPGA 板**。

```
图案函数 ──► preview_slice(活塞模型仿真,先看后发) ──► Board.send(串口)
                                                    254 + 256相位字节 + 253
```

## 依赖

```bash
py -3.13 -m pip install numpy scipy matplotlib pillow pyserial
```

(本机 Python 3.13 已装齐;控制台中文乱码时先 `set PYTHONUTF8=1`)

## 快速开始

```bash
cd D:/Mima/SonicSurface

# 1. 不连板,先看仿真(生成预览图)
py -3.13 MyPatterns/my_patterns.py focus 0 0 --save pv.png --dry
py -3.13 MyPatterns/my_patterns.py focus 0 0 --preview --dry   # 弹窗交互

# 2. 连板实测(先 python ... ports 看串口号)
py -3.13 MyPatterns/my_patterns.py ports
py -3.13 MyPatterns/my_patterns.py focus 0 0 --port COM4       # 发送后板子持续发射
py -3.13 MyPatterns/my_patterns.py off --port COM4             # 停止发射

# 3. 发送前先弹预览,关窗后自动发送
py -3.13 MyPatterns/my_patterns.py vortex 0 0 --m 2 --preview --port COM4
```

## 命令一览

| 命令 | 说明 | 示例 |
|---|---|---|
| `ports` | 列串口 | `ports` |
| `focus X Z` | 单焦点 | `focus 0 0` |
| `vortex X Z [--m M]` | 涡旋(拓扑荷 m) | `vortex 0 0 --m 2` |
| `twin X Z [--angle A]` | 双阱 | `twin 0 0 --angle 90` |
| `bottle X Z` | 瓶形阱(涡旋+双阱) | `bottle 0 0` |
| `points X1 Z1 X2 Z2 …` | 多点 IBP | `points 0.03 0 -0.03 0` |
| `image PATH [--method array/paper]` | 图片图案(GS 迭代;默认 array=阵列感知版,`paper`=论文原版) | `image ../Reproduction/smiley.png` |
| `steer A` | 波束偏转 A 度(仅小角度) | `steer 8` |
| `burst [--dur 1.5] [--freq 200]` | 持续调制触觉爆发(双缓冲 253 高频互换实现 AM)。**嗡鸣 ∝ f²,想安静用 `--freq 40`** | `burst --port COM7 --dist 0.12 --freq 40` |
| `tap [--pulse 0.05] [--count 1]` | **单击脉冲**:50ms 全功率焦点,无调制→几乎无声,只有轻"嗒",像被按了一下 | `tap --port COM7 --count 3` |
| `hold [--freq 100] [--duty 0.7]` | **持续调制发射**(burst 不限时版):100Hz 嗡到你按回车为止。硬件应力低于悬浮模式,可长时间运行;Python 进程须保持(忙等约占一个核) | `hold --port COM7 --dist 0.12 --freq 100` |
| `anim orbit/scan/spin/mpx/images` | 动画(Ctrl+C 停止并关发射) | `anim orbit --kind vortex --port COM4` |
| `off` | 关闭发射 | `off` |

公共参数:`--port`(COM 名或序号)、`--dist`(距离,默认 0.16 m)、`--preview`/`--save`/`--dry`、`--axis y|z|x`(预览切面)、`--fps`(动画帧率)。

## Python API

```python
import sys; sys.path.insert(0, 'MyPatterns')
import numpy as np
from my_patterns import (phases_focus, phases_vortex, phases_twin, phases_bottle,
                         phases_multipoint, phases_from_image, phases_steer,
                         preview_slice, Board, animate, frames_orbit)

pos = (0.0, 0.16, 0.0)
ph = phases_vortex(pos, m=1)                 # 256 个相位(弧度)
preview_slice(ph, axis='y', at=0.16, title='vortex')   # 先看

b = Board('COM4'); b.send(ph); b.close()     # 后发(断开后板子继续发射)

# 双点悬浮(时分复用)
for _ in range(200):
    b.send(phases_focus(( 0.03, 0.16, 0))); 
    b.send(phases_focus((-0.03, 0.16, 0)))

# 图片图案
ph, info = phases_from_image('../Reproduction/smiley.png', dist=0.16)
print('MSE', info['mse'])

# 动画:涡旋绕圈
animate(b, frames_orbit(0.16, 0.03, kind='vortex'), fps=20)
```

## 坐标与方向约定

- 阵列在 **x–z 平面**,法线 **+y**;`EMITTERS` 为 256 个换能器的**实测坐标**(同官方 `SonicSurface.py`)。
- 目标点写 `(x, y, z)`,y = 离阵距离(米);CLI 里焦点给 `X Z`,距离用 `--dist`。
- 图片图案:**图片"上"= +z,"右"= +x**(从 +y 向下看)。若实物图案镜像/颠倒,用 `--flipx / --flipz / --transpose` 修正。

## 重要注意事项

1. **相位校正只在 FPGA 做**。你的固件 `Distribute.vhd` 已内置 `PHASE_CORRECTION`(= 仓库根目录 `超声波参数.txt`),
   本脚本发送**未校正**的原始相位——两边都加会重复叠加。Ultraino 软件里若也填了校正,同理只留一处。
2. **发送后板子持续发射**(FPGA 双缓冲保持最后图案),断电或 `off` 命令才停。
3. 阵元间距 10.5 mm > λ/2(4.3 mm):`steer` 大角度会出现栅瓣,仅小角度可用;聚焦/涡旋/图案不受影响。
4. 幅度命令(251)与 200 Hz 调制(252)属于幅度版/触觉版固件,你的 primary 固件**不支持**;
   本脚本只发相位(0–31,32=关闭),与官方 `sendPhases` 一致。
5. 上板前建议永远先 `--preview` 确认声场形状、焦点位置正确。

## 与 Ultraino 软件的对应关系

| Java 软件 | 本脚本 |
|---|---|
| Traps 面板(focus/twin/vortex)+ send 勾选 | `focus` / `twin` / `vortex` / `bottle` 命令 |
| Control Points + SimpleIB | `points`(IBP 多点) |
| Kinoforms/HoloPatterns(图片全息) | `image`(默认阵列感知 GS;`--method paper` 为论文 §4 算法) |
| RtSlice 实时切片可视化 | `preview_slice` / `--preview` |
| Move/Anim/SwitchTimer | `anim orbit/scan/spin/mpx/images` |
| TransControlPanel + SimpleFPGA 协议 | `Board`(254+256+253 @230400) |

## 已做的验证(2026-09-18)

- `--dry` 仿真:focus(中心单焦点)、twin(双叶夹暗区)、points(IBP 双焦点精确落位)、侧视剖面,均人工核验预览图无误。
- **图像图案(最终态)**:默认 `method='array'`(阵列感知 GS:用与真实声场一致的活塞传播子在实测 256 阵元上迭代)。
  - 定量:字母 F 物理重建与目标相关 **0.91**、笑脸 **0.90**(对齐方向);GS 模型 MSE 0.03~0.05;预览 PNG 中形状清晰可辨。
  - 方向:图像顶部放单亮点的判定实验 → 物理亮斑落在 z=+6.8cm(目标 +6.6cm),**图片"上"=+z、"右"=+x 成立**。
  - `method='paper'`(论文 §4 连续口径+角谱版)保留用于复现对拍:其设计模型偏理想,物理重建相关约 0.19、斑点较重——这是设计模型与离散阵列物理的固有差距,非 bug。
  - 过程中修复两个 bug(均已数值复核):① 网格→通道映射方向写反(`ph_flat[ch]` → `phases[ch]=ph_flat`);② 阵列 GS 的"暗区自由"约束错误(保留背景能量导致亮环,标准 GS 必须把暗区幅度约束为零)。
- 全部帧生成器输出形状/范围检查通过;相位量化(NaN→32 关闭)单测通过;`ports`/`--help` 正常。
- **未做**:真实硬件收发(等你上板;先 `ports` 确认串口,再 `focus --preview` 流程走一遍)。
