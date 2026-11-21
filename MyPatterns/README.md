# MyPatterns — 自制 SonicSurface 图案脚本

完整 PowerShell 操作命令见 [命令清单.md](命令清单.md)，算法原理见 [原理详解.md](docs/原理详解.md)。

目录分工：`my_patterns.py` 是主入口；`image/` 存图片相位算法，`geometry/` 存三维图案算法；`assets/` 存样例图片和手势模型；`results/previews/` 存仿真预览，`results/*_comparison/` 存离线对照，`results/hand_tracking/` 存摄像头设置与标定；`tests/` 是离线测试，`docs/` 存原理说明。运行命令时从仓库根目录出发，示例中的输出路径均指向 `results/`。

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

导入 STL/OBJ/PLY 三维网格的 `mesh3d` 命令还需要 `pyvista`：

```powershell
.\ultrasound_solver\.venv\Scripts\python.exe -m pip install pyvista
```

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
| `image PATH [--method array/gradient/gradient_smooth/backprop/direct/paper]` | 图片图案；`backprop` 为单次反传，`direct` 为不依赖 GS 的相位直接优化 | `image ../Reproduction/smiley.png` |
| `image3d --layer PATH DEPTH_M --layer PATH DEPTH_M` | 多深度图片仿真；默认逐层用 2D GS 生成相位帧，`--mode static` 尝试单帧同时成像 | `image3d --layer MyPatterns/assets/demo_F.png 0.12 --layer Algorithms/EmissionPhasesAmpsForTargetSlice/A.png 0.16 --dry --save MyPatterns/results/previews/image3d_sequential.png` |
| `cup3d` | 稀疏 3D 杯子轮廓；焦点逐点扫描杯口、侧壁、杯底 | `cup3d --dry --save MyPatterns/results/previews/cup3d_preview.png` |
| `mesh3d PATH` | STL/OBJ/PLY 三维网格提取轮廓点并逐点计算焦点相位 | `mesh3d "Mech/two arrays support/leg.stl" --dry --save MyPatterns/results/previews/mesh3d_leg_preview.png` |
| `steer A` | 波束偏转 A 度(仅小角度) | `steer 8` |
| `burst X Z [--kind] [--dur] [--freq] [--duty]` | **按钮式强爆发**,支持任意位置与图案(`--kind focus/trap/vortex/twin/bottle`)。嗡鸣 ∝ f²,想安静用 `--freq 40` | `burst 0.03 0 --port COM7 --dist 0.12` |
| `tap X Z [--kind] [--count]` | **单击脉冲**:50ms 全功率,无调制→几乎无声,只有轻"嗒" | `tap 0.03 0 --count 3 --port COM7` |
| `hold X Z [--kind] [--freq] [--duty]` | **持续调制发射**:嗡到按回车为止。硬件应力低于悬浮,可长时间运行;进程须保持 | `hold --port COM7 --dist 0.12 --freq 100` |

**触觉精度三件套**:`--kind trap`(BFGS 焦点+上下环形零点,轴向边缘收窄,焦点保留 94%)+ **低 `--duty`**(0.3 左右,把衍射受限的轴向旁瓣压到触觉阈值以下——这是主旋钮)+ `--dist` 贴手掌实际高度。轴向感知范围物理极限 ~±1.5cm,trap+duty 组合可逼近该极限。

| `live [X Z] [--kind] [--freq] [--duty]` | **实时调节台**:持续发射中用键盘即时调参数,免停免重发——↑↓频率±10Hz、←→占空比±0.05、WASD 位置±5mm、R/F 高度±5mm、1~5 切换图案、回车退出 | `live --kind trap --freq 100 --port COM7` |
| `anim orbit/scan/spin/mpx/images` | 动画(Ctrl+C 停止并关发射) | `anim orbit --kind vortex --port COM4` |
| `off` | 关闭发射 | `off` |

公共参数:`--port`(COM 名或序号)、`--dist`(距离,默认 0.16 m)、`--preview`/`--save`/`--dry`、`--axis y|z|x`(预览切面)、`--fps`(动画帧率)。

### 图片梯度精修

`--method gradient` 先运行阵列感知 GS，再用 L-BFGS-B 优化笔画均匀度和背景亮斑。会比较 1 个确定性反传起点和 2 个随机起点，在独立的 96×96 网格上按 FPGA 实际 32 档相位评价；只有弱笔画、笔画均匀度和背景均值未明显退化时才选亮斑更低的候选，否则回退到原 GS。默认 `array` 的行为不变。

`--method gradient_smooth` 在该结果上增加亮区相邻采样点的幅度差惩罚，继续筛选亮区更平滑的候选；最低笔画亮度和背景亮斑仍有门槛。A 的离线 128×128 验证网格上，原 GS → 平滑精修的 `stroke_texture` 为 0.074 → 0.072，`stroke_cv` 为 0.264 → 0.249，背景最亮 1% 为 0.727 → 0.635，背景均值则为 0.180 → 0.193。改善有限，颗粒仍可见；指标是模型预测，不是实物测量。

离线对照（不打开串口，不向板卡发送）：

```powershell
cd D:\Mima\SonicSurface
.\ultrasound_solver\.venv\Scripts\python.exe MyPatterns\compare_image_methods.py
```

上面的比较脚本默认将五种算法的对照图与指标 JSON 写入 `MyPatterns/results/direct_comparison/`；`MyPatterns/results/gradient_comparison/` 保存的是此前的梯度算法对照结果。指标使用原 GS 笔画平均幅度作为共同尺度：`stroke_p10` 越高表示弱笔画越亮，`stroke_cv` 与 `stroke_texture` 越低表示笔画越均匀，`background_mean` 与 `background_top_1pct` 越低表示背景越干净。对照图在 128×128 网格评价，独立于 96×96 候选筛选网格。这些都是传播模型的预测，不是实物声场测量。

本机用 `--dry` 测得单张 A：原 GS 从命令启动到算完约 1.1 秒，`gradient_smooth` 约 2.6 秒；包含预览图保存还会更久。数值随电脑和图片变化。

`image --quant-refine` 在图案生成后先按板卡 32 档量化，再逐通道尝试相邻 ±1 档；只有加权图像损失下降、弱笔画和背景约束通过才接受。默认关闭，可与各 `--method` 组合。一次 256 通道扫描做 512 次候选评估，本机在 64×64 模型上约 44–50 ms。`demo_F.png` 的量化损失 0.1368→0.1093、背景最亮 1% 0.571→0.468，弱笔画 0.601→0.595；这些都是模型预测，不是实物测量。

### 不使用 GS 的图片算法

`--method backprop` 把目标亮度当作目标复声场，做一次共轭反传并取 256 路相位。`--method direct` 从该相位出发，直接用 L-BFGS-B 最小化目标平面的亮区误差、暗区亮斑及笔画纹理；两者都不调用 GS。板卡发送前仍会将相位量化到 32 档。默认仍为 `array`（GS）。

离线对照命令（不打开串口）：

```powershell
cd D:\Mima\SonicSurface
.\ultrasound_solver\.venv\Scripts\python.exe MyPatterns\compare_image_methods.py Algorithms\EmissionPhasesAmpsForTargetSlice\A.png MyPatterns\assets\demo_F.png Algorithms\EmissionPhasesAmpsForTargetSlice\circle.png Algorithms\EmissionPhasesAmpsForTargetSlice\curve.png Reproduction\smiley.png
```

结果保存在 `MyPatterns/results/direct_comparison/`。下表是本机一次运行中“读图、构建传播矩阵、生成相位”的耗时，不含进程启动和后续评估、绘图；各方法均在同一 128×128 独立网格上用 32 档相位评估，幅度统一以原 GS 笔画平均值为 1。MSE 越低越接近目标；笔画 CV 和背景最亮 1% 越低越好，但三者不能互相替代。

| 图片 | GS 时间 / MSE | 单次反传时间 / MSE | 直接优化时间 / MSE | GS → 直接优化：笔画 CV / 背景最亮 1% |
|---|---:|---:|---:|---:|
| A | 0.15 s / 0.055 | 0.11 s / 0.111 | 0.25 s / 0.064 | 0.264 → 0.176 / 0.727 → 0.568 |
| F | 0.17 s / 0.037 | 0.11 s / 0.073 | 0.21 s / 0.046 | 0.259 → 0.141 / 0.700 → 0.539 |
| 圆环 | 0.15 s / 0.084 | 0.11 s / 0.089 | 0.21 s / 0.078 | 0.202 → 0.121 / 0.858 → 0.626 |
| 曲线 | 0.15 s / 0.029 | 0.11 s / 0.031 | 0.23 s / 0.031 | 0.211 → 0.118 / 0.695 → 0.512 |
| 笑脸 | 0.15 s / 0.064 | 0.11 s / 0.116 | 0.22 s / 0.084 | 0.210 → 0.176 / 0.819 → 0.667 |

直接优化的笔画更连续、最亮杂波更低，但多数图的整体 MSE 和背景平均值较差；单次反传虽快，不适合直接替代 GS。这些结果只是当前传播模型的预测，尚无实测声场验证。计时依赖硬件和运行环境，完整数值请看对应 JSON。

### 多深度图片试验

普通 `image` 的默认算法仍是 `array`（GS）。`image3d` 的默认 `sequential` 模式给每个深度的图片独立计算一帧 GS 相位，然后逐帧切换。每帧都在**所有指定深度**上离线评估，以观察图案泄漏。`--mode static` 则用一组相位直接优化所有深度平面；目前只适合研究单帧多层串扰。

```powershell
cd D:\Mima\SonicSurface
.\ultrasound_solver\.venv\Scripts\python.exe MyPatterns\my_patterns.py image3d --layer MyPatterns\assets\demo_F.png 0.12 --layer Algorithms\EmissionPhasesAmpsForTargetSlice\A.png 0.16 --dry --save MyPatterns\results\previews\image3d_sequential.png
.\ultrasound_solver\.venv\Scripts\python.exe MyPatterns\my_patterns.py image3d --layer MyPatterns\assets\demo_F.png 0.12 --layer Algorithms\EmissionPhasesAmpsForTargetSlice\A.png 0.16 --mode static --dry --save MyPatterns\results\previews\image3d_example.png
```

`sequential` 输出目标/各深度实得声场的对照图和同名 JSON；`static` 还输出 `_side.png` 侧视图。两种模式都使用 FPGA 对应的 32 档相位量化后评价。2026-10-20 的 F@12 cm、A@16 cm 离线试算：单焦点的另一深度幅度约为本层的 22%/29%，但密集图片的跨层笔画幅度仍达本层约 88%/90%。`sequential` 生成两帧 GS 约 0.3-0.4 秒，本层 MSE 为 0.050/0.034。`static` 优化约 0.13 秒，MSE 为 0.194/0.101，侧视可见明显轴向延伸。**这组结果不能证明可分辨的 3D 触觉**；真实深度分辨率需要声场测量和手部实验。少量焦点的深度分离效果明显好于密集图片，后续触觉方案应从稀疏焦点路径和手的位置跟踪着手。

可显式指定 `--port COM8 --dur 3 --fps 20` 播放 `sequential` 的两帧，结束时自动发送关闭帧。完整串口相位帧在 230400 baud 下理论传输约 11 ms，故限制在 30 fps；此模式没有 200 Hz 幅度调制，触觉效果可能很弱。`static` 可以用 `--mod 200 --dur 3 --port COM8` 做限时调制，但仿真质量较差。先用 `--dry` 查看指标，且设备实际端口须以当次连接情况为准。

### 立体杯子轮廓原型

`cup3d` 将杯口圆环放在默认 16 cm，较小的杯底圆环放在 10 cm，并用四组中间焦点标出侧壁。它逐点生成 24 帧单焦点相位；每一帧仍只发一个焦点。`--radius`、`--bottom`、`--top` 可改变形状。先做离线仿真：

```powershell
cd D:\Mima\SonicSurface
.\ultrasound_solver\.venv\Scripts\python.exe MyPatterns\my_patterns.py cup3d --dry --save MyPatterns\results\previews\cup3d_preview.png
```

预览的 3D 点图是真正要扫描的路径；杯口/杯底/侧面热图是**24 帧的时间平均声强**，并不是同一时刻有一个杯子。默认 20 fps 一圈至少 1.20 秒，串口理论纯传输下限约 0.27 秒；模型中平均图的最高强度约为单个焦点峰值的 8.9%。当前方案未实现 200 Hz 调制或手部追踪，不能保证人能摸出立体杯形。代码只在显式指定 `--port COM8` 时允许实物扫描，播放 `--dur` 秒后关闭通道；仿真以外的触感尚未验证。要获得连续、可定位的 3D 触觉，需要进一步改固件支持高速焦点序列和调制，并根据手的位置只渲染其附近的轮廓。

任意三维模型需要提供 **STL、OBJ 或 PLY 文件**；仅给一张普通照片并没有可靠的深度几何。模型按标准 Z 轴朝上读取，横向 X/Y 映射到阵列 X/Z，模型竖直 Z 映射成离阵深度 Y。`mesh3d` 等比缩放到 `--width 0.06 --height 0.06` 以内，最低点放在 `--near 0.10` 米。优先提取网格边界/锐边；若锐边不足则使用表面顶点，选出默认 24 个焦点。路径是稀疏近似，不会复现原模型的所有细节。

```powershell
.\ultrasound_solver\.venv\Scripts\python.exe MyPatterns\my_patterns.py mesh3d "Mech\two arrays support\leg.stl" --near 0.10 --width 0.06 --height 0.06 --points 24 --dry --save MyPatterns\results\previews\mesh3d_leg_preview.png
```

该仓库 STL 的离线试跑成功：24 帧、20 fps 一圈至少 1.20 秒，时间平均最大强度约为单焦点峰值的 10.5%。每次生成的 JSON 记录物理边界、点坐标、帧数、周期及强度。`mesh3d` 与 `cup3d` 一样，只有显式指定 `--port` 才允许实物播放；`--dry` 完全不打开串口。

单张 F 的无 GS 仿真预览：

```powershell
.\ultrasound_solver\.venv\Scripts\python.exe MyPatterns\my_patterns.py image MyPatterns\assets\demo_F.png --method direct --dry --save MyPatterns\results\direct_comparison\F_preview.png
```

单张图片先离线预览：

```powershell
.\ultrasound_solver\.venv\Scripts\python.exe MyPatterns\my_patterns.py image Algorithms\EmissionPhasesAmpsForTargetSlice\A.png --method gradient_smooth --dry --save MyPatterns\results\gradient_comparison\A_preview.png
```

## 线程安全与后台调制(Modulator)

2026-10-22 起 `Board` 与调制逻辑支持多线程并发,这是手掌跟踪的前置改造:

- **Board 串口写加锁**(可重入),`send`/`send_commit`/`off`/`close` 可跨线程并发调用。
  固件 `Distribute.vhd` 中 253 会打断未完成的图案帧并复位字节计数,跟踪场景下
  "主循环 send + 调制线程 253" 不加锁必然协议错乱——这是硬需求,不是保险。
- **写超时容错**:完整相位帧超时后重发一次;单字节 commit 不重发(可能已到达 FPGA,
  重发会反转双缓冲)。调制线程失败时尽力发送 off 并向主循环报告错误。
- **Modulator 类**:图案/关闭预装双缓冲,后台线程循环 253 产生调制;
  `set_pattern(ph)` 在一把锁内更新图案和关闭图两帧,防止下一次 commit 切回旧图案;
  `stop()` 幂等(关断帧只发一次)。换图约需两个完整串口帧,实际延迟取决于串口与相机。
  `burst`/`hold`/`live` 已改为复用它,行为不变;等待以 sleep 为主(>5ms 段),低频不再占满一个核。
- **温启动**:`phases_trap(..., x0=上一帧相位)` 与 `_gs_from_model(..., x0=...)`;
  跟踪 trap 限制 30 次迭代并按 5 mm 网格缓存;GS 默认从目标场共轭反传相位初始化,
  `anim images` 的下一张图片从上一张相位温启动。

离线测试:`py -3.13 -X utf8 -m unittest discover -s MyPatterns/tests -t MyPatterns -p "test_board_modulator.py"`(假串口,验证锁/重发/调制/温启动)。

## 手部指尖跟踪(hand_track.py)

### 阵列自动定位（俯视 RealSense）

摄像头直接识别 16x16 换能器圆孔，利用已知 10.5mm 阵元间距和相机内参
确定阵元编号，优先用对齐深度图拟合阵列三维姿态；有效板面深度不足时回退到
彩色图 PnP。不再需要手持指尖采 18 个点。圆形阵列完全对称，无法从
圆孔本身判断硬件 +X/+Z：首次打开 `boardcalibrate`，用 A/D 切换八种方向，
确认画面箭头指向硬件实际正轴、绿色框贴合阵列四边且显示
`pose stable 2/2` 后按 C 保存。另存
`results/hand_tracking/hand_track_board_calib.json`，原手工标定不被覆盖。
不要只根据画面上下左右猜硬件正轴；先核对板角/通道定义。

```powershell
.\ultrasound_solver\.venv\Scripts\python.exe -X utf8 MyPatterns\hand_track.py boardcalibrate --realsense
.\ultrasound_solver\.venv\Scripts\python.exe -X utf8 MyPatterns\hand_track.py track --realsense --auto-board --dry --max-seconds 10
.\ultrasound_solver\.venv\Scripts\python.exe -X utf8 MyPatterns\hand_track.py track --realsense --auto-board --port COM8 --duty 0.2 --lost 0.2 --max-seconds 10
```

若圆孔自动识别反复失败，可改用
以下四角深度模式：

```powershell
.\ultrasound_solver\.venv\Scripts\python.exe -X utf8 MyPatterns\hand_track.py boardcalibrate --realsense --manual-corners
```

依次左键点阵列最外侧阵元的
左上、右上、右下、左下中心，右键可撤销最后一点。深度图会拟合板面姿态；
确认绿色框、轴方向和 `pose stable 2/2` 后按 C。此方式保存四角像素位置，
后续 `--auto-board` 启动会用深度复查板面，但相机和板子都必须保持固定；
移动相机后需重新点四角。若深度图没有至少 50 个有效板面点，会拒绝保存。
标定窗口按 S 可保存当前彩色图和对齐深度到
`results/hand_tracking/board_depth_diagnostic.npz` 供排查。

跟踪画面中红圈是识别出的当前目标，紫圈是当前发送相位对应的**计算焦点**，
青色十字是该焦点沿板面法线落在阵列上的垂足，橙框是阵列阵元范围。
普通跟踪窗口会画出手部骨架；左键点击任意可见关节或指尖即可切换发射目标，
顶部 `target` 显示当前部位。切换时旧焦点先关断，再用新部位的深度计算焦点。
也可在启动时用 `--target middle_tip` 等指定初始部位；`palm` 为腕部与
中指根部的中点。板面三维标定的坐标变换适用于这些部位，无需按手指重标。
绿色框是**指尖当前高度**的 x/z 允许范围，不能拿板面橙框与空中的指尖
直接比较；红叉及 `outside x/y/z` 表示本帧越界关断，`no_depth` 表示相机
没有取得可靠指尖深度。显示 `ACTIVE` 但手感较弱时，先检查声场偏轴衰减。
当手移动到画面边缘，程序会尝试给画面补边再做一次 MediaPipe 检测。若整手
关键点暂时丢失，已确认的指尖可由局部图像和 RealSense 深度短时续跟；匹配
失败、深度失效、相机无帧或目标出界都会立即静默。只有一根手指从开始就入镜
时，可加 `--manual-tip`，在普通 `track` 预览里左键点击可见指尖；连续三帧
匹配成功且指尖比背景更近，才会激活焦点。此时显示 `unknown (manual tip)`：
单根手指无法可靠判断是哪根手指，也无法判断点击的物体一定是手指。
误点后按 `r` 立刻清除并静默；
先用 `--dry --manual-tip` 检查红圈与紫圈，确认后正式发射仍须显式加
`--manual-tip`；`--experimental` 禁用身份未知的局部模式。为获得最稳的识别，
仍建议移动或抬高相机、扩大视野，让手掌与目标手指都留在画面内。
手动四角标定会用 RealSense 彩色镜头内参在去畸变射线空间插值阵元；
旧四角文件在启动时也会按实时内参重建网格并复核深度姿态。
焦点标记是模型预测，不是声压实测。普通跟踪默认允许 y=7-25cm，已经
超过 20cm；但当前相机约高于板面 28.5cm，指尖接近 21.5cm 高时，
正对相机的深度可能已小于 D405 的 7cm 最小值，出现 `no_depth`。
要稳定跟踪更高的位置需抬高相机并重新标定；`--height-max 0.27` 只能
改变软件门限，无法扩大相机近距深度能力。
指尖深度会优先选择遮罩内最近的可靠深度层，减少背景被误认为指尖。

阵元间距 10.5mm 大于 40kHz 波长 8.65mm：偏轴聚焦会有栅瓣。
例如模拟 `x=+5cm,y=12cm,z=0` 时，在 `x≈-7cm` 有约 48% 主焦点声压幅度的次峰，
因此远离中心感到两个强点不一定是坐标标定错误。可先在中心附近比较。
要独立于相机核对真实声学方向，可运行限时调制焦点检查：

```powershell
.\ultrasound_solver\.venv\Scripts\python.exe -X utf8 MyPatterns\my_patterns.py focusprobe --dry
.\ultrasound_solver\.venv\Scripts\python.exe -X utf8 MyPatterns\my_patterns.py focusprobe --port COM8 --dist 0.12 --offset 0.04 --dur 8 --freq 200 --duty 0.2
```

实物命令依次检查中心、+X、-X、+Z、-Z。每步按回车后以 200Hz/20% 调制
发射 8 秒，随后自动关闭；输入 q 回车可结束。可把 `--dur` 提至最多 15 秒，
`--static` 则恢复连续静态对照。记录每步最强触点相对板中心在相机画面的
上/下/左/右方，以及是否有对侧第二峰；不要拿 `boardcalibrate` 的箭头
当作独立依据，因为那是手动选定的轴方向。中心正确但四轴整体旋转或镜像，
应重新确认板轴方向；方向正确但对侧另有峰，则优先排查栅瓣。
**实验性偏轴抑制**：周期阵列在远场有角度混叠，固定的 10.5mm 间距
无法靠相位算法实现整个空间“只有一个峰”。`focusprobe --algorithm suppress`
只针对目标高度附近 ±2cm 的有限三维区域，迭代寻找最强副峰，在副峰周围
加抑制约束，同时保留目标点能量；最终按板卡 32 档量化，并在 x/z ±12cm、
目标高度 ±2cm 的独立密网格复核，要求最强点仍在目标附近。复核无改善时
自动回退普通焦点。离线模型中，y=12cm、x/z=±4cm 的四个角点最强远处峰/
目标峰约由 44–46% 降到 35–37%，但目标声压只保留约 81–83%；
x/z=+6cm 的角点未通过验收，会回退。该输出只预测理想阵列的声压，
不代表实物声场或手指触觉；更远处、其它高度也没有单峰保证。
单个角点可先无板试算，再用同一位置分别运行普通和实验算法：

```powershell
.\ultrasound_solver\.venv\Scripts\python.exe -X utf8 MyPatterns\my_patterns.py focusprobe --algorithm suppress --point 0.04 0.04 --dry
.\ultrasound_solver\.venv\Scripts\python.exe -X utf8 MyPatterns\my_patterns.py focusprobe --point 0.04 0.04 --port COM8
.\ultrasound_solver\.venv\Scripts\python.exe -X utf8 MyPatterns\my_patterns.py focusprobe --algorithm suppress --point 0.04 0.04 --port COM8
```

三条命令使用同一 200Hz/20% 调制、12cm 焦点高度、每次 8 秒自动关断。
比较目标点和对侧的触感；固件相位校正、阵元差异和反射未进入模型。
用 `--span 0.03` 将发射限制到中心 x/z ±3cm；默认仍为 ±8cm。
可用 `--lost 0.35` 减少短暂丢手引起的关断；离开画面仍会自动关断。

每次 `--auto-board` 启动都会先从实时画面重新识别阵列；需移开手并露出全部
四角。识别不足、相邻帧姿态不稳定、姿态残差超限、序列号或画面方向不符时不发射。进入跟踪后
应固定相机和阵列；暂不在手遮挡阵列时连续重估板面姿态。首次上板前先用
`--dry` 核对指尖向硬件 +X/+Z 移动时坐标方向确实正确。

### 手工三维标定

RealSense 彩色画面 → MediaPipe 21 点找到指定指尖(默认 `index_tip`,第 8 点)
→ 对齐的深度图反投影成相机三维坐标 → 两层高度标定的刚体变换得到阵列
`(x,y,z)` → One-Euro 滤波 + 5 mm 死区 → 相位 → Modulator(200 Hz 调制)。
`y` 是指尖离阵列面的实际高度。红圈是目标点,绿色区域是关键点骨架生成的
近似 mask,不是逐像素深度分割。深度孔洞、指尖丢失或三维坐标越界时静默。

相机可从阵列旁或上方拍摄,不必让镜头平行于阵列。以阵列中心为原点,
`x/z` 沿阵列面两个轴,`y` 垂直阵列面向上。用尺子在第一层 `y=0.10m`
及第二层 `y=0.15m` 各准备 3x3 个位置:每层 `x/z` 取 -0.06,0,+0.06m。
每个 `x/z` 位置先采 10cm、再采 15cm,然后才移动到下一个位置。
按画面提示把**食指尖**放到每个三维位置,稳定后按空格;`r` 重采上一点,
`q` 放弃。标定窗口按 `V` 可轮换彩色、深度、并排预览,检查标记处
是否有有效指尖深度。采样时会检查与同排、相邻排及另一高度已采点的距离;
`CHECK` 表示三维直线距离与标称步长相差超过 5mm,或移动方向与其他同轴
步长相差超过 20 度。默认网格的高度步长标称 50mm,相邻行列标称 60mm;这些都是
诊断提示,不会自动丢弃采样。第 18 点后进入复核画面,显示 RMS、最大误差
及选中点残差;按 A/D 选点、W 选最大残差点、R 重采选中点、回车验收。
误差仍超限时回车会留在复核画面。复核时按 Q 结束：达标保存正式标定，
超限保存诊断；采样过程中按 Q 才会放弃。
三维直线距离不是指尖相对阵列面的高度。相机与阵列不平行由
最终三维变换处理。标定误差 RMS 超过 12mm 或单点超过 30mm 时不会保存。
若误差超限,终端会逐点显示残差,并在同目录写入
`hand_track_calib_3d_diagnostic.json`(原始相机三维点与提示位置)。重采前确认
10/15cm 量到的是**指尖**而非手掌,以及画面红圈处 `depth` 数值稳定。
相机改为俯视且画面侧转时,先运行 `camtest --realsense --depth --rotate ccw`
检查画面方向并按 ESC 保存;若方向相反,改用 `--rotate cw`。旋转会同时作用于
彩色和对齐深度,三维反投影仍使用传感器原始坐标。移动或旋转相机后必须重新
标定;旧三维标定即使 RMS 合格也不再代表新的安装位置。

```bash
py -3.13 MyPatterns/hand_track.py camtest --realsense                 # 看食指尖与 mask
py -3.13 MyPatterns/hand_track.py camtest --realsense --depth         # 彩色图 + 对齐深度热图
py -3.13 MyPatterns/hand_track.py camtest --realsense --pointcloud    # 另开可旋转的实时 3D 点云
py -3.13 MyPatterns/hand_track.py calibrate --realsense --dist 0.10 --height-step 0.05
py -3.13 MyPatterns/hand_track.py track --realsense --dry            # 看 x/y/z,不发射
py -3.13 MyPatterns/hand_track.py track --realsense --port COM8 --duty 0.2 --lost 0.2 --max-seconds 10  # 短时连板
py -3.13 MyPatterns/hand_track.py track --sim --dry   # 合成圆路径离线回路(无需摄像头)
```

失败标定若仅用于短时中心区域试验,可显式加 `--experimental` 读取
`hand_track_calib_3d_diagnostic.json`。程序重新核对原始点位,仅接受
RMS ≤15mm、最大误差 ≤30mm,并强制 `focus`、中心 x/z ±3cm、y 9-16cm、
`--duty` ≤0.2、丢手 0.2s 关断、最长 20s 自动结束;不会把诊断另存为正式标定。
实验模式默认 2mm 相位更新死区; `--experimental-span 0.05` 可将 x/z
范围显式扩大到中心 ±5cm,仍不超过标定网格的 ±6cm。
如需检查整块阵列范围,可改用 `--experimental-span 0.08
--experimental-height 0.08 0.18`;这会进入原始标定点外的区域,先用 `--dry`
确认画面坐标随真实手指移动方向和高度变化,再连板试验。
先确认 RealSense 仍识别为 **D405 且序列号与诊断一致**,并完成 dry run:

```powershell
.\ultrasound_solver\.venv\Scripts\python.exe -X utf8 MyPatterns\hand_track.py track --realsense --experimental --dry --duty 0.2 --max-seconds 10
.\ultrasound_solver\.venv\Scripts\python.exe -X utf8 MyPatterns\hand_track.py track --realsense --experimental --port COM8 --duty 0.2 --max-seconds 20
```

需要测试横向跟随时,两条命令都加 `--experimental-span 0.05`。

跟踪画面显示 `ACTIVE (unchanged phase)` 时,图案仍由调制线程持续发射;
结束统计会分别给出相位更新次数和焦点激活秒数,相位更新不是 200Hz 脉冲数。
若焦点反复
静默,查看末尾逐帧诊断: `no_hand` 是手势未识别,`no_depth` 是指尖深度失效,
`outside_x/y/z` 是超出实验范围,`steady` 是范围内但移动未超过当前死区。
窗口会列出所有越界轴,末尾的“越界轴次数”分别统计 x/y/z。

- 新标定存 `MyPatterns/results/hand_tracking/hand_track_calib_3d.json`。旧二维
  `hand_track_calib.json` 保留作历史记录,三维跟踪会拒绝加载它。移动相机或
  改变流分辨率后必须重标。正式发射范围: `x/z` ±8cm、`y` 7-20cm。
- **摄像头整定**:画面过暗/视场像被放大时先跑 `camtest`(键盘调曝光/亮度/增益/变焦/分辨率,
  ESC 存 `MyPatterns/results/hand_tracking/hand_track_cam.json`,calibrate/track 自动套用)。常见坑:默认 640x480 从 16:9 传感器
  裁切 → 按 R 切 1280x720 恢复全视场;手动低曝光 → 按 X 开自动曝光。改分辨率/变焦后需重新标定。
- RealSense 红外画面整体忽明忽暗时,可按 X 关闭自动曝光,查看 `auto=OFF`,再用 W/S
  以 1000 µs 步进调曝光。按 W/S 也会切到手动;ESC 会保存曝光状态供后续标定/跟踪恢复。
- **Intel RealSense D405**: `camtest` 仍可仅开彩色流或回退红外流;
  `calibrate`/`track` 必须同时启动彩色 BGR8 与深度 Z16,并将深度对齐到彩色流。
  设备不提供两种流或深度无效时不会发射。首次 camtest ESC 后 JSON 记住
  realsense,但示例仍显式带 `--realsense`。
- `camtest --depth` 的右侧热图显示距相机 7-50cm 的深度,黑色代表无有效深度;
  红圈旁的 `tip depth` 是相机到指尖的距离。`--pointcloud` 再打开鼠标可旋转的
  三维点云窗口,轴为相机坐标,不是阵列 `x/y/z`;这两个预览都不连超声板卡。
  相机窗口可拖拽放大或最大化,按 `F` 切全屏;深度模式按 `V` 在并排、彩色、
  深度单画面之间切换。标定与跟踪窗口也支持拖拽放大和 `F` 全屏。
- 安全:丢手 >0.5 s 或越界(±8 cm)→ 立即换 off 图案静默(调制线程保持,手回来即恢复);
  独立监督线程在相机读取阻塞时也执行丢手关断;Ctrl+C → off 退出。
  开始跟踪前阵列静默(预装 off),检测到手才发射。
- One-Euro 参数已按米制速度整定(`min_cutoff=2 Hz, beta=30`):静止压 2 mm 抖动,
  ~80 mm/s 手速时滞后 ~3 mm(离线仿真值)。
- 依赖 `opencv-python` + `mediapipe`(py -3.13 与 solver venv 均已安装)。
- 离线测试:`py -3.13 -X utf8 -m unittest discover -s MyPatterns/tests -t MyPatterns -p "test_hand_track.py"`(三维标定/深度孔洞/控制回路,无需摄像头)。
- **已验证**:真实 D405 连续读取 1280x720 对齐彩色+深度帧。
  **待实测**:18 点三维标定与上板效果;先在 `--dry`
  预览中分别上下移动手指,确认 `y` 随之变化且 `x/z` 方向正确。

## 接收器与示波器测量(receiver_probe.py)

图中的银色圆柱顶部黑色网面是接收面。此工具让你在 RealSense 画面中用鼠标框选**黑面和一圈银色边框**，跟踪其中心并用三维标定换算到阵列坐标；它不识别接收器型号，也不自动读取示波器。鳄鱼夹和金属筒侧面不能当作敏感面中心。每个位置都让接收面朝向阵列，固定接收器与相机；示波器读 40 kHz 载波的 Vpp，保持同一探头、量程和设置，比较同一焦点目标在不同接收位置的 Vpp。Vpp 比值可比较相对响应；未做探头灵敏度标定时不能换算绝对声压。

先在普通照片上检查选框中心（仅像素，无深度）：

```powershell
python -X utf8 MyPatterns\receiver_probe.py --image "D:\Mima\SonicSurface\receiver.jpeg"
```

实时测量需要有效的 RealSense 三维标定。`--target` 是固定发射焦点 `(x,y,z)`，单位米；下面示例默认不连接板卡，按 `e` 后只记录约 5 秒的接收器位置，并在终端输入外部发射时读到的 Vpp。外部发射由你自行控制。

```powershell
python -X utf8 MyPatterns\receiver_probe.py --target 0 0.12 0
```

若显式加 `--port COM8`，按 `e` 时程序发送静态焦点，默认 5 秒后关闭，再提示输入示波器 Vpp；按 `r` 重新框选，按 `q` 退出并关闭发射。测量时深度或跟踪丢失、接收器移动超过 8 mm 会取消记录并关闭本程序控制的发射。CSV 默认保存在 `results/hand_tracking/receiver_probe.csv`，包含预设焦点、接收面阵列坐标、位置偏差、Vpp、深度来源与跟踪分数。把接收器移到多个位置，在相同 `--target` 下重复按 `e` 才能比较副峰和主峰。接收器扫描允许 x/z 到 ±20 cm、y 为 5-30 cm，便于检查板外副峰；仍要求相机能看到接收面且深度有效。发射焦点目标仍限制在阵列工作范围内。

```powershell
python -X utf8 MyPatterns\receiver_probe.py --target 0.03 0.12 0 --port COM8 --duration 5
```

黑色网面没有可靠深度时，可加 `--allow-rim-depth` 用银色边框深度估计，并在 CSV 标为 `rim_estimate`；若边框深度也不一致则拒绝记录。这种几何估计尤其不能证明焦点准确，最终判断需看示波器读数。本工具尚未与真实相机、板卡和示波器联测。

扫描后用 `python -X utf8 MyPatterns\receiver_probe.py --analyze MyPatterns\results\hand_tracking\receiver_probe.csv` 汇总每个固定目标已采样的最大 Vpp 位置、距目标偏差，以及远区（离目标至少 20 mm）与近区（10 mm 内）最大 Vpp 比。至少采 5 点且近远区都有样本；报告只代表已测点，若要排除其他副峰，还需覆盖关注的二维平面和不同高度。

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
3. 阵元间距 10.5 mm > λ/2(4.3 mm):偏轴聚焦也会有栅瓣;中心焦点通常更干净。
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
