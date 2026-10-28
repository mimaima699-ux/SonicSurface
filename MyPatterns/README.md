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

RealSense 彩色画面 → MediaPipe 21 点找到指定指尖(默认 `index_tip`,第 8 点)
→ 对齐的深度图反投影成相机三维坐标 → 两层高度标定的刚体变换得到阵列
`(x,y,z)` → One-Euro 滤波 + 5 mm 死区 → 相位 → Modulator(200 Hz 调制)。
`y` 是指尖离阵列面的实际高度。红圈是目标点,绿色区域是关键点骨架生成的
近似 mask,不是逐像素深度分割。深度孔洞、指尖丢失或三维坐标越界时静默。

相机固定在阵列旁向上拍摄即可,不必让镜头平行于阵列。以阵列中心为原点,
`x/z` 沿阵列面两个轴,`y` 垂直阵列面向上。用尺子在第一层 `y=0.10m`
及第二层 `y=0.15m` 各准备 3x3 个位置:每层 `x/z` 取 -0.06,0,+0.06m。
每个 `x/z` 位置先采 10cm、再采 15cm,然后才移动到下一个位置。
按画面提示把**食指尖**放到每个三维位置,稳定后按空格;`r` 重采上一点,
`q` 放弃。标定窗口按 `V` 可轮换彩色、深度、并排预览,检查标记处
是否有有效指尖深度。每对采完会显示相机三维两点的直线距离;
它不是指尖相对阵列面的高度,不会单独据此拒绝采样。相机与阵列不平行由
最终三维变换处理。标定误差 RMS 超过 10mm 或单点超过 20mm 时不会保存。
若误差超限,终端会逐点显示残差,并在同目录写入
`hand_track_calib_3d_diagnostic.json`(原始相机三维点与提示位置)。重采前确认
10/15cm 量到的是**指尖**而非手掌,以及画面红圈处 `depth` 数值稳定。

```bash
py -3.13 MyPatterns/hand_track.py camtest --realsense                 # 看食指尖与 mask
py -3.13 MyPatterns/hand_track.py camtest --realsense --depth         # 彩色图 + 对齐深度热图
py -3.13 MyPatterns/hand_track.py camtest --realsense --pointcloud    # 另开可旋转的实时 3D 点云
py -3.13 MyPatterns/hand_track.py calibrate --realsense --dist 0.10 --height-step 0.05
py -3.13 MyPatterns/hand_track.py track --realsense --dry            # 看 x/y/z,不发射
py -3.13 MyPatterns/hand_track.py track --realsense --port COM8      # 三维坐标核对后连板
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
