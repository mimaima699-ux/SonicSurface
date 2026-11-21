"""Build a paper presentation using the supplied PPTX's visual conventions."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "光泳陷阱体显示_论文介绍.pptx"
SOURCE = "Smalley et al., Nature 553, 486-490 (2018), doi:10.1038/nature25176"
BLACK = RGBColor(35, 35, 35)
GRAY = RGBColor(92, 92, 92)

prs = Presentation()
prs.slide_width = Inches(13.333333)
prs.slide_height = Inches(7.5)

prs.core_properties.title = "A photophoretic-trap volumetric display"
prs.core_properties.subject = "Smalley et al. (2018) 论文介绍"
prs.core_properties.author = ""


def text(slide, x, y, w, h, value, size=20, bold=False, font="Microsoft YaHei",
         color=BLACK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.01)
    frame.margin_top = frame.margin_bottom = Inches(0.01)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    para = frame.paragraphs[0]
    para.alignment = align
    para.space_before = para.space_after = Pt(0)
    run = para.add_run()
    run.text = value
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def slide_with_title(title):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    text(slide, 0.58, 0.52, 12.0, 0.78, title, 31.1, True, "Times New Roman")
    return slide


def notes(slide, body):
    slide.notes_slide.notes_text_frame.text = f"Source: {SOURCE}\n\n{body}"


def figure(slide, number):
    image_path = ROOT / "tmp" / "photophoretic_figures" / f"figure{number}.png"
    from PIL import Image
    with Image.open(image_path) as image:
        ratio = image.width / image.height
    height = min(6.95, 12.25 / ratio)
    width = height * ratio
    x = (13.333333 - width) / 2
    y = (7.5 - height) / 2
    slide.shapes.add_picture(str(image_path), Inches(x), Inches(y),
                             width=Inches(width), height=Inches(height))


cover = slide_with_title("A photophoretic-trap volumetric display")
text(cover, 0.58, 2.35, 11.8, 0.75,
     "光泳陷阱体显示：用一颗微粒在空气中绘制彩色三维图像", 23)
text(cover, 0.58, 3.45, 11.8, 0.48,
     "D. E. Smalley 等  ·  Nature 553, 486–490 (2018)", 18)
text(cover, 0.58, 4.22, 11.8, 0.38, "doi:10.1038/nature25176", 15, color=GRAY)
notes(cover, "先给出主线：颗粒是像点，光负责捕获、移动和着色。")

s = slide_with_title("Why a free-space volumetric display?")
text(s, 0.72, 1.66, 11.9, 0.62, "问题：有限的二维出光面限制空中像点的可见方向。", 22)
text(s, 0.72, 2.72, 11.9, 0.9,
     "全息图中的像点仍要通过显示面的光线进入眼睛；从某些侧面看，像点会被出光口“裁掉”。", 20)
text(s, 0.72, 4.08, 11.9, 1.0,
     "作者的目标：让散射光与图像点位于同一位置，使图像可以绕看，也能围绕实体物体。", 20)
text(s, 0.72, 5.75, 11.9, 0.6,
     "关键词：体显示（volumetric display）与视角裁切（clipping / vignetting）", 18, color=GRAY)
notes(s, "对应论文第 486 页及 Extended Data Fig. 1。这里讨论的是有限孔径下的观看几何。")

s = slide_with_title("Operating principles of the OTD")
text(s, 0.72, 1.58, 11.9, 0.68, "光泳陷阱抓住颗粒，扫描陷阱让颗粒走过三维路径。", 22)
text(s, 0.72, 2.54, 11.9, 0.68, "405 nm 捕获光形成陷阱；微粒吸收光后受到与周围空气有关的热力。", 20)
text(s, 0.72, 3.54, 11.9, 0.68, "振镜与移动透镜改变陷阱位置，被困住的颗粒随之移动。", 20)
text(s, 0.72, 4.54, 11.9, 0.68, "同轴 RGB 激光照亮颗粒；它散射的光就是观察者看到的彩色点。", 20)
text(s, 0.72, 5.54, 11.9, 0.88,
     "快速重复路径后，视觉暂留把依次出现的点连成图像；任一瞬间只有一颗亮点。", 20)
notes(s, "对应论文 Fig. 1a。405 nm 光负责捕获，RGB 光负责照明；颗粒不是自身发光。")

s = prs.slides.add_slide(prs.slide_layouts[6])
figure(s, 1)
notes(s, "Fig. 1c 的图形为真实的快速循环扫描示例。图中 10 Hz 是图注中的表述，正文给出最高 12.8 fps。")

s = slide_with_title("Photophoretic trapping")
text(s, 0.72, 1.55, 11.9, 0.58, "光泳力：吸光颗粒受热不均，与空气分子交换动量后获得净力。", 21)
text(s, 0.72, 2.62, 11.9, 0.9,
     "它与常见的光镊不同：这里主要依靠热效应，适合操纵吸光的微米级颗粒。", 20)
text(s, 0.72, 3.98, 11.9, 1.05,
     "作者让透镜倾斜约 1°，利用球差和像散塑造焦点附近的明暗分布，从而形成可移动的捕获区域。", 20)
text(s, 0.72, 5.55, 11.9, 0.8,
     "球差、像散通常是成像误差；在这里，它们成为设计光阱的工具。", 19, color=GRAY)
notes(s, "对应论文第 486–487 页和 Methods。光泳机理在文中以热适应与辐射计效应解释；具体粒子形状使实际力学更复杂。")

s = slide_with_title("Viewing the image from several angles")
figure(s, 4)
text(s, 0.72, 6.12, 11.9, 0.54,
     "论文 Fig. 2a–d：蝶形图与多视角棱柱；侧视时，棱柱仍可见。", 18)
notes(s, "论文 Fig. 2a–d。蝶形图和棱柱图来自长曝光照片；这页展示的是观察几何。")

s = prs.slides.add_slide(prs.slide_layouts[6])
figure(s, 2)
notes(s, "论文 Fig. 2e–h：远距离投射、沙盘上方、围绕实体手臂。均为长曝光照片。")

s = prs.slides.add_slide(prs.slide_layouts[6])
figure(s, 3)
notes(s, "论文报告像点尺寸小于约 10 微米，演示约 1600 dpi。图 3 的照片为长曝光。")

s = slide_with_title("Results: separate the measurements")
text(s, 0.72, 1.54, 11.9, 0.75,
     "12.8 帧/秒：1,307 个顶点的简单矢量图，约 16,700 点/秒。", 21)
text(s, 0.72, 2.66, 11.9, 0.75,
     "约 10 μm：最小图像点尺寸；图像分辨率演示约 1,600 dpi。", 21)
text(s, 0.72, 3.78, 11.9, 0.75,
     "1,827 mm/s：单独优化速度实验中的最高颗粒线速度。", 21)
text(s, 0.72, 4.90, 11.9, 0.75,
     "17.2 小时：最长一次捕获保持时间，实验由研究人员终止。", 21)
text(s, 0.72, 6.05, 11.9, 0.5,
     "这些数据来自不同原型或不同测试条件，不能拼成一台设备的同时性能。", 18, color=GRAY)
notes(s, "对应论文 Table 1。速度上限来自独立测试，不等于 1307 点图形的实际绘制速度。")

s = slide_with_title("Real-time display versus long exposure")
text(s, 0.72, 1.60, 11.9, 0.75,
     "实时演示：Fig. 1c 的简单图形以 12.8 帧/秒循环绘制。", 22)
text(s, 0.72, 2.80, 11.9, 1.0,
     "复杂彩色照片：Fig. 2 和 Fig. 3 都用了长曝光，拍摄时间约 8–56 秒。", 20)
text(s, 0.72, 4.28, 11.9, 1.0,
     "当时的彩色系统需要在像点处等待绿色激光达到目标功率，拖慢了图像绘制。", 20)
text(s, 0.72, 5.78, 11.9, 0.65,
     "结论：论文验证了彩色、精细的空间成像，也验证了简单图形的实时视觉暂留。", 19, color=GRAY)
notes(s, "对应 Methods 第一段。Fig. 2a 16.2 s、2b–d 8.4 s、2h 56.4 s、3d 19 s。长曝光图像并非实时视频帧。")

s = slide_with_title("Limitations and relevance to SonicSurface")
text(s, 0.72, 1.55, 11.9, 0.72,
     "瓶颈：单颗粒扫描限制图像复杂度；颗粒状态和气流影响陷阱稳定性。", 20)
text(s, 0.72, 2.70, 11.9, 0.72,
     "后续方向：多颗粒并行、空间光调制器，以及更稳健的颗粒捕获。", 20)
text(s, 0.72, 4.10, 11.9, 0.95,
     "对 SonicSurface 的启发：把三维轨迹离散成点，预计算各点的相位，再依次移动触觉焦点。", 20)
text(s, 0.72, 5.65, 11.9, 0.82,
     "光泳颗粒显示与超声触觉焦点使用不同的物理机制；可迁移的是轨迹设计与时序控制。", 18, color=GRAY)
notes(s, "论文第 489 页讨论扫描速度、气流、陷阱差异及照明光溢出。SonicSurface 项目联系基于本地代码与控制流程，是应用启发，不是原论文结果。")

prs.save(OUTPUT)
print(f"Created {OUTPUT} ({len(prs.slides)} slides)")
