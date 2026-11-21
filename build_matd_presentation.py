"""Build a plain-language, editable presentation of Hirayama et al. (2019)."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "MATD_论文介绍_白底黑字.pptx"
FONT = "Microsoft YaHei"
BLACK = RGBColor(0, 0, 0)
GRAY = RGBColor(90, 90, 90)
WHITE = RGBColor(255, 255, 255)
SOURCE = "Hirayama et al., Nature 575, 320-323 (2019), doi:10.1038/s41586-019-1739-5"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
prs.core_properties.title = "用超声在空中画出可触摸的三维图形"
prs.core_properties.subject = "MATD 论文介绍"
prs.core_properties.author = ""


def text_box(slide, x, y, w, h, text, size=21, bold=False, color=BLACK,
             align=PP_ALIGN.LEFT, margin=0):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_before = Pt(0)
    p.space_after = Pt(0)
    run = p.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shape


def base_slide(title, source=SOURCE):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    text_box(slide, 0.75, 0.43, 11.9, 0.68, title, 30, True)
    if source:
        text_box(slide, 0.76, 7.06, 11.8, 0.22, source, 9, color=GRAY)
    text_box(slide, 12.32, 7.03, 0.28, 0.24, str(len(prs.slides)), 9,
             color=GRAY, align=PP_ALIGN.RIGHT)
    return slide


def bullets(slide, lines, start_y=1.52, step=1.07, size=21):
    for i, line in enumerate(lines):
        text_box(slide, 0.93, start_y + i * step, 11.6, step - 0.11,
                 f"•  {line}", size)


def note(slide, content):
    # Notes stay editable and give the speaker a little more context.
    slide.notes_slide.notes_text_frame.text = content


slide = base_slide("用超声在空中画出可触摸的三维图形")
text_box(slide, 0.9, 2.1, 11.5, 1.05,
         "一颗被超声悬浮的小球，同时承担三维显示、声音与触觉反馈", 25)
text_box(slide, 0.9, 3.66, 11.5, 0.85,
         "A volumetric display for visual, tactile and audio presentation using acoustic trapping", 19)
text_box(slide, 0.9, 4.65, 11.5, 0.5,
         "Ryuji Hirayama 等，Nature，2019", 18)
note(slide, "开场可以先展示论文的补充视频，再指出画面中的线条实际只由一颗小球高速扫描形成。")

slide = base_slide("01  这篇论文要解决什么问题")
bullets(slide, [
    "体积显示希望让图形真正出现在三维空间，而不依赖眼镜。",
    "很多方案主要提供视觉；此前的声悬浮显示，粒子运动速度和画面复杂度有限。",
    "作者的目标：用同一种物理机制，同时提供视觉、可听声音和空中触觉。",
    "核心问题是：声学陷阱能否快到形成稳定的视觉图形，还能分出资源做声音与触觉？",
], step=1.13)
note(slide, "体积显示与普通屏幕上的立体影像不同：显示点本身位于空间中，因此可从多个方向观看。")

slide = base_slide("02  核心思路：让一颗小球充当“画笔”")
bullets(slide, [
    "超声波在空中形成声学陷阱，悬浮一颗直径约 2 mm 的白色小球。",
    "快速移动陷阱，小球沿预定三维路径运动；RGB 灯同步给它上色。",
    "人眼把短时间内出现的光点连成图形，这就是视觉暂留。",
    "同一组超声阵列还能把声能聚焦到手指，并通过调制产生可听声音。",
], step=1.13)
text_box(slide, 1.05, 6.18, 11.2, 0.45,
         "要点：空气本身没有发光；看到的是被照亮、快速移动的小球。", 19, True)
note(slide, "论文采用半径约 1 mm 的白色发泡聚苯乙烯（EPS）颗粒。")

slide = base_slide("03  装置由什么组成")
bullets(slide, [
    "上下相对放置两块超声阵列；每块 16 × 16 个换能器，共 512 个。",
    "两阵列间距约 23.4 cm，工作超声频率为 40 kHz。",
    "FPGA 控制每个换能器的相位和幅度，最高每秒更新 40,000 次。",
    "白色小球负责散射光；RGB LED 负责颜色；可控区域约 10 × 10 × 10 cm³。",
], step=1.13)
note(slide, "资料来源：论文 Methods 的 Experimental setup overview，以及正文 Operating principles。")

slide = base_slide("04  三维图像是怎样出现的")
bullets(slide, [
    "① 计算一条闭合的二维或三维路径，并给路径上的位置指定颜色。",
    "② 调整换能器相位，让声学陷阱沿路径移动，小球随之运动。",
    "③ LED 与小球位置同步，改变它反射出的红、绿、蓝光。",
    "④ 路径在约 0.1 秒内被重复扫过，观察者看到连续的形状。",
], step=1.12)
text_box(slide, 1.05, 6.18, 11.2, 0.45,
         "论文中的肉眼可见动态示例，扫描频率约为 10–12.5 Hz。", 19, True)
note(slide, "可以用挥动发光棒留下光轨作类比，但这里的路径是在三维空间中。")

slide = base_slide("05  为什么小球能移动得这么快")
bullets(slide, [
    "各换能器发声的时间稍有差别；相位控制使声波在目标位置形成陷阱。",
    "移动时，小球通常略微落后于陷阱中心，声辐射力持续拉着它加速。",
    "40 kHz 的高更新率使陷阱连续前移，小球不必在每一步完全停稳。",
    "视觉单模式下，实测峰值：水平 3.75 m/s，竖直方向 8.75 m/s。",
], step=1.13)
note(slide, "8.75 m/s 来自竖直线性速度测试；不能理解为任何图形、任何三模态状态下都能达到。论文把这种运动解释为动态平衡，而非每步静止平衡。")

slide = base_slide("06  触觉是怎样加入的")
bullets(slide, [
    "主陷阱继续控制小球；第二个聚焦点负责刺激手指附近的皮肤。",
    "系统以很高的频率轮流分配位置：75% 的更新给小球，25% 给触觉点。",
    "触觉点的声压按约 250 Hz 变化，皮肤感到振动。",
    "触觉焦点可放在图形附近；与小球位置完全重合容易干扰悬浮。",
], step=1.13)
note(slide, "每四次 25 微秒更新中，前三次分配给主陷阱，第四次给触觉点。论文中触觉点的超声声压超过 150 dB；这一数值是高频超声测量，不宜直接类比日常可听噪声。")

slide = base_slide("07  声音是怎样加入的")
bullets(slide, [
    "对 40 kHz 超声信号做幅度调制，空气中的非线性传播使人听到音频成分。",
    "作者用上边带调制，尽量减少失真，同时保持小球悬浮。",
    "非定向声音可与视觉、触觉同时出现；论文示例达到日常对话量级。",
    "定向声音需要占用第二个聚焦点；这台原型机此时无法同时产生触觉点。",
], step=1.13)
note(slide, "文中展示同时发声约 60 dB；Methods 也对不同声学模式进行了空间声压测量。不要把定向声音与三模态同步展示混为同一实验条件。")

slide = base_slide("08  性能数据：加入功能后的代价", source="论文 Table 1；下表为水平方向的保守测量值")
rows = [
    ["工作模式", "仅视觉", "视觉 + 声音", "视觉 + 声音 + 触觉"],
    ["最高水平速度", "3.75 m/s", "3.375 m/s", "2.5 m/s"],
    ["最高加速度", "141 m/s²", "122 m/s²", "62 m/s²"],
    ["急转弯速度", "0.75 m/s", "0.5 m/s", "0.375 m/s"],
    ["已展示帧率", "12.5 Hz", "10 Hz", "10 Hz"],
]
table = slide.shapes.add_table(5, 4, Inches(0.84), Inches(1.63), Inches(11.72), Inches(4.25)).table
for idx, width in enumerate((2.6, 2.8, 2.9, 3.42)):
    table.columns[idx].width = Inches(width)
for i, row in enumerate(rows):
    table.rows[i].height = Inches(0.85)
    for j, value in enumerate(row):
        cell = table.cell(i, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE
        cell.margin_left = Inches(0.12)
        cell.margin_right = Inches(0.08)
        cell.margin_top = Inches(0.04)
        cell.margin_bottom = Inches(0.04)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.text = value
        for p in cell.text_frame.paragraphs:
            for run in p.runs:
                run.font.name = FONT
                run.font.size = Pt(18 if i else 17)
                run.font.bold = i == 0 or j == 0
                run.font.color.rgb = BLACK
text_box(slide, 0.92, 6.21, 11.6, 0.55,
         "触觉需要分走部分声学能量，因此速度和加速度下降。", 19, True)
note(slide, "表 1 为不同工作条件下的保守水平运动参数；不同模式的最高速度不能混用。")

slide = base_slide("09  论文实际展示了什么")
bullets(slide, [
    "肉眼可见：彩色线条、面孔、三维结等简单动态内容。",
    "二维和三维的复杂彩色光栅图，也可由扫描粒子逐点绘制。",
    "但部分精细图片采用 8 秒或 20 秒长曝光；照片中的完整细节并非实时肉眼画质。",
    "论文还分别测量了粒子运动、声音输出和手指附近的触觉焦点。",
], step=1.13)
note(slide, "对应论文 Fig. 1b,c,e 为 POV 动态演示；Fig. 1d、Fig. 2b,c、Fig. 3c 等含长曝光结果。Fig. 3a,b 金字塔为 2 秒曝光。")

slide = base_slide("10  局限与研究价值")
bullets(slide, [
    "目前主要靠一颗小球逐点扫描，画面尺寸、亮度和复杂度都有限。",
    "手从上方或下方挡住阵列，可能扰乱声场，使小球掉落。",
    "触觉点不宜与正在运动的小球完全重合；定向声音也会占用第二陷阱。",
    "它的价值是验证了一条路径：同一声学平台可协调三维显示、发声和触觉。",
], step=1.13)
note(slide, "作者讨论了提高换能器功率、频率、照明质量和运动预测模型等方向；这些是未来改进，不是当前原型已完成的能力。")

slide = base_slide("11  难词速查")
terms = [
    ("声泳 / 声致迁移", "利用声波产生的力移动微小物体。"),
    ("声学陷阱", "声波叠加形成的稳定区域，可将小球悬在空中。"),
    ("视觉暂留（POV）", "人眼把短时间内依次出现的光点融合成连续形状。"),
    ("时分复用", "快速轮流把阵列分配给悬浮小球和刺激手指。"),
    ("幅度调制", "让超声强弱随音频信号变化，从而生成可听声。"),
    ("戈尔科夫势", "描述小颗粒在声场中受力趋势的数学模型。"),
]
for i, (term, meaning) in enumerate(terms):
    y = 1.39 + i * 0.87
    text_box(slide, 0.93, y, 3.1, 0.64, term, 17, True)
    text_box(slide, 4.05, y, 8.2, 0.64, meaning, 17)
note(slide, "这里的解释为理解论文所需的简化版本。声辐射力可从戈尔科夫势的空间变化推导。")

slide = base_slide("12  总结")
bullets(slide, [
    "实现了什么：用高速悬浮粒子形成三维图形，同时生成声音与空中触觉。",
    "如何实现：超声相位控制粒子，RGB 控制颜色，时分复用提供触觉，幅度调制提供声音。",
    "最重要的边界：这是研究原型；长曝光图像、运动峰值和三模态演示需要分别解读。",
], start_y=1.75, step=1.32, size=22)
text_box(slide, 0.95, 6.17, 11.4, 0.47,
         "论文：doi:10.1038/s41586-019-1739-5", 18)
note(slide, "建议结束时回到第一张图：真正新颖的是统一声学机制，而不是单纯把小球悬起来。")

prs.save(OUTPUT)
print("Presentation saved")
print(f"slides={len(prs.slides)}")
