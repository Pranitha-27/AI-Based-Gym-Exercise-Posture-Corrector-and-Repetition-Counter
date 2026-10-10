"""
Generates an editable Microsoft PowerPoint (.pptx) presentation containing the
System Architecture Flowchart with native shapes, connectors, and styled text.
Creates both Dark Mode and Light Mode slide variants.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

def create_flowchart_presentation(output_path="architecture_flowchart.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # Standard 16:9 widescreen
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme Definitions
    themes = [
        {
            "name": "Dark Theme (Modern / Technical)",
            "bg": RGBColor(18, 20, 24),
            "card_bg": RGBColor(28, 32, 40),
            "card_border": RGBColor(0, 180, 240),
            "title_color": RGBColor(255, 255, 255),
            "sub_color": RGBColor(180, 215, 245),
            "arrow_color": RGBColor(0, 180, 240),
            "heading": RGBColor(0, 210, 255)
        },
        {
            "name": "Light Theme (Academic / Paper)",
            "bg": RGBColor(250, 252, 255),
            "card_bg": RGBColor(255, 255, 255),
            "card_border": RGBColor(30, 80, 160),
            "title_color": RGBColor(20, 35, 60),
            "sub_color": RGBColor(70, 90, 120),
            "arrow_color": RGBColor(30, 80, 160),
            "heading": RGBColor(20, 60, 140)
        }
    ]

    for theme in themes:
        slide = prs.slides.add_slide(blank_layout)

        # Background Fill
        bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg_shape.fill.solid()
        bg_shape.fill.fore_color.rgb = theme["bg"]
        bg_shape.line.fill.background()

        # Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.25), Inches(11.733), Inches(0.6))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = "System Architecture & Execution Flowchart"
        p.font.name = "Segoe UI"
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = theme["heading"]
        p.alignment = PP_ALIGN.CENTER

        # Helper to create styled node boxes
        def add_box(left, top, width, height, title, subtitle):
            shape = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(left), Inches(top), Inches(width), Inches(height)
            )
            shape.fill.solid()
            shape.fill.fore_color.rgb = theme["card_bg"]
            shape.line.color.rgb = theme["card_border"]
            shape.line.width = Pt(1.5)

            tf = shape.text_frame
            tf.word_wrap = True
            tf.margin_left = Inches(0.1)
            tf.margin_right = Inches(0.1)
            tf.margin_top = Inches(0.08)
            tf.margin_bottom = Inches(0.08)

            # Title
            p1 = tf.paragraphs[0]
            p1.text = title
            p1.font.name = "Segoe UI"
            p1.font.size = Pt(11)
            p1.font.bold = True
            p1.font.color.rgb = theme["title_color"]
            p1.alignment = PP_ALIGN.CENTER

            # Subtitle
            if subtitle:
                p2 = tf.add_paragraph()
                p2.text = subtitle
                p2.font.name = "Segoe UI"
                p2.font.size = Pt(9)
                p2.font.color.rgb = theme["sub_color"]
                p2.alignment = PP_ALIGN.CENTER

            return shape

        # Helper to draw down arrow
        def add_down_arrow(center_x, top_y, bottom_y):
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.DOWN_ARROW,
                Inches(center_x - 0.08), Inches(top_y), Inches(0.16), Inches(bottom_y - top_y)
            )
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = theme["arrow_color"]
            arrow.line.fill.background()
            return arrow

        # Horizontal Center of slide is 13.333 / 2 = 6.666
        cx = 6.666
        w_main = 3.6
        h_box = 0.65

        # --- LEVEL 1: Input Source ---
        y1 = 0.95
        add_box(cx - w_main/2, y1, w_main, h_box, "Input Source", "(Webcam / Video File / Demo)")

        # --- LEVEL 2: MediaPipe PoseLandmarker ---
        y2 = 1.85
        add_down_arrow(cx, y1 + h_box, y2)
        add_box(cx - w_main/2, y2, w_main, h_box, "MediaPipe PoseLandmarker", "(33 3D Keypoints)")

        # --- LEVEL 3: Kinematics Engine ---
        y3 = 2.75
        add_down_arrow(cx, y2 + h_box, y3)
        add_box(cx - w_main/2, y3, w_main, h_box, "Kinematics Engine", "(2D/3D Angles, Inclinations, EMA Filter)")

        # --- LEVEL 4: Auto-Perspective Resolver ---
        y4 = 3.65
        add_down_arrow(cx, y3 + h_box, y4)
        add_box(cx - w_main/2, y4, w_main, h_box, "Auto-Perspective Resolver", "(Left / Right / Frontal View)")

        # --- LEVEL 5: Exercise State Machine ---
        y5 = 4.55
        add_down_arrow(cx, y4 + h_box, y5)
        add_box(cx - w_main/2, y5, w_main, h_box, "Exercise State Machine", "(Squat, Push-up, Curl, Plank, etc.)")

        # --- LEVEL 6: Branching (Rep Counter & Posture Violation Detector) ---
        y6 = 5.45
        w6 = 3.8
        gap6 = 0.6
        x6_left = cx - gap6/2 - w6
        x6_right = cx + gap6/2

        # Draw branching lines/arrows from Level 5 to Level 6
        add_down_arrow(x6_left + w6/2, y5 + h_box, y6)
        add_down_arrow(x6_right + w6/2, y5 + h_box, y6)

        add_box(x6_left, y6, w6, h_box, "Rep Counter & Quality Gate", "(Good Reps vs. Bad Reps, Form Score %)")
        add_box(x6_right, y6, w6, h_box, "Posture Violation Detector", "(Depth, Spine Lean, Valgus, Sagging)")

        # --- LEVEL 7: Output Layer (3 Columns) ---
        y7 = 6.35
        w7 = 3.6
        gap7 = 0.35
        # Total width = 3 * 3.6 + 2 * 0.35 = 10.8 + 0.7 = 11.5
        x7_left = cx - (3 * w7 + 2 * gap7) / 2
        x7_mid = x7_left + w7 + gap7
        x7_right = x7_mid + w7 + gap7

        # Connectors down to Level 7
        add_down_arrow(x7_left + w7/2, y6 + h_box, y7)
        add_down_arrow(x7_mid + w7/2, y6 + h_box, y7)
        add_down_arrow(x7_right + w7/2, y6 + h_box, y7)

        add_box(x7_left, y7, w7, h_box, "Visualizer HUD", "(Color-Coded Skeleton, Angle Badges, Progress Meter)")
        add_box(x7_mid, y7, w7, h_box, "Asynchronous Audio Engine", "(Daemon Worker Chimes & Alerts)")
        add_box(x7_right, y7, w7, h_box, "Workout Reporter", "(Telemetry Logs, JSON & Terminal Summary)")

    prs.save(output_path)
    print(f"PowerPoint presentation generated at: {os.path.abspath(output_path)}")
    return os.path.abspath(output_path)

if __name__ == "__main__":
    create_flowchart_presentation()
