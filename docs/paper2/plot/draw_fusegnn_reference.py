"""Draw a reference revision of the manuscript's FuseGNN architecture."""

import base64
from html import escape
from math import hypot
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "latex" / "layout_parameter_surrogate.pdf"
OUT = ROOT / "fusegnn_reference.svg"
BLUE, ORANGE, GREEN = "#2b6ca5", "#a85f36", "#427b4b"
INK = "#18344d"


def make_svg(input_panel):
    out = ['''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
width="1296" height="620" viewBox="0 0 1296 620">
<title>FuseGNN architecture reference</title>
<desc>Each task independently encodes and fuses the same kinds of inputs. The graph
encoder, tabular encoder, and scalar head have separate weights in each network.</desc>
<rect width="1296" height="620" fill="white"/>
<g font-family="Georgia, 'Times New Roman', serif" fill="#18344d">''']

    def label(x, y, value, size=15, bold=False, anchor="middle"):
        weight = "700" if bold else "400"
        out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                   f'font-size="{size}" font-weight="{weight}">{escape(value)}</text>')

    def box(x, y, w, h, fill, stroke, radius=9):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                   f'fill="{fill}" stroke="{stroke}" stroke-width="1.25"/>')

    def arrow(points, head=True):
        coords = " ".join(f"{x},{y}" for x, y in points)
        out.append(f'<polyline points="{coords}" fill="none" stroke="#325976" '
                   'stroke-width="1.5" stroke-linejoin="round"/>')
        if head:
            (a, b), (x, y) = points[-2:]
            length = hypot(x - a, y - b)
            u, v = (x - a) / length, (y - b) / length
            left = f"{x - 8*u + 4*v:.2f},{y - 8*v - 4*u:.2f}"
            right = f"{x - 8*u - 4*v:.2f},{y - 8*v + 4*u:.2f}"
            out.append(f'<polygon points="{x},{y} {left} {right}" fill="#325976"/>')

    def card(x, y, w, h, lines, fill, color):
        box(x, y, w, h, fill, color)
        start = y + h / 2 + 5 - (len(lines) - 1) * 10
        for i, line in enumerate(lines):
            label(x + w / 2, start + 20 * i, line, 15, i == 0)

    def cells(x, y, fills, width=13):
        for i, fill in enumerate(fills):
            box(x + width * i, y, width, 10, fill, "#8298a8", 0)

    for x, w, title in [(6, 204, "(a) Inputs"), (220, 226, "(b) Representation"),
                        (456, 514, "(c) FuseGNN"), (980, 310, "(d) Task instances")]:
        box(x, 7, w, 606, "white", BLUE, 12)
        box(x, 7, w, 46, "#f8fbff", BLUE, 12)
        label(x + w / 2, 37, title, 24, True)

    out.append(f'<image x="10" y="59" width="196" height="542" '
               f'xlink:href="data:image/png;base64,{input_panel}"/>')
    arrow([(210, 170), (229, 170)])
    arrow([(210, 346), (229, 346)])
    arrow([(210, 514), (229, 514)])

    box(230, 67, 206, 221, "#f4f9ff", BLUE)
    label(333, 93, "Room-level graph (G)", 17, True)
    nodes = [(261, 142), (298, 129), (298, 168), (267, 192), (334, 155),
             (369, 127), (404, 145), (401, 188), (364, 181)]
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (2, 4),
                 (4, 5), (4, 8), (5, 6), (6, 7), (6, 8), (7, 8)]:
        arrow([nodes[i], nodes[j]], False)
    for x, y in nodes:
        out.append(f'<circle cx="{x}" cy="{y}" r="7" fill="#4b94c3" stroke="{BLUE}"/>')
    label(333, 214, "41 node features")
    label(333, 232, "8 edge attributes")
    label(333, 255, "7 room-graph summaries")
    cells(291, 264, ["#c4d8ee"] * 7, 12)

    box(230, 302, 206, 126, "#fff6ee", ORANGE)
    label(333, 329, "Design and conditions (d)", 17, True)
    label(333, 357, "8 standardized numerical inputs")
    label(333, 381, "4 categorical inputs")
    cells(265, 402, ["#e8ae82"] * 8 + ["#faf2dc"] * 4, 11)

    box(230, 450, 206, 120, "#f2f8ef", GREEN)
    label(333, 479, "Global layout statistics (q)", 17, True)
    label(333, 510, "41 standardized numerical inputs")
    cells(274, 535, ["#b8d6ae"] * 9)

    label(473, 79, "Architecture of one task network", 16, True, "start")
    card(474, 93, 200, 86, ["Node projection", "3 GraphSAGE layers",
                            "Batch norm + residual + ReLU"], "#eaf4fe", BLUE)
    card(700, 93, 242, 86, ["Node pooling", "Mean (H) and maximum (H)"],
         "#eaf4fe", BLUE)
    arrow([(436, 134), (474, 134)])
    arrow([(674, 134), (700, 134)])
    card(474, 194, 200, 60, ["Edge attribute MLP"], "#eaf4fe", BLUE)
    card(700, 194, 242, 60, ["Edge mean pooling (H/2)"], "#eaf4fe", BLUE)
    arrow([(436, 203), (452, 203), (452, 224), (474, 224)])
    arrow([(674, 224), (700, 224)])
    card(537, 269, 405, 43, ["7 room-graph summaries · direct input"], "#f4f9ff", BLUE)
    arrow([(436, 265), (448, 265), (448, 290), (537, 290)])

    card(474, 331, 468, 92, ["Joint tabular encoder (H)",
                            "49 numerical inputs + 4 learned 4-D embeddings",
                            "Batch norm on numeric inputs · two-layer MLP"],
         "#fff3e8", ORANGE)
    arrow([(436, 371), (474, 371)])
    arrow([(436, 511), (450, 511), (450, 406), (474, 406)])

    for y in (134, 224, 290, 377):
        arrow([(942, y), (955, y)], False)
    arrow([(955, 134), (955, 486), (942, 486)])
    card(474, 446, 468, 86, ["Concatenate  ·  3.5H + 7",
                            "Node mean + node max + edge mean + graph summaries + tabular"],
         "#f3f8fe", BLUE)
    cells(493, 514, ["#7aafd4"] * 10 + ["#c3dceb"] * 10 + ["#aeb8d7"] * 5 +
          ["#e3dfd5"] * 2 + ["#e8ae82"] * 8, 12)
    arrow([(708, 532), (708, 551)])
    card(474, 551, 468, 43, ["Scalar head  ·  Linear → ReLU → Dropout → Linear"],
         "#f3f8fe", BLUE)

    label(1135, 79, "Separate training · no shared weights", 16, True)
    for y, title, width, fused, transform, result, fill, color in [
        (96, "Feasibility classifier Φf", 256, 903, "Sigmoid", "Feasibility score  p_g",
         "#eaf4fe", BLUE),
        (353, "Reinforcement regressor Φs", 128, 455, "Inverse log-mass transform",
         "Reinforcement mass  m_g", "#f0f7ed", GREEN),
    ]:
        box(995, y, 280, 224, fill, color)
        label(1135, y + 27, title, 18, True)
        card(1013, y + 43, 244, 58,
             ["Own encoders, fusion and head", f"H = {width}   ·   Fusion: {fused}"],
             "white", color)
        arrow([(1135, y + 101), (1135, y + 120)])
        card(1013, y + 120, 244, 37, [transform], "white", color)
        arrow([(1135, y + 157), (1135, y + 184)])
        label(1135, y + 204, result, 17, True)

    return "\n".join([*out, "</g></svg>"])


if __name__ == "__main__":
    with pymupdf.open(ORIGINAL) as doc:
        clip = pymupdf.Rect(9, 42, 202, 447)
        panel = doc[0].get_pixmap(matrix=pymupdf.Matrix(2.5, 2.5), clip=clip)
        image = base64.b64encode(panel.tobytes("png")).decode("ascii")
    OUT.write_text(make_svg(image), encoding="utf-8")
    with pymupdf.open(OUT) as doc:
        doc[0].get_pixmap().save(OUT.with_suffix(".png"))
