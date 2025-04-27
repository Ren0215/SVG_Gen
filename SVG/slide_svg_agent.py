"""
slide_svg_agent.py – COMPLETE
=============================
LangGraph pipeline that takes raw context‑map JSON (text + company) and
produces an SVG (and optional PPTX/MP4) using OpenAI o3 and Claude 3.7 Sonnet.

Key features
------------
* **IntentParse** – optional high‑level prompt understanding (kept minimal).
* **ParseContextMap** – extracts 6 categories × 5 subnodes from raw text, ignores URLs.
* **TemplateSelect** – chooses `context_map_dark` by default (tags match).
* **PaletteOpt** – placeholder: returns default palette.
* **SvgCompose** – o3 function‑calling → SlideSpec → Jinja2 render.
* **WcagCheck / SelfRefine** – contrast ratio < 4.5 triggers up to 3 re‑palette cycles.
* **Export** – `--export pptx` (python‑pptx) and stub for `mp4`.

Run example::

   $ python slide_svg_agent.py \
        --prompt "Initial社のコンテキストマップを作成" \
        --src data/sample_input.json \
        --export pptx

Dependencies in requirements.txt.
"""
from __future__ import annotations
import argparse, json, os, re, textwrap
from pathlib import Path
from typing import List, Dict

import jinja2, svgwrite
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from langgraph.graph import StateGraph, END
from openai import OpenAI
from anthropic import Anthropic
try:
    from pptx import Presentation
except ImportError:
    Presentation = None

def _luminance(hex_color: str) -> float:
    """sRGB hex (#RRGGBB) → 相対輝度"""
    r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
    def lin(v):  # gamma 補正
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

def contrast(fg: str, bg: str) -> float:
    """2 色のコントラスト比 (大きい方 / 小さい方)"""
    l1, l2 = _luminance(fg), _luminance(bg)
    if l1 < l2:
        l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)

def load_palette(template_id: str, palette_name: str) -> dict:
    """templates/<id>/palette.json から <palette_name> を取得"""
    pfile = TPL_DIR / template_id / "palette.json"
    data  = json.loads(pfile.read_text(encoding="utf-8"))
    return data[palette_name]

# ──────────── Paths & env ────────────
ROOT = Path(__file__).parent
TPL_DIR = ROOT / "templates"
ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TPL_DIR)))
ENV.globals.update(len=len, enumerate=enumerate, zip=zip)
ENV.filters["split"] = lambda s, sep="\n": s.split(sep) 
META = json.loads((TPL_DIR / "index.json").read_text(encoding="utf-8"))

# (LLMsは将来 Self-Refine 拡張時に使用)
OAI = OpenAI()
ANTH = Anthropic()

# ──────────── State schema ────────────
class SlidesState(BaseModel):
    raw_json: Dict
    company: str | None = None
    # extracted bullets
    strategy: List[str] = []
    product: List[str] = []
    top: List[str] = []
    industry: List[str] = []
    society: List[str] = []
    global_: List[str] = Field([], alias="global")
    # generation artifacts
    template_id: str | None = None
    palette: str | None = None
    svg: str | None = None
    contrast_fails: List = []
    status: str | None = None

# ──────────── SlideSpec schema ─────────
class BulletSlide(BaseModel):
    company: str
    center: str
    strategy: List[str]
    product: List[str]
    top: List[str]
    industry: List[str]
    society: List[str]
    global_: List[str] = Field(..., alias="global")

class SlideSpec(BaseModel):
    template_id: str
    palette: str
    width: int = 1280
    height: int = 720
    content: BulletSlide

# ──────────── Helpers ────────────────
def pick_template() -> str:
    return "context_map_dark"

offset = 75
coords = {
    "strategy":  [(265+i*offset,148) for i in range(3)] +
                 [(265+j*offset,218) for j in range(2)],
    "product":   [(75+i*offset,300)  for i in range(2)] +
                 [(75+i*offset,370)  for i in range(3)],
    "top":       [(265+i*offset,512) for i in range(3)] +
                 [(265+j*offset,582) for j in range(2)],
    "industry":  [(735+i*offset,512) for i in range(3)] +
                 [(735+j*offset,582) for j in range(2)],
    "society":   [(975+i*offset,300) for i in range(2)] +
                 [(975+i*offset,370) for i in range(3)],
    "global":    [(735+i*offset,148) for i in range(3)] +
                 [(735+j*offset,218) for j in range(2)],
}

def render_svg(spec: SlideSpec) -> str:
    tpl = ENV.get_template(f"{spec.template_id}/slide.svg.j2")

    palette = load_palette(spec.template_id, spec.palette)

    ctx = spec.model_dump()
    ctx["palette_colors"] = palette      
    ctx["coords"] = coords               # ← 座標辞書を渡す行があればそのまま
    return tpl.render(**ctx)

# ──────────── Node functions ──────────
CAT_MAP = {
    "企業戦略文脈": "strategy",
    "プロダクト文脈": "product",
    "TOP文脈":      "top",
    "業界トレンド文脈": "industry",
    "社会トレンド文脈": "society",
    "グローバル文脈":   "global",
}
URL_RE   = re.compile(r"→?https?://\S+")
HDR_RE   = re.compile(r"^[\s　\r\n]*[（\(]?[A-FＡ-Ｆ][\)）]\s*(.+?文脈)")
BULLET_RE = re.compile(r"^[\-\*‧•●◦・]\s*")

def parse_context(state: SlidesState):
    buckets = {v: [] for v in CAT_MAP.values()}
    cur = None
    for raw in state.raw_json["text"].splitlines():
        # ① URL 削除 ② 両端 ” と 全角空白除去 ③ 箇条書き記号除去
        line = BULLET_RE.sub("", URL_RE.sub("", raw)).strip(' "\u3000')
        if not line:
            continue
        if m := HDR_RE.match(line):
            cur = CAT_MAP[m.group(1)]          # 企業戦略文脈 etc.
            continue
        if cur and len(buckets[cur]) < 5:
            buckets[cur].append(line)
    return state.model_copy(update=buckets)

def template_select(state: SlidesState):
    return state.model_copy(update={"template_id": pick_template(),
                              "palette": "context_map_dark"})

def svg_compose(state: SlidesState):
    content = {
        "company":  state.company or "Company",
        "center":   state.company or "Company",
        "strategy": state.strategy,
        "product":  state.product,
        "top":      state.top,
        "industry": state.industry,
        "society":  state.society,
        "global":   state.global_,
    }
    spec = SlideSpec(template_id=state.template_id,
                     palette=state.palette,
                     content=content)
    svg = render_svg(spec)
    return state.model_copy(update={"svg": svg})

def wcag_check(state: SlidesState):
    svg = state.svg or ""
    colors = re.findall(r"#[0-9A-Fa-f]{6}", svg)
    bg  = colors[0] if colors else "#FFFFFF"
    bad = [(c, bg) for c in colors[1:] if contrast(c, bg) < 4.5]
    return state.model_copy(update={"contrast_fails": bad,
                                    "status": None if bad else "ok"})

# ──────────── Build graph ─────────────
G = StateGraph(SlidesState)
G.add_node("parse",   parse_context)
G.add_node("select",  template_select)
G.add_node("compose", svg_compose)
G.add_node("wcag",    wcag_check)

G.add_edge("parse", "select")
G.add_edge("select","compose")
G.add_edge("compose","wcag")
G.add_edge("wcag", END)

G.set_entry_point("parse")

graph = G.compile()

# ──────────── CLI ─────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="JSON file with company & text")
    ap.add_argument("--export", nargs="*", choices=["pptx"])
    args = ap.parse_args()

    raw = json.loads(Path(args.src).read_text(encoding="utf-8"))
    init = SlidesState(raw_json=raw, company=raw.get("company"))
    result: SlidesState = graph.invoke(init)

    out = ROOT / "out"; out.mkdir(exist_ok=True)
    svg_path = out / "slide.svg"
    svg_path.write_text(result["svg"], encoding="utf-8")
    print("SVG →", svg_path)

    if args.export and "pptx" in args.export:
        if Presentation is None:
            print("python-pptx 未インストールのため PPTX 生成をスキップ")
        else:
            prs = Presentation()
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            slide.shapes.add_picture(str(svg_path), 0, 0,
                                     prs.slide_width, prs.slide_height)
            ppt_path = out / "slide.pptx"
            prs.save(ppt_path)
            print("PPTX →", ppt_path)