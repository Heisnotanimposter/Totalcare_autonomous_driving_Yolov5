#!/usr/bin/env python3
"""
Convert project Markdown deliverables (sources, kernels, reports, metrics)
through `md -> LaTeX (.tex) -> PDF (.pdf)` inside `report/` using Tectonic.
"""

import pathlib
import re
import shutil
import subprocess
from typing import List, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
MD_DIR = REPORT_DIR / "md"
TEX_DIR = REPORT_DIR / "tex"
PDF_DIR = REPORT_DIR / "pdf"

GITHUB_BLOB_BASE = "https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5/blob/main/"
GITHUB_TREE_BASE = "https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5/tree/main/"

SOURCE_DOCS: List[Tuple[str, str, str]] = [
    ("01_FINAL_REPORT.md", "FINAL_REPORT.md", "최종 결과 보고서 및 기본 평가표 점검"),
    ("02_ML_LAYER_SPEC.md", "ML_LAYER_SPEC.md", "머신러닝·딥러닝 모델 Layer별 기술설명서 및 참고자료"),
    ("03_ARCHITECTURE.md", "architecture.md", "시스템 아키텍처 및 좌표 변환 명세"),
    ("04_DATASET_AUDIT.md", "dataset_audit.md", "KITTI Tracking 데이터셋 무결성 감사 및 분할 명세"),
    ("05_EXPERIMENT_PLAN.md", "experiment_plan.md", "실험 계획 및 v1→v2→v3 정량 벤치마크"),
    ("06_VISUALIZATION_SPEC.md", "visualization_spec.md", "좌우 분할 BEV World Model 시각화 명세"),
    ("07_SOURCE_KERNEL_METRICS.md", "", "소스·커널·보고서·지표 통합 색인 (Appendix)"),
]


def _normalize_repo_url(url: str) -> str:
    prefix = f"file://{ROOT}/"
    if url.startswith(prefix):
        rel = url[len(prefix):]
    elif url.startswith("./"):
        rel = url[2:]
    elif url.startswith("http://") or url.startswith("https://"):
        return url
    else:
        rel = url
    target = ROOT / rel
    base = GITHUB_TREE_BASE if target.is_dir() else GITHUB_BLOB_BASE
    return base + rel


def _escape_latex_plain(s: str, allow_breaks: bool = False) -> str:
    s = s.replace("\\", "\x03BS\x04")
    for ch in ("{", "}"):
        s = s.replace(ch, "\\" + ch)
    s = s.replace("\x03BS\x04", r"\textbackslash{}")
    for old, new in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        s = s.replace(old, new)

    if allow_breaks:
        for ch in (".", "/", ",", "-", ":"):
            s = s.replace(ch, ch + r"\allowbreak{}")
        s = s.replace("_", r"\_\allowbreak{}")
    else:
        s = s.replace("_", r"\_")

    unicode_map = {
        "→": r"$\rightarrow$",
        "←": r"$\leftarrow$",
        "↔": r"$\leftrightarrow$",
        "±": r"$\pm$",
        "×": r"$\times$",
        "÷": r"$\div$",
        "≤": r"$\le$",
        "≥": r"$\ge$",
        "≈": r"$\approx$",
        "≠": r"$\ne$",
        "σ": r"$\sigma$",
        "θ": r"$\theta$",
        "ψ": r"$\psi$",
        "Δ": r"$\Delta$",
        "∈": r"$\in$",
        "∏": r"$\prod$",
        "√": r"$\sqrt{}$",
        "—": "---",
        "–": "--",
        "−": "-",
    }
    for u, tex in unicode_map.items():
        s = s.replace(u, tex)
    return s


def _escape_url(u: str) -> str:
    return u.replace("%", r"\%").replace("#", r"\#").replace("_", r"\_").replace("&", r"\&")


def format_inline_md(text: str) -> str:
    """Recursively convert bold, italic, links, code spans, and inline math into clean LaTeX."""
    # Protect `code` and $math$ first with placeholder tokens so nested **bold `code`**, **bold $math$** work cleanly
    protected: List[str] = []

    def _stash(latex_fragment: str) -> str:
        idx = len(protected)
        protected.append(latex_fragment)
        return f"\x01TOK{idx}\x02"

    # Stash inline code
    def _code_sub(m: re.Match) -> str:
        code_inner = _escape_latex_plain(m.group(1), allow_breaks=True)
        return _stash(r"\texttt{" + code_inner + "}")

    work = re.sub(r"`([^`]+)`", _code_sub, text)

    # Stash inline math $...$
    def _math_sub(m: re.Match) -> str:
        return _stash(m.group(0))

    work = re.sub(r"\$[^$\n]+?\$", _math_sub, work)

    # Stash markdown links [label](url)
    def _link_sub(m: re.Match) -> str:
        raw_lbl = m.group(1)
        raw_url = m.group(2)
        lbl = _escape_with_tokens(raw_lbl, protected)
        url = _normalize_repo_url(raw_url)
        return _stash(r"\href{" + _escape_url(url) + "}{" + lbl + "}")

    work = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _link_sub, work)

    # Convert **bold** (including bold spanning across stashed code/math tokens!) and *italic*
    work = re.sub(r"\*\*([^*]+?)\*\*", lambda m: _stash(r"\textbf{" + _escape_with_tokens(m.group(1), protected) + "}"), work)
    work = re.sub(r"\*([^*\n]+?)\*", lambda m: _stash(r"\textit{" + _escape_with_tokens(m.group(1), protected) + "}"), work)

    return _escape_with_tokens(work, protected)


def _escape_with_tokens(s: str, protected: List[str]) -> str:
    parts = re.split(r"(\x01TOK\d+\x02)", s)
    res: List[str] = []
    for part in parts:
        tm = re.match(r"^\x01TOK(\d+)\x02$", part)
        if tm:
            res.append(protected[int(tm.group(1))])
        else:
            # Allow long bare URLs inside plain prose to wrap cleanly too
            if "https://" in part or "http://" in part:
                res.append(_escape_latex_plain(part, allow_breaks=True))
            else:
                res.append(_escape_latex_plain(part, allow_breaks=False))
    return "".join(res)


def _split_md_table_row(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells: List[str] = []
    cur: List[str] = []
    in_code = False
    i = 0
    while i < len(stripped):
        ch = stripped[i]
        if ch == "\\" and i + 1 < len(stripped) and stripped[i + 1] == "|":
            cur.append("|")
            i += 2
            continue
        if ch == "`":
            in_code = not in_code
            cur.append(ch)
        elif ch == "|" and not in_code:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def _render_table(rows: List[str]) -> str:
    if len(rows) < 2:
        return "\n".join(format_inline_md(r) for r in rows) + "\n"
    headers = _split_md_table_row(rows[0])
    ncols = len(headers)
    body_rows = [_split_md_table_row(r) for r in rows[2:]]

    weights = [max(4, min(22, len(re.sub(r"[`*_\[\]()]", "", h)))) for h in headers]
    for br in body_rows[:30]:
        for idx in range(min(ncols, len(br))):
            clean_c = re.sub(r"\([^)]*\)|\[|\]|`|\*\*", "", br[idx])
            weights[idx] = max(weights[idx], min(34, len(clean_c)))
    total_w = max(1, sum(weights))
    avail = max(0.74, 0.92 - 0.018 * ncols)
    col_specs = [f">{{\\raggedright\\arraybackslash}}p{{{avail * (w / total_w):.3f}\\textwidth}}" for w in weights]

    lines = [
        "",
        r"\begin{footnotesize}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}",
        r"\begin{longtable}{" + " ".join(col_specs) + "}",
        r"\toprule",
        " & ".join(r"\textbf{" + format_inline_md(h) + "}" for h in headers) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(r"\textbf{" + format_inline_md(h) + "}" for h in headers) + r" \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for br in body_rows:
        padded = (br + [""] * ncols)[:ncols]
        lines.append(" & ".join(format_inline_md(c) for c in padded) + r" \\")
    lines.extend([r"\end{longtable}", r"\end{footnotesize}", ""])
    return "\n".join(lines)


def markdown_to_latex_body(md_text: str, heading_shift: int = 0) -> str:
    """Convert Markdown string into a clean LaTeX body fragment with nested lists."""
    lines = md_text.splitlines()
    out: List[str] = []
    i = 0
    # Stack of (indent_spaces, 'itemize' | 'enumerate')
    list_stack: List[Tuple[int, str]] = []

    def close_all_lists() -> None:
        while list_stack:
            _, kind = list_stack.pop()
            out.append(r"\end{" + kind + "}")

    def close_lists_to(indent: int, kind: str) -> None:
        while list_stack and (
            list_stack[-1][0] > indent or (list_stack[-1][0] == indent and list_stack[-1][1] != kind)
        ):
            _, k = list_stack.pop()
            out.append(r"\end{" + k + "}")
        if not list_stack or list_stack[-1][0] < indent:
            opt = "[leftmargin=1.5em,itemsep=2pt]" if kind == "itemize" else "[leftmargin=1.8em,itemsep=2pt]"
            out.append(r"\begin{" + kind + "}" + opt)
            list_stack.append((indent, kind))

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            close_all_lists()
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                cl = (
                    lines[i]
                    .replace("│", "|")
                    .replace("▼", "v")
                    .replace("─", "-")
                    .replace("┌", "+")
                    .replace("└", "+")
                    .replace("┐", "+")
                    .replace("┘", "+")
                )
                code_lines.append(cl)
                i += 1
            i += 1
            out.append(r"\begin{quote}\begin{footnotesize}\begin{verbatim}")
            out.extend(code_lines)
            out.append(r"\end{verbatim}\end{footnotesize}\end{quote}")
            continue

        if stripped.startswith("$$"):
            close_all_lists()
            if stripped.endswith("$$") and len(stripped) > 4:
                out.append(r"\[" + stripped[2:-2].strip() + r"\]")
                i += 1
            else:
                i += 1
                mb = []
                while i < len(lines) and not lines[i].strip().startswith("$$"):
                    mb.append(lines[i])
                    i += 1
                i += 1
                out.append(r"\[" + "\n".join(mb) + r"\]")
            continue

        if not stripped:
            close_all_lists()
            out.append("")
            i += 1
            continue

        if stripped in ("---", "***", "___"):
            close_all_lists()
            out.append(r"\par\noindent\rule{\textwidth}{0.4pt}\par\medskip")
            i += 1
            continue

        hm = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if hm:
            close_all_lists()
            lvl = max(1, min(4, len(hm.group(1)) + heading_shift))
            title = format_inline_md(hm.group(2).strip())
            if lvl == 1:
                out.append(r"\section{" + title + "}")
            elif lvl == 2:
                out.append(r"\subsection{" + title + "}")
            elif lvl == 3:
                out.append(r"\subsubsection{" + title + "}")
            else:
                out.append(r"\paragraph{" + title + "}")
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?[\s:\-|]+\|?$", lines[i + 1].strip()):
            close_all_lists()
            t_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                t_rows.append(lines[i])
                i += 1
            out.append(_render_table(t_rows))
            continue

        bm = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        nm = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if bm:
            close_lists_to(len(bm.group(1)), "itemize")
            out.append(r"\item " + format_inline_md(bm.group(2)))
            i += 1
            continue
        if nm:
            close_lists_to(len(nm.group(1)), "enumerate")
            out.append(r"\item " + format_inline_md(nm.group(3)))
            i += 1
            continue

        close_all_lists()
        out.append(format_inline_md(stripped) + "\n")
        i += 1

    close_all_lists()
    return "\n".join(out)


def tex_preamble(title: str, subtitle: str, lang: str = "ko") -> str:
    hangul_block = (
        "\\usepackage{kotex}\n"
        "\\usepackage{fontspec}\n"
        "\\setmainhangulfont{Apple SD Gothic Neo}[BoldFont={* Bold}]\n"
        "\\setsanshangulfont{Apple SD Gothic Neo}[BoldFont={* Bold}]\n"
        "\\setmonofont{Menlo}[Scale=0.82]\n"
        if lang == "ko"
        else "\\usepackage{fontspec}\n\\setmonofont{Menlo}[Scale=0.82]\n"
    )
    return rf"""\documentclass[10.5pt,a4paper]{{article}}
\usepackage{{geometry}}
\geometry{{margin=18mm,top=20mm,bottom=20mm}}
{hangul_block}\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs,longtable,array}}
\usepackage{{graphicx}}
\usepackage{{enumitem}}
\usepackage{{xcolor}}
\usepackage{{fancyhdr}}
\usepackage[hidelinks]{{hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!65!black}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.18}}
\setcounter{{secnumdepth}}{{0}}
\sloppy
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\small\color{{gray}}{_escape_latex_plain(title)}}}
\fancyhead[R]{{\small\color{{gray}}KITTI Tracking Temporal World Model}}
\fancyfoot[C]{{\small\thepage}}
\begin{{document}}
\begin{{center}}
{{\LARGE\bfseries {_escape_latex_plain(title)}}}\\[6pt]
{{\normalsize\color{{gray}} {_escape_latex_plain(subtitle)}}}\\[4pt]
{{\small\href{{https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5}}{{https://github.com/Heisnotanimposter/Totalcare\_autonomous\_driving\_Yolov5}}}}
\end{{center}}
\vspace{{4pt}}
\hrule
\vspace{{8pt}}
"""


def compile_tectonic(tex_path: pathlib.Path, pdf_out_path: pathlib.Path) -> None:
    cmd = ["tectonic", "-o", str(pdf_out_path.parent), str(tex_path)]
    subprocess.run(cmd, check=True)
    generated = pdf_out_path.parent / (tex_path.stem + ".pdf")
    if generated != pdf_out_path and generated.exists():
        generated.replace(pdf_out_path)


def build_language_bundle(lang: str) -> pathlib.Path:
    is_ko = lang == "ko"
    src_md_dir = REPORT_DIR / ("md_ko" if is_ko else "md_en")
    tex_sub_dir = TEX_DIR / lang
    pdf_sub_dir = PDF_DIR / lang
    tex_sub_dir.mkdir(parents=True, exist_ok=True)
    pdf_sub_dir.mkdir(parents=True, exist_ok=True)

    if is_ko:
        for dst_name, src_rel, _ in SOURCE_DOCS:
            if dst_name in ("01_FINAL_REPORT.md", "02_ML_LAYER_SPEC.md") and src_rel:
                shutil.copy2(ROOT / src_rel, src_md_dir / dst_name)
    else:
        for dst_name, src_rel, _ in SOURCE_DOCS:
            if dst_name in ("03_ARCHITECTURE.md", "04_DATASET_AUDIT.md", "05_EXPERIMENT_PLAN.md", "06_VISUALIZATION_SPEC.md") and src_rel:
                shutil.copy2(ROOT / src_rel, src_md_dir / dst_name)

    for md_name, _, doc_desc in SOURCE_DOCS:
        md_path = src_md_dir / md_name
        stem = md_path.stem
        tex_path = tex_sub_dir / f"{stem}.tex"
        pdf_path = pdf_sub_dir / f"{stem}.pdf"

        raw_md = md_path.read_text(encoding="utf-8")
        lines = raw_md.splitlines()
        title = doc_desc
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip()
            raw_md = "\n".join(lines[1:])

        tex_doc = tex_preamble(title, title, lang=lang) + markdown_to_latex_body(raw_md, heading_shift=-1) + "\n\\end{document}\n"
        tex_path.write_text(tex_doc, encoding="utf-8")
        compile_tectonic(tex_path, pdf_path)
        print(f"[OK:{lang.upper()}] {md_path.name} -> {tex_path.name} -> {pdf_path.name}")

    suffix = "KOR" if is_ko else "ENG"
    master_tex = TEX_DIR / f"Totalcare_Autonomous_Driving_Report_{suffix}.tex"
    master_pdf = REPORT_DIR / f"Totalcare_Autonomous_Driving_Report_{suffix}.pdf"

    if is_ko:
        m_title = "Camera-Centric Temporal Ego-centric World Model 최종 보고서 (KOR)"
        m_sub = "KITTI Tracking 기반 검출·세그멘테이션·도로/차선·추적·World Model 및 Layer별 기술명세 통합본"
        fig_sec = "학습 곡선 및 시연 키프레임 시각 자료"
        figs = [
            (ROOT / "results/v2_improved/outputs/train_results/results.png", "v2 YOLO11m @ 960 학습 손실 및 Val mAP 수렴 곡선"),
            (ROOT / "results/v2_improved/outputs/train_results/confusion_matrix_normalized.png", "v2 YOLO11m @ 960 정규화 혼동 행렬 (Car / Pedestrian / Cyclist)"),
            (ROOT / "results/v3_bev_fixed/world_outputs/world_model_0015_staged/keyframes/seq_0015_frame_000045_stage5.png", "v3 시퀀스 0015 Frame 45: BEV 타원 종횡비·오경보 수정 후 정상 주행(Caution 0.25)"),
            (ROOT / "results/v3_bev_fixed/world_outputs/world_model_0019_staged/keyframes/seq_0019_frame_000055_stage5.png", "v3 시퀀스 0019 Frame 55: 우측 보도 보행자 비경보 유지 및 정면 접근 차량 Braking(0.72)"),
            (ROOT / "results/v4_road_lane/detect_outputs/outputs/train_results/results.png", "v4 YOLO11l @ 1280 (중복 제거 KITTI-Det 병합) 학습 손실 및 Val mAP 수렴 곡선"),
            (ROOT / "results/v4_road_lane/detect_outputs/outputs/train_results/BoxPR_curve.png", "v4 Val Precision-Recall 곡선 (TTA all classes mAP@0.5 = 0.740)"),
            (ROOT / "results/v4_road_lane/detect_outputs/outputs/train_results/BoxF1_curve.png", "v4 Val F1-Confidence 곡선 (all classes 0.71 at conf 0.217)"),
            (ROOT / "results/v4_road_lane/detect_outputs/outputs/train_results/confusion_matrix_normalized.png", "v4 정규화 혼동 행렬 (Car 0.90 / Pedestrian 0.87 / Cyclist 0.55)"),
            (ROOT / "results/v4_road_lane/detect_outputs/outputs/train_results/test_BoxPR_curve.png", "v4 Held-Out Test Precision-Recall 곡선 (TTA all classes mAP@0.5 = 0.670)"),
            (ROOT / "results/v4_road_lane/detect_outputs/outputs/train_results/test_BoxF1_curve.png", "v4 Held-Out Test F1-Confidence 곡선 (all classes 0.68 at conf 0.568)"),
            (ROOT / "results/v4_road_lane/road_outputs/road_lane_outputs/road_lane_preview_0015_f0045.png", "v4 SegFormer-B2 도로·비도로·자차로 세그멘테이션 및 2차 다항식 차선 프리뷰 (0015 f0045)"),
            (ROOT / "results/v4_road_lane/world_outputs/v4_world_outputs/world_model_0015_staged/keyframes/seq_0015_frame_000045_stage5.png", "v4 시퀀스 0015 Frame 45: 도로·차선 마스킹 및 BEV 도로 폴리곤·곡률 차선 통합 출력"),
            (ROOT / "results/v4_road_lane/world_outputs/v4_world_outputs/world_model_0019_staged/keyframes/seq_0019_frame_000055_stage5.png", "v4 시퀀스 0019 Frame 55: 비도로/차선 컨텍스트 배지([LANE]/[NON-RD]) 및 Safety Shield 통합 출력"),
        ]
    else:
        m_title = "Camera-Centric Temporal Ego-centric World Model Final Report (ENG)"
        m_sub = "Consolidated KITTI Tracking Report: Detection, Segmentation, Road/Lane, Tracking, World Model, and Per-Layer Specs"
        fig_sec = "Training Curves and Demonstration Keyframes"
        figs = [
            (ROOT / "results/v2_improved/outputs/train_results/results.png", "v2 YOLO11m @ 960 Training Losses and Validation mAP Convergence"),
            (ROOT / "results/v2_improved/outputs/train_results/confusion_matrix_normalized.png", "v2 YOLO11m @ 960 Normalized Confusion Matrix (Car / Pedestrian / Cyclist)"),
            (ROOT / "results/v3_bev_fixed/world_outputs/world_model_0015_staged/keyframes/seq_0015_frame_000045_stage5.png", "v3 Seq 0015 Frame 45: Aspect-Corrected BEV Ellipses & False-Alarm Suppression (Caution 0.25)"),
            (ROOT / "results/v3_bev_fixed/world_outputs/world_model_0019_staged/keyframes/seq_0019_frame_000055_stage5.png", "v3 Seq 0019 Frame 55: Sidewalk Pedestrian Non-Alert & Approaching Vehicle Braking (0.72)"),
            (ROOT / "results/v4_road_lane/detect_outputs/outputs/train_results/results.png", "v4 YOLO11l @ 1280 (Deduped KITTI-Det Merge) Training Losses and Validation mAP Convergence"),
            (ROOT / "results/v4_road_lane/detect_outputs/outputs/train_results/BoxPR_curve.png", "v4 Val Precision-Recall Curve (TTA all classes mAP@0.5 = 0.740)"),
            (ROOT / "results/v4_road_lane/detect_outputs/outputs/train_results/BoxF1_curve.png", "v4 Val F1-Confidence Curve (all classes 0.71 at conf 0.217)"),
            (ROOT / "results/v4_road_lane/detect_outputs/outputs/train_results/confusion_matrix_normalized.png", "v4 Normalized Confusion Matrix (Car 0.90 / Pedestrian 0.87 / Cyclist 0.55)"),
            (ROOT / "results/v4_road_lane/detect_outputs/outputs/train_results/test_BoxPR_curve.png", "v4 Held-Out Test Precision-Recall Curve (TTA all classes mAP@0.5 = 0.670)"),
            (ROOT / "results/v4_road_lane/detect_outputs/outputs/train_results/test_BoxF1_curve.png", "v4 Held-Out Test F1-Confidence Curve (all classes 0.68 at conf 0.568)"),
            (ROOT / "results/v4_road_lane/road_outputs/road_lane_outputs/road_lane_preview_0015_f0045.png", "v4 SegFormer-B2 Road/Non-Road/Ego-Lane Segmentation & Quadratic Lane Preview (0015 f0045)"),
            (ROOT / "results/v4_road_lane/world_outputs/v4_world_outputs/world_model_0015_staged/keyframes/seq_0015_frame_000045_stage5.png", "v4 Seq 0015 Frame 45: Integrated Road/Lane Masking & BEV Road Polygon / Curvilinear Lanes"),
            (ROOT / "results/v4_road_lane/world_outputs/v4_world_outputs/world_model_0019_staged/keyframes/seq_0019_frame_000055_stage5.png", "v4 Seq 0019 Frame 55: Road/Lane Context Badges ([LANE]/[NON-RD]) & Safety Shield Integration"),
        ]

    parts: List[str] = [
        tex_preamble(m_title, m_sub, lang=lang),
        r"\tableofcontents",
        r"\newpage",
    ]

    for md_name, _, doc_desc in SOURCE_DOCS:
        md_text = (src_md_dir / md_name).read_text(encoding="utf-8")
        lines = md_text.splitlines()
        part_title = doc_desc
        if lines and lines[0].startswith("# "):
            part_title = lines[0][2:].strip()
            md_text = "\n".join(lines[1:])
        parts.append(r"\section{" + _escape_latex_plain(part_title) + "}")
        parts.append(markdown_to_latex_body(md_text, heading_shift=0))
        parts.append(r"\newpage")

    parts.append(r"\section{" + _escape_latex_plain(fig_sec) + "}")
    for fpath, cap in figs:
        if fpath.exists():
            parts.append(r"\begin{figure}[htbp]\centering")
            parts.append(r"\includegraphics[width=0.92\textwidth]{" + str(fpath) + "}")
            parts.append(r"\caption{" + _escape_latex_plain(cap) + "}")
            parts.append(r"\end{figure}")

    parts.append(r"\end{document}")
    master_tex.write_text("\n".join(parts), encoding="utf-8")
    compile_tectonic(master_tex, master_pdf)
    print(f"[DONE:{suffix}] Consolidated PDF ready: {master_pdf}")
    return master_pdf


def main() -> None:
    MD_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    ko_pdf = build_language_bundle("ko")
    en_pdf = build_language_bundle("en")

    spec_dir = ROOT / "report to specific"
    spec_dir.mkdir(parents=True, exist_ok=True)
    for src_pdf in (ko_pdf, en_pdf):
        shutil.copy2(src_pdf, spec_dir / src_pdf.name)
        print(f"[COPY] {src_pdf.name} -> {spec_dir / src_pdf.name}")


if __name__ == "__main__":
    main()
