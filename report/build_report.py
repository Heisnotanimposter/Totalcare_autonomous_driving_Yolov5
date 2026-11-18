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


def tex_preamble(title: str, subtitle: str) -> str:
    return rf"""\documentclass[10.5pt,a4paper]{{article}}
\usepackage{{kotex}}
\usepackage{{geometry}}
\geometry{{margin=18mm,top=20mm,bottom=20mm}}
\usepackage{{fontspec}}
\setmainhangulfont{{Apple SD Gothic Neo}}[BoldFont={{* Bold}}]
\setsanshangulfont{{Apple SD Gothic Neo}}[BoldFont={{* Bold}}]
\setmonofont{{Menlo}}[Scale=0.82]
\usepackage{{amsmath,amssymb}}
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


def main() -> None:
    MD_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Sync root markdown files into report/md/
    for dst_name, src_rel, _ in SOURCE_DOCS:
        if src_rel:
            shutil.copy2(ROOT / src_rel, MD_DIR / dst_name)

    # 2. Convert each MD -> standalone LaTeX -> standalone PDF
    for md_name, _, doc_desc in SOURCE_DOCS:
        md_path = MD_DIR / md_name
        stem = md_path.stem
        tex_path = TEX_DIR / f"{stem}.tex"
        pdf_path = PDF_DIR / f"{stem}.pdf"

        raw_md = md_path.read_text(encoding="utf-8")
        lines = raw_md.splitlines()
        title = doc_desc
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip()
            raw_md = "\n".join(lines[1:])

        # In standalone docs, top H1 was stripped so shift `##` (-1) -> `\section`
        tex_doc = tex_preamble(title, doc_desc) + markdown_to_latex_body(raw_md, heading_shift=-1) + "\n\\end{document}\n"
        tex_path.write_text(tex_doc, encoding="utf-8")
        compile_tectonic(tex_path, pdf_path)
        print(f"[OK] {md_path.name} -> {tex_path.name} -> {pdf_path.name}")

    # 3. Build Master Consolidated Final Report PDF (all 7 docs + visual figures)
    master_tex = TEX_DIR / "Totalcare_Autonomous_Driving_Report.tex"
    master_pdf = REPORT_DIR / "Totalcare_Autonomous_Driving_Report.pdf"

    parts: List[str] = [
        tex_preamble(
            "Camera-Centric Temporal Ego-centric World Model 최종 보고서",
            "KITTI Tracking 기반 검출·세그멘테이션·추적·World Model 및 Layer별 기술명세 통합본",
        ),
        r"\tableofcontents",
        r"\newpage",
    ]

    for md_name, _, doc_desc in SOURCE_DOCS:
        md_text = (MD_DIR / md_name).read_text(encoding="utf-8")
        lines = md_text.splitlines()
        part_title = doc_desc
        if lines and lines[0].startswith("# "):
            part_title = lines[0][2:].strip()
            md_text = "\n".join(lines[1:])
        parts.append(r"\section{" + _escape_latex_plain(part_title) + "}")
        # Inside master book, `\section` is chapter heading so `##` (`heading_shift=0`) becomes `\subsection` cleanly!
        parts.append(markdown_to_latex_body(md_text, heading_shift=0))
        parts.append(r"\newpage")

    figs = [
        (ROOT / "results/v2_improved/outputs/train_results/results.png", "v2 YOLO11m @ 960 학습 손실 및 Val mAP 수렴 곡선"),
        (ROOT / "results/v2_improved/outputs/train_results/confusion_matrix_normalized.png", "v2 YOLO11m @ 960 정규화 혼동 행렬 (Car / Pedestrian / Cyclist)"),
        (ROOT / "results/v3_bev_fixed/world_outputs/world_model_0015_staged/keyframes/seq_0015_frame_000045_stage5.png", "시퀀스 0015 Frame 45: v3 BEV 타원 종횡비·오경보 수정 후 정상 주행(Caution 0.25) 및 MobileSAM 마스크"),
        (ROOT / "results/v3_bev_fixed/world_outputs/world_model_0015_staged/keyframes/seq_0015_frame_000055_stage5.png", "시퀀스 0015 Frame 55: 전방 횡단 자전거(#53) 소형 CONFLICT 마커 및 Speed Reduction(0.48)"),
        (ROOT / "results/v3_bev_fixed/world_outputs/world_model_0019_staged/keyframes/seq_0019_frame_000055_stage5.png", "시퀀스 0019 Frame 55: 우측 보도 보행자(#40) 비경보 유지 및 정면 접근 차량(#9) Braking Recommendation(0.72)"),
    ]
    parts.append(r"\section{학습 곡선 및 시연 키프레임 시각 자료}")
    for fpath, cap in figs:
        if fpath.exists():
            parts.append(r"\begin{figure}[htbp]\centering")
            parts.append(r"\includegraphics[width=0.92\textwidth]{" + str(fpath) + "}")
            parts.append(r"\caption{" + _escape_latex_plain(cap) + "}")
            parts.append(r"\end{figure}")

    parts.append(r"\end{document}")
    master_tex.write_text("\n".join(parts), encoding="utf-8")
    compile_tectonic(master_tex, master_pdf)
    print(f"[DONE] Consolidated PDF ready: {master_pdf}")


if __name__ == "__main__":
    main()
