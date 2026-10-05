"""
Automated PDF Generation Script for Assignment 2
Uses markdown parsing + Headless Chrome rendering for publication-quality output.
"""

import argparse
import html
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

import markdown

SOURCE_MD = "DDM501_Assignment2_ML_Pipeline_Design.md"
TARGET_HTML = "DDM501_Assignment2_Report.html"
MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS_STYLES = """
@page {
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
    @bottom-right {
        content: "Page " counter(page);
        font-family: 'Inter', sans-serif;
        font-size: 9pt;
        color: #64748b;
    }
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1e293b;
    line-height: 1.6;
    font-size: 10pt;
    padding: 0;
    margin: 0;
}

h1 {
    color: #0f172a;
    font-size: 20pt;
    font-weight: 700;
    border-bottom: 2px solid #0284c7;
    padding-bottom: 6px;
    margin-top: 24pt;
    page-break-after: avoid;
}

h2 {
    color: #0369a1;
    font-size: 14pt;
    font-weight: 600;
    margin-top: 18pt;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
    page-break-after: avoid;
}

h3 {
    color: #0f172a;
    font-size: 11pt;
    font-weight: 600;
    margin-top: 14pt;
    page-break-after: avoid;
}

h4 {
    color: #334155;
    font-size: 10pt;
    font-weight: 600;
    margin-top: 10pt;
    page-break-after: avoid;
}

p, li {
    font-size: 9.5pt;
    color: #334155;
    text-align: justify;
}

ul, ol {
    margin-top: 4px;
    margin-bottom: 8px;
    padding-left: 20px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 8pt;
    page-break-inside: avoid;
}

th {
    background-color: #0f172a;
    color: #ffffff;
    font-weight: 600;
    text-align: left;
    padding: 7px 9px;
    border: 1px solid #0f172a;
}

td {
    padding: 6px 9px;
    border: 1px solid #cbd5e1;
    vertical-align: top;
}

tr:nth-child(even) {
    background-color: #f8fafc;
}

blockquote {
    border-left: 4px solid #0284c7;
    background-color: #f0f9ff;
    padding: 10px 14px;
    margin: 12px 0;
    border-radius: 0 4px 4px 0;
    color: #0369a1;
    font-style: italic;
}

.toc {
    padding: 12px 18px;
    border: 1px solid #cbd5e1;
    background: #f8fafc;
}

.toc ul {
    list-style: none;
    padding-left: 14px;
}

.toc a {
    color: #0369a1;
    text-decoration: none;
}

.page-break {
    page-break-after: always;
}

pre {
    background-color: #0f172a;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
    font-size: 8pt;
    padding: 12px;
    border-radius: 6px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-break: break-word;
    break-inside: auto;
}

.mermaid {
    margin: 16px auto;
    text-align: center;
    break-inside: auto;
}

.mermaid svg {
    display: block;
    max-width: 100% !important;
    height: auto !important;
    margin: 0 auto;
}

code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5pt;
    background-color: #f1f5f9;
    color: #0f172a;
    padding: 2px 4px;
    border-radius: 4px;
}

pre code {
    background-color: transparent;
    color: inherit;
    padding: 0;
}

@media print {
    .mermaid svg {
        max-height: 235mm;
    }
}
"""

def _chrome_binary() -> str:
    configured_binary = os.getenv("CHROME_BIN")
    if configured_binary:
        if not Path(configured_binary).is_file():
            raise FileNotFoundError(f"CHROME_BIN does not point to a file: {configured_binary}")
        return configured_binary

    for binary in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        executable = shutil.which(binary)
        if executable:
            return executable
    if Path(MAC_CHROME).is_file():
        return MAC_CHROME
    raise FileNotFoundError("Google Chrome or Chromium is required; set CHROME_BIN to its executable.")


def _filename_slug(value: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_")
    if not slug:
        raise ValueError("Student name must contain at least one ASCII letter or digit.")
    return slug


def generate_pdf(student_name: str, student_id: str) -> Path:
    source = Path(SOURCE_MD)
    if not source.is_file():
        raise FileNotFoundError(f"Report source file not found: {source}")
    if not re.fullmatch(r"[A-Za-z0-9-]+", student_id):
        raise ValueError("Student ID may contain only letters, digits, and hyphens.")

    name_slug = _filename_slug(student_name)
    target_pdf = Path(f"DDM501_Assignment2_{student_id}_{name_slug}.pdf")
    md_text = source.read_text(encoding="utf-8")
    md_text = re.sub(
        r"(?m)^(\*\*Student Name:\*\* ).*$",
        lambda match: match.group(1) + student_name + "<br>",
        md_text,
    )
    md_text = re.sub(
        r"(?m)^(\*\*Student ID:\*\* ).*$",
        lambda match: match.group(1) + student_id + "<br>",
        md_text,
    )
    md_text = md_text.replace("[StudentID]_[Name]", f"{student_id}_{name_slug}")
    md_text = md_text.replace("[StudentID]", student_id).replace("[Name]", student_name)

    html_content = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"],
        extension_configs={"toc": {"toc_depth": "2-3"}},
    )
    html_content, diagram_count = re.subn(
        r"<pre><code class=\"language-mermaid\">(.*?)</code></pre>",
        lambda match: f'<div class="mermaid">{html.escape(html.unescape(match.group(1)))}</div>',
        html_content,
        flags=re.DOTALL,
    )
    if diagram_count == 0:
        raise ValueError("The report contains no Mermaid diagrams to render.")

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DDM501 Individual Assignment 2</title>
<script>
window.MathJax = {{
    tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
    }},
    startup: {{ typeset: false }},
    svg: {{ fontCache: 'global' }}
}};
window.addEventListener("DOMContentLoaded", async () => {{
    try {{
        mermaid.initialize({{ startOnLoad: false, securityLevel: "strict", theme: "neutral" }});
        for (const diagram of document.querySelectorAll(".mermaid")) {{
            await mermaid.run({{ nodes: [diagram] }});
        }}
        await MathJax.startup.promise;
        await MathJax.typesetPromise();
        document.documentElement.dataset.renderComplete = "true";
    }} catch (error) {{
        document.documentElement.dataset.renderError = error?.message || JSON.stringify(error);
    }}
}});
</script>
<script defer src="{(Path(__file__).resolve().parents[1] / 'node_modules/mermaid/dist/mermaid.min.js').as_uri()}"></script>
<script defer src="{(Path(__file__).resolve().parents[1] / 'node_modules/mathjax-full/es5/tex-svg.js').as_uri()}"></script>
<style>
{CSS_STYLES}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""

    Path(TARGET_HTML).write_text(full_html, encoding="utf-8")
    chrome = _chrome_binary()
    browser_flags = [
        chrome,
        "--headless",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--virtual-time-budget=15000",
    ]
    rendered_dom = subprocess.run(
        [*browser_flags, "--dump-dom", Path(TARGET_HTML).resolve().as_uri()],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if rendered_dom.returncode != 0:
        raise RuntimeError(f"Chrome HTML rendering failed: {rendered_dom.stderr.strip()}")
    if 'data-render-complete="true"' not in rendered_dom.stdout:
        detail = re.search(r'data-render-error="([^"]*)"', rendered_dom.stdout)
        raise RuntimeError(f"Math or diagram rendering failed: {detail.group(1) if detail else rendered_dom.stderr.strip()}")
    if rendered_dom.stdout.count("<svg") < diagram_count or "<mjx-container" not in rendered_dom.stdout:
        raise RuntimeError("Chrome did not produce all expected Mermaid diagrams and MathJax equations.")

    static_html = re.sub(
        r"<script\b[^>]*>.*?</script\s*>",
        "",
        rendered_dom.stdout,
        flags=re.DOTALL | re.IGNORECASE,
    )
    Path(TARGET_HTML).write_text(static_html, encoding="utf-8")
    result = subprocess.run(
        [
            *browser_flags,
            "--no-pdf-header-footer",
            f"--print-to-pdf={target_pdf.resolve()}",
            Path(TARGET_HTML).resolve().as_uri(),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Chrome PDF generation failed: {result.stderr.strip()}")
    if not target_pdf.is_file() or target_pdf.stat().st_size == 0:
        raise RuntimeError("Chrome exited without producing a non-empty PDF.")
    return target_pdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Assignment 2 PDF report.")
    parser.add_argument("--student-name", required=True)
    parser.add_argument("--student-id", required=True)
    args = parser.parse_args()
    output = generate_pdf(args.student_name, args.student_id)
    print(f"Generated {output} ({output.stat().st_size / 1024:.1f} KB)")
