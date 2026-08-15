"""
utils/table_extractor.py
Extracts actual tables from PDFs using pdfplumber lines strategy.
Works on any PDF with real table borders/lines.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber
import pandas as pd

logger = logging.getLogger(__name__)


def extract_tables_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract all real tables from a PDF using line detection.
    Works on any PDF that has bordered tables.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info(f"Extracting tables from: {path.name}")
    tables      = []
    table_index = 0

    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):

            # Try lines strategy first (works for bordered tables)
            page_tables = page.extract_tables({
                "vertical_strategy":   "lines",
                "horizontal_strategy": "lines",
            })

            # Fallback to lines_strict
            if not page_tables:
                page_tables = page.extract_tables({
                    "vertical_strategy":   "lines_strict",
                    "horizontal_strategy": "lines_strict",
                })

            for raw_table in page_tables:
                if not raw_table or len(raw_table) < 2:
                    continue

                # Clean cells — replace None, strip whitespace, fix newlines
                cleaned = []
                for row in raw_table:
                    cleaned_row = [
                        str(cell).replace('\n', ' ').strip()
                        if cell is not None else ""
                        for cell in row
                    ]
                    if any(c for c in cleaned_row):
                        cleaned.append(cleaned_row)

                if len(cleaned) < 2:
                    continue

                # First non-empty row as header
                headers = cleaned[0]
                rows    = cleaned[1:]

                # Remove completely empty rows
                rows = [r for r in rows if any(c for c in r)]

                if not rows:
                    continue

                # Build DataFrame
                try:
                    # Pad rows to match header length
                    max_cols = max(len(headers), max(len(r) for r in rows))
                    headers  = headers + [''] * (max_cols - len(headers))
                    rows     = [r + [''] * (max_cols - len(r)) for r in rows]

                    # Make headers unique
                    seen, unique_headers = {}, []
                    for h in headers:
                        h = h if h else "Column"
                        if h in seen:
                            seen[h] += 1
                            unique_headers.append(f"{h}_{seen[h]}")
                        else:
                            seen[h] = 0
                            unique_headers.append(h)

                    df = pd.DataFrame(rows, columns=unique_headers)
                    # Drop fully empty columns
                    df = df.loc[:, (df != '').any(axis=0)]
                    df = df.reset_index(drop=True)

                except Exception as e:
                    logger.warning(f"DataFrame build failed: {e}")
                    continue

                table_index += 1
                tables.append({
                    "index":   table_index,
                    "page":    page_num,
                    "source":  path.name,
                    "title":   f"Table {table_index} (Page {page_num})",
                    "headers": list(df.columns),
                    "rows":    df.values.tolist(),
                    "df":      df,
                    "context": f"Extracted from page {page_num} of {path.name}",
                    "shape":   f"{len(df)} rows × {len(df.columns)} columns",
                    "raw_text": df.to_string(index=False),
                })

    logger.info(f"Total tables extracted: {len(tables)} from {path.name}")
    return tables


def table_to_markdown(table: Dict[str, Any]) -> str:
    try:
        return table["df"].to_markdown(index=False)
    except Exception:
        return table.get("raw_text", "")


def table_to_text(table: Dict[str, Any]) -> str:
    df    = table["df"]
    title = table.get("title", f"Table {table['index']}")
    lines = [f"{title} ({table['shape']}):"]
    lines.append(" | ".join(str(h) for h in df.columns))
    lines.append("-" * 60)
    for _, row in df.iterrows():
        lines.append(" | ".join(str(v) for v in row.values))
    return "\n".join(lines)


def summarise_table(table: Dict[str, Any]) -> str:
    table_text = table_to_text(table)
    return (
        f"The following table was extracted from page {table['page']} "
        f"of a document called '{table['source']}'.\n\n"
        f"{table_text}\n\n"
        f"Please explain what this table represents, what each column means, "
        f"and what key information it contains. Be concise and professional."
    )