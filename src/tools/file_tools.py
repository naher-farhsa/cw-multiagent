"""
file_tools.py — File I/O and chart-generation tools used by the Writing Agent Team.

This module provides three LangChain @tool-decorated functions that the writer
agent uses to persist, retrieve, and visualise content:

  * ``write_document``  — saves a Markdown article to the doc-result folder.
                          → fulfils the **writer** role in the Writing Agent Team.

  * ``read_document``   — reads back a previously saved file so the writer can
                          review or extend existing content.
                          → fulfils the **note-taker** role in the Writing Agent Team.

  * ``generate_chart``  — creates a matplotlib bar / line / pie chart from data
                          provided by the LLM and saves it as a .png image to
                          the doc-result folder.
                          → fulfils the **chart-generator** role in the Writing Agent Team.

The output directory (``OUTPUT_DIR``) is resolved in settings.py; it is created
automatically when settings.py is imported, so these tools can always safely
open or write files within it.

Why @tool?
  LangGraph's ``create_react_agent`` discovers callable tools via the ``@tool``
  decorator, which attaches name / description / argument-schema metadata the
  LLM reads to decide when and how to call each function.
"""

from typing import Annotated, List, Optional
from langchain_core.tools import tool      # decorator that exposes a function as an LLM tool
import matplotlib                          # imported first to set the backend before pyplot
matplotlib.use("Agg")                      # non-interactive backend: renders to file without a GUI window
import matplotlib.pyplot as plt            # the actual plotting API
from src.config.settings import OUTPUT_DIR


# ── write_document ─────────────────────────────────────────────────────────────
@tool
def write_document(
    content: Annotated[str, "Text content to be written to the document"],
    file_name: Annotated[str, "File name to save the document (e.g. 'article.md')"],
) -> str:
    """Create and save a text document to the doc-result output folder.

    The writer agent calls this tool after producing a finished Markdown
    article.  The file is always written with UTF-8 encoding to support
    international characters.

    Args:
        content:   The full text (Markdown or plain text) to write.
        file_name: Target filename, e.g. ``"ai_trends.md"``.  The file is
                   placed inside ``OUTPUT_DIR``; any path separators in
                   ``file_name`` are treated as part of the name, not
                   subdirectories.

    Returns:
        A confirmation string with the resolved file path so the LLM can
        relay success to the supervisor.
    """
    # Construct the full path by joining the configured output directory with
    # the agent-supplied filename.
    target = OUTPUT_DIR / file_name

    # Open in write mode ("w"); overwrites if the file already exists.
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)

    # Also print to the terminal so the human operator can see where the file landed.
    print(f"\nDocument saved to: {target}")
    return f"Document saved to '{target}'."


# ── read_document ──────────────────────────────────────────────────────────────
@tool
def read_document(
    file_name: Annotated[str, "File name to read from the doc-result folder"],
    start: Annotated[Optional[int], "First line to return (0-indexed). Default: 0"] = None,
    end: Annotated[Optional[int], "Last line to return (exclusive). Default: all lines"] = None,
) -> str:
    """Read a previously saved document from the doc-result output folder.

    Supports optional line-range slicing so the writer agent can inspect a
    specific section of a large file without loading the entire content into
    the LLM context.

    Args:
        file_name: Name of the file to read (must already exist in ``OUTPUT_DIR``).
        start:     0-indexed line number to start from.  ``None`` means line 0.
        end:       Exclusive upper bound for the line range.  ``None`` means
                   read to the end of the file.

    Returns:
        A single string containing the requested lines joined together,
        preserving their original newline characters.

    Raises:
        FileNotFoundError: If ``file_name`` does not exist in ``OUTPUT_DIR``.
    """
    target = OUTPUT_DIR / file_name

    with open(target, "r", encoding="utf-8") as f:
        lines = f.readlines()  # read all lines; each ends with '\n'

    # Default start to 0 if the agent did not specify a starting line
    if start is None:
        start = 0

    # Python list slicing handles end=None correctly (returns to end of list)
    return "".join(lines[start:end])


# ── generate_chart ─────────────────────────────────────────────────────────────
@tool
def generate_chart(
    title: Annotated[str, "Title displayed at the top of the chart"],
    labels: Annotated[List[str], "Category labels shown on the X-axis (or pie slices)"],
    values: Annotated[List[float], "Numeric values corresponding to each label"],
    file_name: Annotated[str, "Output filename for the chart image (e.g. 'stats.png')"],
    chart_type: Annotated[
        str,
        "Chart style to render. Accepted values: 'bar', 'line', 'pie'. Defaults to 'bar'.",
    ] = "bar",
    x_label: Annotated[str, "Label for the X-axis (ignored for pie charts)"] = "",
    y_label: Annotated[str, "Label for the Y-axis (ignored for pie charts)"] = "",
) -> str:
    """Generate a matplotlib chart from LLM-supplied data and save it as a PNG.

    The chart-generator role of the Writing Agent Team calls this tool when the
    article would benefit from a visual — for example, a bar chart of statistics
    gathered by the researcher, or a line chart showing a trend over time.

    The resulting .png file is saved to ``OUTPUT_DIR`` (doc-result/) so the
    writer agent can reference it in the Markdown article with an image tag::

        ![Chart](doc-result/stats.png)

    Supported chart types:
        - ``"bar"``  — vertical bar chart; best for comparing discrete categories.
        - ``"line"`` — line chart; best for showing continuous trends over time.
        - ``"pie"``  — pie chart; best for showing proportional composition.

    Args:
        title:      Text displayed as the chart title (e.g. ``"AI Market Share 2024"``).
        labels:     List of category names, one per data point
                    (e.g. ``["Google", "OpenAI", "Anthropic"]``).
        values:     Numeric value for each label in the same order
                    (e.g. ``[40.5, 35.2, 24.3]``).
        file_name:  Filename for the saved image, e.g. ``"market_share.png"``.
                    The file is placed inside ``OUTPUT_DIR``.
        chart_type: One of ``"bar"``, ``"line"``, or ``"pie"``.
                    Defaults to ``"bar"`` if not specified.
        x_label:    Optional label for the horizontal axis (bar / line only).
        y_label:    Optional label for the vertical axis (bar / line only).

    Returns:
        A confirmation string with the resolved file path on success, or an
        error message string if rendering or saving fails (so the LLM can
        report the problem without crashing the workflow).

    Raises:
        Does not raise — all exceptions are caught and returned as strings
        so the agent workflow continues gracefully.
    """
    # ── Input validation ───────────────────────────────────────────────────────
    if len(labels) != len(values):
        return (
            f"Error: 'labels' has {len(labels)} items but 'values' has {len(values)}. "
            "They must be the same length."
        )

    if not labels:
        return "Error: 'labels' and 'values' must not be empty."

    # Normalise chart_type so "Bar", "BAR", etc. all work
    chart_type = chart_type.lower().strip()
    if chart_type not in ("bar", "line", "pie"):
        return (
            f"Error: Unknown chart_type '{chart_type}'. "
            "Choose one of: 'bar', 'line', 'pie'."
        )

    try:
        # ── Create figure ──────────────────────────────────────────────────────
        # figsize=(10, 6) gives a landscape aspect ratio suitable for articles.
        fig, ax = plt.subplots(figsize=(10, 6))

        # ── Render the chosen chart type ───────────────────────────────────────
        if chart_type == "bar":
            # x_pos: integer positions along the x-axis for each bar
            x_pos = range(len(labels))
            bars = ax.bar(x_pos, values, color="#4C9BE8", edgecolor="white", linewidth=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)

            # Add the numeric value on top of each bar for readability
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # horizontally centred on bar
                    bar.get_height() + max(values) * 0.01,  # slightly above the bar top
                    f"{val:g}",            # :g removes trailing zeros (e.g. 3.0 → 3)
                    ha="center", va="bottom", fontsize=9, color="#222222",
                )

        elif chart_type == "line":
            ax.plot(labels, values, marker="o", color="#4C9BE8", linewidth=2, markersize=7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
            # Shade the area under the line for visual emphasis
            ax.fill_between(range(len(labels)), values, alpha=0.15, color="#4C9BE8")

        elif chart_type == "pie":
            # autopct shows the percentage inside each slice
            ax.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",   # one decimal place, e.g. "34.5%"
                startangle=90,       # start from the top of the circle
                colors=plt.cm.Paired.colors,  # use a distinct colour palette
            )
            ax.axis("equal")  # ensure the pie is circular, not elliptical

        # ── Shared styling ─────────────────────────────────────────────────────
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

        if chart_type != "pie":
            # Axis labels are only meaningful for bar and line charts
            if x_label:
                ax.set_xlabel(x_label, fontsize=11)
            if y_label:
                ax.set_ylabel(y_label, fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.4)  # subtle horizontal grid lines
            ax.spines[["top", "right"]].set_visible(False)  # remove chart border clutter

        fig.tight_layout()  # prevent labels from being clipped at the figure edge

        # ── Save to disk ───────────────────────────────────────────────────────
        target = OUTPUT_DIR / file_name
        fig.savefig(target, dpi=150, bbox_inches="tight")  # dpi=150 → crisp but not huge
        plt.close(fig)  # IMPORTANT: release memory; prevents figure accumulation in long runs

        print(f"\nChart saved to: {target}")
        return (
            f"Chart '{title}' saved to '{target}'. "
            f"You can embed it in the Markdown article with: ![{title}]({target})"
        )

    except Exception as exc:
        # Return the error as a string so the agent can report it without crashing
        plt.close("all")  # clean up any partially created figures
        return f"Error generating chart: {exc}"
