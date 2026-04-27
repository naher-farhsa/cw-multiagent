from typing import Annotated, Optional
from langchain_core.tools import tool
from src.config.settings import OUTPUT_DIR


@tool
def write_document(
    content: Annotated[str, "Text content to be written to the document"],
    file_name: Annotated[str, "File name to save the document (e.g. 'article.md')"],
) -> str:
    """Create and save a text document into the doc-result folder."""
    target = OUTPUT_DIR / file_name
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nDocument saved to: {target}")
    return f"Document saved to '{target}'."


@tool
def read_document(
    file_name: Annotated[str, "File name to read from the doc-result folder"],
    start: Annotated[Optional[int], "Start line. Default is 0"] = None,
    end: Annotated[Optional[int], "End line. Default is None"] = None,
) -> str:
    """Read a document from the doc-result folder."""
    target = OUTPUT_DIR / file_name
    with open(target, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if start is None:
        start = 0
    return "".join(lines[start:end])
