"""
tools — LangChain @tool functions exposed to the agent teams.

Modules:
    web_tools  — tavily_tool:      Tavily Search wrapper used by the Research Agent Team.
                                   Covers the 'web scrapper' role in the diagram.

    file_tools — write_document:   Saves Markdown articles to doc-result/.
                                   Covers the 'writer' role in the Writing Agent Team.
                 read_document:    Reads back files from doc-result/ for review.
                                   Covers the 'note-taker' role in the Writing Agent Team.
                 generate_chart:   Creates matplotlib bar/line/pie charts as .png files.
                                   Covers the 'chart-generator' role in the Writing Agent Team.
"""
