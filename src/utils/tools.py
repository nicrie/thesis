import os


def get_figure_directory(chapter: str) -> str:
    root_thesis = "/home/nrieger/Projects/phd/thesis"
    return os.path.join(root_thesis, "content", chapter, "figs")
