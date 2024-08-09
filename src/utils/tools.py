import os
from typing import Optional


def get_figure_path(chapter: str, figname: Optional[str] = None) -> str:
    root_thesis = "/home/nrieger/Projects/phd/thesis"
    if figname is not None:
        return os.path.join(root_thesis, "content", chapter, "figs", figname)
    else:
        return os.path.join(root_thesis, "content", chapter, "figs")
