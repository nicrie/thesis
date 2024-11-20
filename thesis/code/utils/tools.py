import os


def get_figure_path(
    chapter: str, output_type: None | str = None, figname: None | str = None
) -> str:
    root_thesis = "/home/nrieger/Projects/phd/thesis"
    path = os.path.join(root_thesis, "content", chapter, "figs")
    if output_type is not None:
        path = os.path.join(path, output_type)
    if figname is not None:
        path = os.path.join(path, figname)
    return path
