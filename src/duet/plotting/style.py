"""Apply the DUET publication matplotlib style."""
from __future__ import annotations

from pathlib import Path


_STYLESHEET_PATH = Path(__file__).parent / "duet_publication.mplstyle"


def apply_style() -> None:
    """Apply the DUET publication matplotlib stylesheet.

    Call once per script / notebook, before creating any figures. Sets
    rcParams for font family, sizes, line widths, tick style, spines,
    color cycle (Okabe-Ito), and save format.

    Does NOT call seaborn.set_theme() — seaborn plot functions respect
    matplotlib rcParams natively, and set_theme would override the values
    set here.
    """
    import matplotlib.pyplot as plt

    plt.style.use(str(_STYLESHEET_PATH))


def style_context():
    """Context manager that applies the stylesheet only inside a ``with`` block.

    Unlike :func:`apply_style`, the caller's global matplotlib settings are
    restored on exit. Used by the public API's plotting helpers.

    Examples
    --------
    >>> from duet.plotting import style_context
    >>> with style_context():
    ...     pass  # figures created here use the DUET stylesheet
    """
    import matplotlib.pyplot as plt

    return plt.style.context(str(_STYLESHEET_PATH))
