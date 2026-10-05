import logging
import os
import platform

import psutil

logger = logging.getLogger(__name__)

_process = psutil.Process(os.getpid())
_IS_LINUX = platform.system() == "Linux"


def _parse_proc_status():
    """Parse /proc/self/status for VmRSS, RssAnon, and RssFile in one pass.

    Returns (rss_mb, rss_anon_mb, rss_file_mb). Any field that cannot be
    read is returned as None.
    """
    rss = anon = file = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024  # kB → MB
                elif line.startswith("RssAnon:"):
                    anon = int(line.split()[1]) / 1024
                elif line.startswith("RssFile:"):
                    file = int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return rss, anon, file


def log_memory(label: str) -> None:
    """Log current RSS of this process at DEBUG level.

    On Linux 4.5+, includes RssAnon (heap/stack — non-reclaimable) vs
    RssFile (mmap page cache — kernel-reclaimable) breakdown.

    Usage:
        log_memory("after symmetrization")
        # DEBUG: [mem] after symmetrization: RSS=782.3 MB  (anon=200.0 MB, file=582.3 MB)

    Lightweight (one open + one syscall on Linux, one syscall elsewhere),
    suitable for leaving in hot paths gated behind DEBUG log level.
    """
    if _IS_LINUX:
        rss_mb, anon_mb, file_mb = _parse_proc_status()
        # Fall back to psutil if /proc/self/status failed.
        if rss_mb is None:
            rss_mb = _process.memory_info().rss / (1024 * 1024)
        if anon_mb is not None and file_mb is not None:
            logger.debug(
                "[mem] %s: RSS=%.1f MB  (anon=%.1f MB, file=%.1f MB)",
                label, rss_mb, anon_mb, file_mb,
            )
            return

    else:
        rss_mb = _process.memory_info().rss / (1024 * 1024)

    logger.debug("[mem] %s: RSS=%.1f MB", label, rss_mb)
