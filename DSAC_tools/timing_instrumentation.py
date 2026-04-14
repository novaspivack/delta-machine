"""Timing instrumentation for DSAC profiling.

Enable by setting DSAC_PROFILE=1 environment variable.
Uses file-based storage for multiprocessing compatibility.
"""

from __future__ import annotations

import os
import time
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_PROFILE_ENABLED = os.getenv("DSAC_PROFILE", "0") == "1"
_PROFILE_DIR = Path(os.getenv("DSAC_PROFILE_DIR", "/tmp/dsac_profile"))
_TIMINGS: Dict[str, List[float]] = defaultdict(list)
_COUNTS: Dict[str, int] = defaultdict(int)

if _PROFILE_ENABLED:
    _PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def profile_section(name: str):
    """Context manager for timing code sections."""
    if not _PROFILE_ENABLED:
        return _NullContext()
    return _TimingContext(name)


class _NullContext:
    """No-op context manager when profiling is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class _TimingContext:
    """Context manager that records timing."""
    def __init__(self, name: str):
        self.name = name
        self.start = None
    
    def __enter__(self):
        if _PROFILE_ENABLED:
            self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if _PROFILE_ENABLED and self.start is not None:
            elapsed = time.perf_counter() - self.start
            _TIMINGS[self.name].append(elapsed)
            _COUNTS[self.name] += 1
            # Also write to file for multiprocessing compatibility
            _write_timing(self.name, elapsed)


def get_timing_stats() -> Dict[str, Dict[str, float]]:
    """Get timing statistics (includes file-based timings from workers)."""
    # Merge in-memory and file-based timings
    all_timings = _load_all_timings()
    for name, timings in _TIMINGS.items():
        all_timings[name].extend(timings)
    
    stats = {}
    for name, timings in all_timings.items():
        if timings:
            stats[name] = {
                "total": sum(timings),
                "mean": sum(timings) / len(timings),
                "min": min(timings),
                "max": max(timings),
                "count": len(timings),
            }
    return stats


def print_timing_stats():
    """Print timing statistics."""
    if not _PROFILE_ENABLED:
        return
    
    stats = get_timing_stats()
    if not stats:
        return
    
    print("\n" + "=" * 80)
    print("TIMING STATISTICS")
    print("=" * 80)
    print(f"{'Section':<40} {'Total (s)':>12} {'Mean (ms)':>12} {'Count':>8} {'%':>8}")
    print("-" * 80)
    
    total_time = sum(s["total"] for s in stats.values())
    
    # Sort by total time
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True)
    
    for name, s in sorted_stats:
        pct = (s["total"] / total_time * 100) if total_time > 0 else 0
        print(f"{name:<40} {s['total']:>12.4f} {s['mean']*1000:>12.2f} {s['count']:>8} {pct:>7.1f}%")
    
    print("-" * 80)
    print(f"{'TOTAL':<40} {total_time:>12.4f}")
    print("=" * 80)


def _write_timing(name: str, elapsed: float):
    """Write timing to file for multiprocessing compatibility."""
    if not _PROFILE_ENABLED:
        return
    try:
        file_path = _PROFILE_DIR / f"{name}.timings"
        with open(file_path, "a") as f:
            f.write(f"{elapsed}\n")
    except Exception:
        pass  # Ignore errors in worker processes


def _load_all_timings() -> Dict[str, List[float]]:
    """Load all timing files."""
    all_timings = defaultdict(list)
    if not _PROFILE_DIR.exists():
        return all_timings
    
    for file_path in _PROFILE_DIR.glob("*.timings"):
        name = file_path.stem
        try:
            with open(file_path) as f:
                timings = [float(line.strip()) for line in f if line.strip()]
                all_timings[name].extend(timings)
        except Exception:
            pass
    
    return all_timings


def reset_timings():
    """Reset all timing data."""
    _TIMINGS.clear()
    _COUNTS.clear()
    if _PROFILE_DIR.exists():
        for file_path in _PROFILE_DIR.glob("*.timings"):
            try:
                file_path.unlink()
            except Exception:
                pass

