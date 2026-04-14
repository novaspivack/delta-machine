#!/usr/bin/env python3
"""Profile DSAC to identify performance bottlenecks."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root.parent))

from delta_machine.cli import main as cli_main


def profile_cli(args):
    """Run CLI with profiling enabled."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        # Parse CLI args and run
        sys.argv = ["delta_machine.cli", "headless"] + args
        cli_main()
    finally:
        profiler.disable()
        
        # Save profile stats
        stats_file = project_root / "runs" / "profile_stats.prof"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(stats_file))
        
        # Print top time consumers
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        print("\n" + "=" * 80)
        print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
        print("=" * 80)
        stats.print_stats(30)
        
        print("\n" + "=" * 80)
        print("TOP 30 FUNCTIONS BY TOTAL TIME")
        print("=" * 80)
        stats.sort_stats("tottime")
        stats.print_stats(30)
        
        print(f"\nProfile data saved to: {stats_file}")
        print("To view interactively: python -m pstats", stats_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile DSAC execution")
    parser.add_argument("--scenario", required=True, help="Scenario name")
    parser.add_argument("--scenario-dir", default="scenarios", help="Scenario directory")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to profile")
    parser.add_argument("--workers", type=int, default=9, help="Number of workers")
    
    args = parser.parse_args()
    
    cli_args = [
        "--scenario", args.scenario,
        "--scenario-dir", args.scenario_dir,
        "--steps", str(args.steps),
        "--workers", str(args.workers),
    ]
    
    profile_cli(cli_args)

