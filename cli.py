#!/usr/bin/env python3
"""CLI entry point for the sensor-crossover experiment."""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def cmd_run(args):
    """Run the full experiment pipeline."""
    from run_experiment import main
    main()


def cmd_publish(args):
    """Stage HTML reports in results/ for GitHub Pages deployment."""
    reports = []
    for name in ("index.html", "dataset.html"):
        path = os.path.join(cfg.RESULTS_DIR, name)
        if os.path.exists(path):
            reports.append(name)

    if not reports:
        print("No HTML reports found in results/. Run the experiment first.")
        sys.exit(1)

    # Ensure .nojekyll exists so GitHub Pages serves raw HTML
    nojekyll = os.path.join(cfg.RESULTS_DIR, ".nojekyll")
    if not os.path.exists(nojekyll):
        open(nojekyll, "w").close()

    print(f"Reports ready in results/ ({len(reports)} files):")
    for f in reports:
        print(f"  results/{f}")
    print("\nCommit and push to deploy via GitHub Pages (source: results/).")


def main():
    parser = argparse.ArgumentParser(
        prog="sensor-crossover",
        description="Finite-sample crossover in sensor classification",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Run the full experiment pipeline")
    sub.add_parser("publish", help="Verify reports are ready for GitHub Pages")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    {"run": cmd_run, "publish": cmd_publish}[args.command](args)


if __name__ == "__main__":
    main()
