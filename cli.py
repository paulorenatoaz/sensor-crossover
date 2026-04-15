#!/usr/bin/env python3
"""CLI entry point for the sensor-crossover experiment."""

import argparse
import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def cmd_run(args):
    """Run the full experiment pipeline."""
    from run_experiment import main
    main()


def cmd_publish(args):
    """Copy HTML reports to docs/ for GitHub Pages."""
    docs_dir = os.path.join(cfg.PROJECT_ROOT, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    copied = []
    for name in ("index.html", "dataset.html"):
        src = os.path.join(cfg.RESULTS_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(docs_dir, name))
            copied.append(name)

    if not copied:
        print("No HTML reports found in results/. Run the experiment first.")
        sys.exit(1)

    # Create a minimal .nojekyll so GitHub Pages serves raw HTML
    open(os.path.join(docs_dir, ".nojekyll"), "w").close()

    print(f"Published {len(copied)} report(s) to docs/:")
    for f in copied:
        print(f"  docs/{f}")
    print("\nPush to GitHub and enable Pages (source: docs/) to go live.")


def main():
    parser = argparse.ArgumentParser(
        prog="sensor-crossover",
        description="Finite-sample crossover in sensor classification",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Run the full experiment pipeline")
    sub.add_parser("publish", help="Copy reports to docs/ for GitHub Pages")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    {"run": cmd_run, "publish": cmd_publish}[args.command](args)


if __name__ == "__main__":
    main()
