"""Export the graph assembled from the local corpus and the reviewed graph schema."""

import argparse
import json
import os
from pathlib import Path

from .corpus import DATA_DIR
from .knowledge import Retriever


def main():
    parser = argparse.ArgumentParser(description="Rebuild the graph and report normalization and conflicts.")
    parser.add_argument("--corpus-dir", type=Path, default=Path(os.getenv("DEEPVOCAL_DATA_DIR") or DATA_DIR) / "knowledge")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    graph = Retriever(corpus_dir=args.corpus_dir).graph
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(graph.report(), ensure_ascii=False, indent=2, default=str) + "\n",
                               encoding="utf-8")
    print(json.dumps(graph.counts(), indent=2))


if __name__ == "__main__":
    main()
