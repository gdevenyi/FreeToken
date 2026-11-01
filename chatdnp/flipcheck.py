"""Per-question flips between two gsm8k_probe.py runs.

Prints one line for each question where one run is correct and the other is wrong, then one JSON
line with the counts. Exit status is 2 if the two files do not have the same questions.

usage: flipcheck.py A.json B.json
"""

import json
import sys


def load(path):
    with open(path) as f:
        return {r["index"]: r for r in json.load(f)["results"]}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(a) & set(b))
    if any(a[i]["question"] != b[i]["question"] for i in common):
        print("the two runs have different questions at the same index", file=sys.stderr)
        sys.exit(2)
    counts = {"both_correct": 0, "a_only": 0, "b_only": 0, "both_wrong": 0}
    for i in common:
        ca, cb = a[i]["correct"], b[i]["correct"]
        if ca and cb:
            counts["both_correct"] += 1
        elif ca or cb:
            counts["a_only" if ca else "b_only"] += 1
            word = "A-right B-wrong" if ca else "A-wrong B-right"
            print(f"flip {i}: {word} gold={a[i]['gold']} a={a[i]['pred']} b={b[i]['pred']}")
        else:
            counts["both_wrong"] += 1
    only_one = len(set(a) ^ set(b))
    print(json.dumps({"a": sys.argv[1], "b": sys.argv[2], "common": len(common), "only_in_one": only_one, **counts}))


if __name__ == "__main__":
    main()
