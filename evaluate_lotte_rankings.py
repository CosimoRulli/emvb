import argparse
from collections import defaultdict
import jsonlines
import os
import sys

def evaluate_dataset(gt_path, rankings_path, k,):

    rankings = defaultdict(list)
    with open(rankings_path, "r") as f:
        for line in f:
            items = line.strip().split("\t")
            qid, pid, rank = items[:3]
            qid = int(qid)
            pid = int(pid)
            rank = int(rank)
            rankings[qid].append(pid)
            assert rank == len(rankings[qid])

    success = 0
    

    with jsonlines.open(gt_path, mode="r") as f:
        for line in f:
            qid = int(line["qid"])
            answer_pids = set(line["answer_pids"])
            if len(set(rankings[qid][:k]).intersection(answer_pids)) > 0:
                success += 1
    print(
        f"Success@{k}: {success / len(rankings) * 100:.1f}"
    )


def main(args):
    # Success@5 
    evaluate_dataset(
                args.gt_path,
                args.rankings,
                5,
            )
    # Success@100
    evaluate_dataset(
                args.gt_path,
                args.rankings,
                100,
            )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoTTE evaluation script")
    parser.add_argument(
        "-gt",
        "--gt-path",
        type=str,
        required=True,
        help="Path to LoTTE pooled gt/relevance",
    )
    parser.add_argument(
        "-r",
        "--rankings",
        type=str,
        required=True,
        help="Path to rankings",
    )
    args = parser.parse_args()
    main(args)
