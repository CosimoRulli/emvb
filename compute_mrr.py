import datetime
import tqdm
import json

import sys

def file_tqdm(file):
    print(f"#> Reading {file.name}")

    #with tqdm.tqdm(total=os.path.getsize(file.name) / 1024.0 / 1024.0, unit="MiB") as pbar:
    for line in file:
        yield line
            #pbar.update(len(line) / 1024.0 / 1024.0)

        #pbar.close()


def print_message(*s, condition=True, pad=False):
    s = ' '.join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        msg = msg if not pad else f'\n{msg}\n'
        print(msg, flush=True)


    return msg



"""
    Evaluate MS MARCO Passages ranking.
"""

import os
import math
import tqdm
import ujson
import random

from argparse import ArgumentParser
from collections import defaultdict
#from colbert.utils.utils import print_message, file_tqdm


def main(args):
    qid2positives = defaultdict(list)
    qid2ranking = defaultdict(list)
    qid2mrr = {}
    qid2recall = {depth: {} for depth in [100, 1000]}

    with open(args.qrels) as f:
        print_message(f"#> Loading QRELs from {args.qrels} ..")
        for line in file_tqdm(f):
            qid, _, pid, label = map(int, line.strip().split())
            assert label == 1

            qid2positives[qid].append(pid)

    with open(args.ranking) as f:
        print(f"#> Loading ranked lists from {args.ranking} ..")
        for line in file_tqdm(f):
            qid, pid, rank, *score = line.strip().split('\t')
            qid, pid, rank = int(qid), int(pid), int(rank)

            if len(score) > 0:
                assert len(score) == 1
                score = float(score[0])
            else:
                score = None

            qid2ranking[qid].append((rank, pid, score))

    if not set.issubset(set(qid2ranking.keys()), set(qid2positives.keys())):
        print("You provided more queries than necessary.. Make sure you did not provided the wrong test set")

    num_judged_queries = len(qid2positives)
    num_ranked_queries = len(qid2ranking)

    if num_judged_queries != num_ranked_queries:
        print()
        print("#> [WARNING] num_judged_queries != num_ranked_queries")
        print(f"#> {num_judged_queries} != {num_ranked_queries}")
        print()

    print(f"#> Computing MRR@10 for {num_judged_queries} queries.")

    for qid in qid2positives:
        ranking = qid2ranking[qid]
        positives = qid2positives[qid]

        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed

            if pid in positives:
                if rank <= 10:
                    qid2mrr[qid] = 1.0 / rank
                break

        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed

            if pid in positives:
                for depth in qid2recall:
                    if rank <= depth:
                        qid2recall[depth][qid] = qid2recall[depth].get(qid, 0) + 1.0 / len(positives)  # salva i dizionari qid2recall e qid2mrr

    assert len(qid2mrr) <= num_ranked_queries, (len(qid2mrr), num_ranked_queries)


    # with open("qid2mrr.json", "w") as mrrfile:
    #     json.dump(qid2mrr, mrrfile)

    # with open("qid2recall.json", "w") as recallfile:
    #     json.dump(qid2recall, recallfile)

    print()
    mrr_10_sum = sum(qid2mrr.values())
    print(f"#> MRR@10 = {mrr_10_sum / num_judged_queries}")
    print("#> MRR@10 (only for ranked queries) = {:.3f}".format(mrr_10_sum / num_ranked_queries))
    

    for depth in qid2recall:
        assert len(qid2recall[depth]) <= num_ranked_queries, (len(qid2recall[depth]), num_ranked_queries)

        print()
        metric_sum = sum(qid2recall[depth].values())
        print(f"#> Recall@{depth} = {metric_sum / num_judged_queries}")
        print("#> Recall@{} (only for ranked queries) = {:.3f}".format(depth,metric_sum / num_ranked_queries ))


        print()
    print(file=sys.stderr) # new line
    if args.annotate:
        print_message(f"#> Writing annotations to {args.output} ..")

        with open(args.output, 'w') as f:
            for qid in qid2positives:
                ranking = qid2ranking[qid]
                positives = qid2positives[qid]

                for rank, (_, pid, score) in enumerate(ranking):
                    rank = rank + 1  # 1-indexed
                    label = int(pid in positives)

                    line = [qid, pid, rank, score, label]
                    line = [x for x in line if x is not None]
                    line = '\t'.join(map(str, line)) + '\n'
                    f.write(line)


if __name__ == "__main__":
    parser = ArgumentParser(description="msmarco_passages.")

    # Input Arguments.
    parser.add_argument('--qrels', dest='qrels', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking', required=True, type=str)
    parser.add_argument('--annotate', dest='annotate', default=False, action='store_true')

    args = parser.parse_args()

    if args.annotate:
        args.output = f'{args.ranking}.annotated'
        assert not os.path.exists(args.output), args.output

    main(args)
