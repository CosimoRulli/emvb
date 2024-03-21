#!/bin/bash

# Define path to QRELs file
qrels_file="aux_data/msmarco/qrels.dev.tsv"

# Reproduce Table 1 with m = 16
index_path_1=""

# k = 10
./build/perf_emvb -k 10 -nprobe 2 -thresh 0.4 -out-second-stage 128 -thresh-query 0.4 -n-doc-to-score 500 -queries-id-file aux_data/msmarco/queries_dev_small_idonly.tsv  -alldoclens-path aux_data/msmarco/doclens_msmarco.npy -index-dir-path $index_path_1 -out-file results_10.tsv 
python compute_mrr.py --qrels $qrels_file --ranking results_10.tsv

# k = 100
./build/perf_emvb -k 100 -nprobe 2 -thresh 0.4 -out-second-stage 512 -thresh-query 0.4 -n-doc-to-score 1000 -queries-id-file aux_data/msmarco/queries_dev_small_idonly.tsv  -alldoclens-path aux_data/msmarco/doclens_msmarco.npy -index-dir-path $index_path_1 -out-file results_100.tsv 
python compute_mrr.py --qrels $qrels_file --ranking results_100.tsv

# k = 1000
./build/perf_emvb -k 1000 -nprobe 4 -thresh 0.4 -out-second-stage 1024 -thresh-query 0.5 -n-doc-to-score 4000 -queries-id-file aux_data/msmarco/queries_dev_small_idonly.tsv  -alldoclens-path aux_data/msmarco/doclens_msmarco.npy -index-dir-path $index_path_1 -out-file results_1000.tsv 
python compute_mrr.py --qrels $qrels_file --ranking results_1000.tsv

# Reproduce Table 2 with m = 32
index_path_2=""

# k = 10
./build/perf_emvb -k 10 -nprobe 1 -thresh 0.4 -out-second-stage 128 -thresh-query 0.5 -n-doc-to-score 500 -queries-id-file aux_data/msmarco/queries_dev_small_idonly.tsv  -alldoclens-path aux_data/msmarco/doclens_msmarco.npy -index-dir-path $index_path_2 -out-file results_10.tsv 
python compute_mrr.py --qrels $qrels_file --ranking results_10.tsv

# k = 100
./build/perf_emvb -k 100 -nprobe 3 -thresh 0.4 -out-second-stage 256 -thresh-query 0.4 -n-doc-to-score 2000 -queries-id-file aux_data/msmarco/queries_dev_small_idonly.tsv  -alldoclens-path aux_data/msmarco/doclens_msmarco.npy -index-dir-path $index_path_2 -out-file results_100.tsv 
python compute_mrr.py --qrels $qrels_file --ranking results_100.tsv

# k = 1000
./build/perf_emvb -k 1000 -nprobe 4 -thresh 0.4 -out-second-stage 1024 -thresh-query 0.4 -n-doc-to-score 5000 -queries-id-file aux_data/msmarco/queries_dev_small_idonly.tsv  -alldoclens-path aux_data/msmarco/doclens_msmarco.npy -index-dir-path $index_path_2 -out-file results_1000.tsv 
python compute_mrr.py --qrels $qrels_file --ranking results_1000.tsv
