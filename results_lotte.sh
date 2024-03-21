
qrels_file="aux_data/lotte/queries_ids_lotte.tsv"
index_path="/data2/cosimorulli/EMVB/indexes/260k_m32_LOTTE_OPQ"


# k = 10
./build/perf_emvb -k 10 -nprobe 4 -thresh 0.4 -out-second-stage 512 -thresh-query 0.5 -n-doc-to-score 4000 -queries-id-file aux_data/lotte/queries_id_lotte.tsv  -alldoclens-path aux_data/lotte/doclens_lotte.npy -index-dir-path $index_path -out-file results_10_lotte.tsv 

python evaluate_lotte_rankings.py -gt aux_data/lotte_pooled_qas.search.jsonl -r results_10_lotte.tsv 

# k = 100
./build/perf_emvb -k 100 -nprobe 4 -thresh 0.4 -out-second-stage 1024 -thresh-query 0.5 -n-doc-to-score 4000 -queries-id-file aux_data/lotte/queries_id_lotte.tsv  -alldoclens-path aux_data/lotte/doclens_lotte.npy -index-dir-path $index_path -out-file results_100_lotte.tsv 

python evaluate_lotte_rankings.py -gt aux_data/lotte_pooled_qas.search.jsonl -r results_100_lotte.tsv 

# k = 1000
./build/perf_emvb -k 1000 -nprobe 4 -thresh 0.4 -out-second-stage 2048 -thresh-query 0.5 -n-doc-to-score 4000 -queries-id-file aux_data/lotte/queries_id_lotte.tsv  -alldoclens-path aux_data/lotte/doclens_lotte.npy -index-dir-path $index_path -out-file results_1000_lotte.tsv 

python evaluate_lotte_rankings.py -gt aux_data/lotte_pooled_qas.search.jsonl -r results_1000_lotte.tsv 
