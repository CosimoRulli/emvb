#include <iostream>
#include <vector>
#include <set>
#include <chrono>
#include <cfloat>
#include <string>
#include <tuple>
#include <queue>
#include <fstream>
#include <cstdint>
#include <omp.h>
#include <cnpy.h>

#include "include/parser.hpp"

#include "DocumentScorer.hpp"

using namespace std;

void configure(cmd_line_parser::parser &parser)
{
    parser.add("k", "Number of nearest neighbours.", "-k", false);
    parser.add("nprobe", "Number of cell to look during index search.", "-nprobe", false);
    parser.add("index_dir_path", "Path to the decomposed index", "-index-dir-path", false);
    parser.add("thresh", "Threshold", "-thresh", false);
    parser.add("thresh_query", "Threshold", "-thresh-query", false);

    parser.add("out_second_stage", "Number of candidate documents selected with bitvectors", "-out-second-stage", false);
    parser.add("n_doc_to_score", "Number of document to score", "-n-doc-to-score", false);
    parser.add("queries_id_file", "Path to queries_id file", "-queries-id-file", false); // todo remove in the future questo troiaio (id only tsv)
    parser.add("alldoclens_path", "Path to the doclens file", "-alldoclens-path", false);
    parser.add("outputfile", "Path to the output file used to compute the metrics", "-out-file", false);

}

int main(int argc, char **argv)
{
    omp_set_num_threads(1);

    cmd_line_parser::parser parser(argc, argv);
    configure(parser);
    bool success = parser.parse();
    if (!success)
        return 1;

    int k = parser.get<int>("k");
    float thresh = parser.get<float>("thresh");
    float thresh_query = parser.get<float>("thresh_query");



    size_t n_doc_to_score = parser.get<size_t>("n_doc_to_score");
    size_t nprobe = parser.get<size_t>("nprobe");
    size_t out_second_stage = parser.get<size_t>("out_second_stage");
    string queries_id_file = parser.get<string>("queries_id_file");
    string index_dir_path = parser.get<string>("index_dir_path");
    string alldoclens_path = parser.get<string>("alldoclens_path");
    string outputfile = parser.get<string>("outputfile");

    string queries_path = index_dir_path + "/query_embeddings.npy";

    cnpy::NpyArray queriesArray = cnpy::npy_load(queries_path);

    size_t n_queries = queriesArray.shape[0];
    size_t vec_per_query = queriesArray.shape[1];
    size_t len = queriesArray.shape[2];

    cout << "Dimension: " << len << "\n"
         << "Number of queries: " << n_queries << "\n"
         << "Vector per query " << vec_per_query << "\n";
    uint16_t values_per_query = vec_per_query * len;
    valType *loaded_query_data = queriesArray.data<valType>();

    // load qid mapping file
    auto qid_map = load_qids(queries_id_file);

    cout << "queries id loaded\n";

    // load documents
    DocumentScorer document_scorer(alldoclens_path, index_dir_path, vec_per_query);



    ofstream out_file; // file with final output
    out_file.open(outputfile);

    uint64_t total_time = 0;
    // uint64_t tot_time_score = 0;
    // uint64_t time_centroids_selection = 0;
    // uint64_t time_second_stage = 0;

    // uint64_t time_document_filtering = 0;

    cout << "SEARCH STARTED\n";
    for (size_t query_id = 0; query_id < n_queries; query_id++)
    {
        auto start = chrono::high_resolution_clock::now();
        globalIdxType q_start = query_id * values_per_query;

        // PHASE 1: candidate documents retrieval
        auto candidate_docs = document_scorer.find_candidate_docs(loaded_query_data, q_start, nprobe, thresh);


        // PHASE 2: candidate document filtering
        auto selected_docs = document_scorer.compute_hit_frequency(candidate_docs, thresh, n_doc_to_score);

        //  PHASE 3: second stage filtering
        auto selected_docs_2nd = document_scorer.second_stage_filtering(loaded_query_data, q_start, selected_docs, out_second_stage);

        // PHASE 4: document scoring
        auto query_res = document_scorer.compute_topk_documents_selected(loaded_query_data, q_start, selected_docs_2nd, k, thresh_query);

        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
        total_time += elapsed;
        // write top k file

        for (int i = 0; i < k; i++)
        {
            out_file << qid_map[query_id] << "\t" << get<0>(query_res[i]) << "\t" << i + 1 << "\t" << get<1>(query_res[i]) << endl;
        }
        out_file.flush();
    }

    out_file.flush();
    out_file.close();
    cout<< "Average Elapsed Time per query "<< total_time / n_queries << "\n";

    return 0;
}
