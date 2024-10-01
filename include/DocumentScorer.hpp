#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <tuple>
#include "utils.cpp"
#include <random>
#include <cstring>
#include "ProductQuantizerX.hpp"
#include <unordered_set>
#include "mkl.h"
#include <immintrin.h>
#include <filesystem>
#include "Heap.hpp"
using namespace std;
using namespace cnpy;

const uint32_t BUFFER_SIZE = 50000;

// Select the correct function at compile time
#if defined(__AVX512F__)  // AVX-512 is supported
    #define filter_if_optimal filter_if_avx512
#else  // Use the scalar version if no AVX-512 support
    #define filter_if_optimal filter_if
#endif


// Select the correct function at compile time
#if defined(__AVX512F__)  // AVX-512 is supported
    #define compute_score_by_column_reduction_optimal compute_score_by_column_reduction
#else  // Use the scalar version if no AVX-512 support
    #define compute_score_by_column_reduction_optimal compute_score_by_column_reduction_scalar
#endif


#if defined(__AVX512F__)  // AVX-512 is supported
    #define filter_centroids_in_scoring_optimal filter_centroids_in_scoring
#else  // Use the scalar version if no AVX-512 support
    #define filter_centroids_in_scoring_optimal filter_centroids_in_scoring_scalar
#endif



class DocumentScorer
{
private:
    NpyArray doclensArray, centroidsArray, centroidsAssignmentArray, pqCodesArray, pqCentroidsArray;
    vector<vector<size_t>> centroids_to_pid;
    uint8_t *pq_codes;
    float *centroids;
    size_t *centroids_assignments;

    numDocsType n_docs;
    size_t K; // length of document and query vectors is the same
    numVectorsType tot_embedding;
    size_t n_centroids;
    vector<numDocsType> emb2pid; // store the doc id of each embedding
    int M;
    float alpha, beta;
    vector<float> scores, maxs;
    vector<float> centroids_scores;
    vector<float> centroids_scores_transposed;

    vector<uint32_t> bitvectors;
    vector<uint64_t> bitvectors_centroids;
    int *all_doclens;
    vector<globalIdxType> doc_offsets; // store the starting position of each document
    size_t topt;
    size_t *start_sorted;

    int *buffer_centroids;
    map<numDocsType, vector<float>> map_doc_centroid_scores;

    int *GLOBAL_INDEXES;

public:
    ProductQuantizerX pq; 
    size_t globalCounter = 0;

    DocumentScorer(const string doclens_path, const string decomposed_index_path, const size_t max_query_terms)
    {
        start_sorted = new size_t[BUFFER_SIZE];
        M = max_query_terms;
        alpha = 1.;
        beta = 0.;
        maxs.resize(M);

        string codes_path = decomposed_index_path + "/residuals.npy";
        pqCodesArray = cnpy::npy_load(codes_path);
        pq_codes = pqCodesArray.data<uint8_t>();

        string centroids_path = decomposed_index_path + "/centroids.npy";
        centroidsArray = cnpy::npy_load(centroids_path);
        centroids = centroidsArray.data<valType>();
        n_centroids = centroidsArray.shape[0];
        cout << "Number of Centroids:  " << n_centroids << "\n";

        string centroids_assignment_path = decomposed_index_path + "/index_assignment.npy";
        centroidsAssignmentArray = cnpy::npy_load(centroids_assignment_path);

        centroids_assignments = centroidsAssignmentArray.data<size_t>();
        centroids_scores.resize(M * n_centroids);
        centroids_scores_transposed.resize(M * n_centroids);

        doclensArray = cnpy::npy_load(doclens_path);
        all_doclens = doclensArray.data<int>();
        n_docs = doclensArray.shape[0]; // number of documents in the entire collection
        K = centroidsArray.shape[1];
        tot_embedding = centroidsAssignmentArray.shape[0];
        cout << "Total Number of Embeddings: " << tot_embedding << "\n";
        // build emb2pid vector
        emb2pid.resize(tot_embedding);
        numVectorsType offset = 0;
        for (numDocsType i = 0; i < n_docs; i++)
        {
            int len = all_doclens[i];
            for (numVectorsType j = offset; j < offset + len; j++)
            {
                emb2pid[j] = i;
            }
            offset = offset + len;
        }
        emb2pid.shrink_to_fit();
        // build doc_offsets vector
        doc_offsets.resize(n_docs);
        doc_offsets[0] = 0;
        for (numDocsType i = 1; i < n_docs; i++)
        {
            doc_offsets[i] = doc_offsets[i - 1] + all_doclens[i - 1]; // Here we do not multiply with K
        }
        doc_offsets.shrink_to_fit();
        std::cout << "Number of documents: " << n_docs << endl;
        size_t nbits = 8; // todo this should not be hardcoded
        size_t ntotal = pqCodesArray.shape[0];
        string pq_centroids_path = decomposed_index_path + "/pq_centroids.npy";
        pqCentroidsArray = cnpy::npy_load(pq_centroids_path);

        vector<float> pqcentroids{pqCentroidsArray.data<float>(), pqCentroidsArray.data<float>() + pqCentroidsArray.shape[0]};
        pq = ProductQuantizerX(K, pqCodesArray.shape[1], nbits, pq_codes, pqcentroids);

        string centroids_to_pids = decomposed_index_path + "/centroids_to_pids.txt";
        load_centroid_to_pids(centroids_to_pids);

        init_bitvectors_32(this->n_centroids, this->bitvectors);
        size_t bitvectors_centroids_size = (n_docs / 64) + 1;

        bitvectors_centroids.resize(bitvectors_centroids_size);
        fill(bitvectors_centroids.begin(), bitvectors_centroids.end(), (size_t)0);

        buffer_centroids = new int[400];

        GLOBAL_INDEXES = new int[16];
        for (int ix = 0; ix < 16; ix++)
        {
            GLOBAL_INDEXES[ix] = ix;
        }
    }

    ~DocumentScorer()
    {
        delete[] start_sorted;
    }

    /// Starting functions for phase 1.

    vector<float> compute_query_centroids_distances(const float *queries_data, const globalIdxType q_start)
    {
        vector<float> current_scores(M * n_centroids);
        int N = n_centroids;
        // dnnl_sgemm('N', 'T', M, N, K, alpha, queries_data + q_start, K, centroids, K, beta, current_scores.data(), N);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, queries_data + q_start, K, centroids, K, beta, current_scores.data(), N);

        return current_scores;
    }


    size_t *filter_if(const float th, const size_t i)
    {
        size_t *sorted_indexes = start_sorted;
        size_t idx = 0;
        for (size_t j = 0; j < n_centroids; j++)
        {
            // Access the current value
            float current_value = centroids_scores[i * n_centroids + j];
            
            // Compare it to the threshold
            if (current_value > th)
            {
                sorted_indexes[idx] = j;
                idx++;
            }
        }
        return sorted_indexes + idx;
    }


    size_t *filter_if_avx512(const float th, const size_t i)
    {

        size_t *sorted_indexes = start_sorted;
        __m512 broad_th = _mm512_set1_ps(th);
        __m512 current_values;
        size_t idx = 0;
        for (size_t j = 0; j < n_centroids; j += 16)
        {
            // load unaligned at the moment.
            current_values = _mm512_loadu_ps((const void *)&(centroids_scores[i * n_centroids + j]));
            __mmask16 mask = _mm512_cmp_ps_mask(current_values, broad_th, _CMP_GT_OS);
            if (mask != (uint16_t)0)
            {

                for (size_t bit = 0; bit < 16; bit++)
                {
                    if ((mask >> bit) & (uint16_t)1)
                    {
                        sorted_indexes[idx] = j + bit;
                        idx++;
                    }
                }
            }
        }
        return sorted_indexes + idx;
    }

  

    vector<numDocsType>
    find_candidate_docs(const float *queries_data, const globalIdxType q_start, const size_t nprobe, const float th)
    {
        centroids_scores = compute_query_centroids_distances(queries_data, q_start);

        vector<size_t> closest_centroids_ids;
        closest_centroids_ids.reserve(nprobe * M);

        for (int i = 0; i < M; i++)
        {
            size_t current_n_probe = nprobe;

            // auto sorted_indexes = filter_branchless_avx512(th, i);
            auto sorted_indexes = filter_if_optimal(th, i);
            // auto sorted_indexes = filter_branchless(th, i);

            assign_bitvector_32(start_sorted, sorted_indexes - start_sorted, i, this->bitvectors);

            if (sorted_indexes - start_sorted >= nprobe)
            {
                vector<float> candidate_centroids_scores;
                for (int j = 0; j < sorted_indexes - start_sorted; j++)

                {
                    candidate_centroids_scores.push_back(centroids_scores[i * n_centroids + start_sorted[j]]);
                }

                while (current_n_probe > 0)
                {
                    size_t temp_argmax = std::distance(candidate_centroids_scores.begin(), max_element(candidate_centroids_scores.begin(), candidate_centroids_scores.end()));
                    size_t argmax = start_sorted[temp_argmax];
                    closest_centroids_ids.push_back(argmax);

                    // in the first iteration, save the centroid assigments

                    candidate_centroids_scores[temp_argmax] = -1;
                    current_n_probe--;
                }
            }
            else
            {
                // do not perform multi-probing if no centroid candidate is found with the threshold.
                auto start = centroids_scores.begin() + i * n_centroids;
                auto end = centroids_scores.begin() + (i + 1) * n_centroids;
                size_t argmax = std::distance(start, std::max_element(start, end));
                closest_centroids_ids.push_back(argmax);
            }
        }

        set<size_t> unique_centroids;
        for (auto &centr : closest_centroids_ids)
        {
            unique_centroids.insert(centr);
        }

        vector<numDocsType> candidate_docs;

        for (auto &uq : unique_centroids)
        {
            auto current_list = centroids_to_pid[uq];
            for (auto &doc_id : current_list)
            {
                if (!check_bit_64(doc_id, this->bitvectors_centroids))
                {
                    candidate_docs.push_back(doc_id);
                    set_bit_64(doc_id, this->bitvectors_centroids);
                }
            }
        }
        // for (auto &uq : unique_centroids)
        // {
        //     auto current_list = centroids_to_pid[uq];
        //     for (auto &doc_id : current_list)
        //     {
        //         set_bit_64(doc_id);
        //     }
        // }

        fill(bitvectors_centroids.begin(), bitvectors_centroids.end(), (size_t)0);

        return candidate_docs;
    }

    /// Ending functions for phase 1.

    /// Starting functions for phase 2.

    vector<numDocsType> compute_hit_frequency(vector<numDocsType> &candidate_documents, const float th, const size_t k_centroids)
    {
        if (k_centroids >= candidate_documents.size())
        {
            return candidate_documents;
        }

        vector<size_t> doc_scores(candidate_documents.size());
        vector<size_t> unsorted_indexes(candidate_documents.size());

        auto heap = HeapIntegers(k_centroids);

        // for (auto &doc_id : candidate_documents)
        // TODO: by adding push_with_id to the heap, we could iterate over doc_id instead that ond
        for (size_t doc_idx = 0; doc_idx < candidate_documents.size(); doc_idx++)
        {
            auto doc_id = candidate_documents[doc_idx];
            auto doclen = all_doclens[doc_id];
            auto doc_offset = doc_offsets[doc_id];

            vector<size_t> centroid_ids(doclen);
            uint32_t mask = 0;
            for (int i = 0; i < doclen; i++)
            {
                centroid_ids[i] = centroids_assignments[doc_offset + i];
                auto cid = centroids_assignments[doc_offset + i];
                mask |= bitvectors[cid];
            }

            size_t score = popcount(mask);

            heap.push(score);
        }
        auto sorted_indexes = heap.arg_topk();
        vector<numDocsType> selected_docs(k_centroids);

        for (size_t i = 0; i < sorted_indexes.size(); i++)
        {
            auto idx = sorted_indexes[i];
            selected_docs[i] = candidate_documents[idx];
        }

        reset_bitvectors_32(this->bitvectors);
        return selected_docs;
    }

    /// Ending functions for phase 2.

    /// Starting functions for phase 3.
    inline vector<float> compute_ip_with_centroids(const float *queries, const numDocsType doc_id)
    {

        auto doclen = all_doclens[doc_id];
        auto doc_offset = doc_offsets[doc_id];
        vector<size_t> centroid_ids(doclen);
        for (int i = 0; i < doclen; i++)
        {
            centroid_ids[i] = centroids_assignments[i + doc_offset];
        }

        vector<float> centroid_distances(M * doclen);
        for (int j = 0; j < doclen; j++)
        {
            size_t centroid_id = centroid_ids[j];
            for (int i = 0; i < M; i++)
            {
                // writing transposed
                centroid_distances[j * M + i] = centroids_scores_transposed[centroid_id * M + i];
            }
        }

        return centroid_distances;
    }

    void transpose_centroids_scores_mkl_oplace()
    {
        float alpha = 1.0;
        mkl_somatcopy('R' /* row-major ordering */,
                      'T' /* A will be transposed */,
                      M /* rows */,
                      n_centroids /* cols */,
                      alpha /* scales the input matrix */,
                      centroids_scores.data() /* source matrix */,
                      n_centroids /* src_stride */,
                      centroids_scores_transposed.data(),
                      M /* dst_stride */);
    }

    inline float compute_score_by_column_reduction(const vector<float> &centroid_distances, const size_t doclen, const size_t M)
    {

        __m512 maxs0 = _mm512_loadu_ps((const void *)&centroid_distances[0]);
        __m512 maxs1 = _mm512_loadu_ps((const void *)&centroid_distances[16]);

        for (size_t i = 1; i < doclen; i++)
        {
            __m512 current0 = _mm512_loadu_ps((const void *)&centroid_distances[i * M]);
            __m512 current1 = _mm512_loadu_ps((const void *)&centroid_distances[i * M + 16]);

            __mmask16 m0 = _mm512_cmp_ps_mask(current0, maxs0, _CMP_GT_OS);
            __mmask16 m1 = _mm512_cmp_ps_mask(current1, maxs1, _CMP_GT_OS);

            maxs0 = _mm512_mask_blend_ps(m0, maxs0, current0);
            maxs1 = _mm512_mask_blend_ps(m1, maxs1, current1);
        }
        __m512 half_sum = _mm512_add_ps(maxs0, maxs1);


        return _mm512_reduce_add_ps(half_sum);
    }


    inline float compute_score_by_column_reduction_scalar(const std::vector<float> &centroid_distances, const size_t doclen, const size_t M)
    {
        // Initialize maxs0 and maxs1 using the first two chunks of 16 values each.
        std::vector<float> maxs0(16), maxs1(16);
        for (size_t j = 0; j < 16; j++) {
            maxs0[j] = centroid_distances[j];
            maxs1[j] = centroid_distances[16 + j];
        }

        // Process remaining rows
        for (size_t i = 1; i < doclen; i++)
        {
            for (size_t j = 0; j < 16; j++) {
                // Compare each value and store the max in maxs0 and maxs1
                maxs0[j] = std::max(maxs0[j], centroid_distances[i * M + j]);
                maxs1[j] = std::max(maxs1[j], centroid_distances[i * M + 16 + j]);
            }
        }

        // Add corresponding elements in maxs0 and maxs1
        float sum = 0.0f;
        for (size_t j = 0; j < 16; j++) {
            sum += maxs0[j] + maxs1[j];
        }

        return sum;
    }

    vector<numDocsType> second_stage_filtering(const float *queries_data, const globalIdxType q_start, const vector<numDocsType> &doc_ids, const size_t n_documents)
    {
        transpose_centroids_scores_mkl_oplace();
        priority_queue<tuple<numDocsType, valType>, vector<tuple<numDocsType, valType>>, Compare> min_heap;
        // auto heap = HeapIntegers(n_documents);

        for (size_t doc_idx = 0; doc_idx < doc_ids.size(); doc_idx++)
        {

            auto doc_id = doc_ids[doc_idx];
            auto doclen = all_doclens[doc_id];
            //auto doc_offset = doc_offsets[doc_id];

            auto centroid_distances = compute_ip_with_centroids(queries_data + q_start, doc_id);

            auto score = compute_score_by_column_reduction_optimal(centroid_distances, doclen, M);
            // TODO: here, replace with heap (Maybe)

            if (min_heap.size() < n_documents)
            {
                // here, save the centroid_scores
                min_heap.push(make_tuple(doc_id, score));
                map_doc_centroid_scores[doc_id] = centroid_distances;
            }
            else
            {
                tuple<numDocsType, valType> t = min_heap.top();
                if (score > get<1>(t))
                {
                    // here, save the centroid_scores
                    map_doc_centroid_scores[doc_id] = centroid_distances;

                    tuple<numDocsType, valType> t = min_heap.top();
                    map_doc_centroid_scores.erase(get<0>(t));
                    min_heap.pop();
                    min_heap.push(make_tuple(doc_id, score));
                }
            }
        }

        vector<numDocsType> selected_docs;
        selected_docs.reserve(n_documents);

        while (min_heap.size())
        {
            selected_docs.push_back(get<0>(min_heap.top()));
            min_heap.pop();
        }
        return selected_docs;
    }

    /// Ending functions for phase 3.

    /// Starting functions for phase 4.

    priority_queue<tuple<numDocsType, valType>, vector<tuple<numDocsType, valType>>, Compare> compute_topk_documents(const float *queries_data, const globalIdxType q_start, const vector<numDocsType> &doc_ids, const size_t k)
    {
        priority_queue<tuple<numDocsType, valType>, vector<tuple<numDocsType, valType>>, Compare> min_heap;
        // auto heap = HeapFloats(k);

        pq.precompute_distance_table(queries_data + q_start, M);

        for (numDocsType doc_id : doc_ids)
        {
            auto doclen = all_doclens[doc_id];
            auto doc_offset = doc_offsets[doc_id];

            auto centroid_distances = map_doc_centroid_scores[doc_id];
            auto pq_distances = pq.compute_distances_with_offset(doc_offset, doclen, M);

            vector<float> distances(M * doclen);
            vector<float> maxs(M);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < doclen; j++)
                {
                    // distances[i * doclen + j] = centroid_distances[i * doclen + j] + pq_distances[i * doclen + j];
                    distances[i * doclen + j] = centroid_distances[j * M + i] + pq_distances[i * doclen + j];
                }
            }

            for (int i = 0; i < M; i++)
            {
                maxs[i] = *std::max_element(&distances[i * doclen], &distances[(i + 1) * doclen]);
            }

            float score = accumulate(maxs.begin(), maxs.end(), 0.0f);
            // TODO: replace with new heap
            // heap.add(score);

            if (min_heap.size() < k)
                min_heap.push(make_tuple(doc_id, score));
            else
            {
                tuple<numDocsType, valType> t = min_heap.top();
                if (score > get<1>(t))
                {
                    min_heap.pop();
                    min_heap.push(make_tuple(doc_id, score));
                }
            }
        }

        map_doc_centroid_scores.clear();

        return min_heap;
    }

    vector<tuple<size_t, float>> compute_topk_documents_2(const float *queries_data, const globalIdxType q_start, const vector<numDocsType> &doc_ids, const size_t k)
    {
        auto heap = HeapFloats(k);
        pq.precompute_distance_table(queries_data + q_start, M);

        for (numDocsType doc_id : doc_ids)
        {

            auto doclen = all_doclens[doc_id];
            auto doc_offset = doc_offsets[doc_id];
            this->globalCounter += doclen;
            auto centroid_distances = map_doc_centroid_scores[doc_id];
            auto pq_distances = pq.compute_distances_with_offset(doc_offset, doclen, M);

            vector<float> distances(M * doclen);
            vector<float> maxs(M);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < doclen; j++)
                {
                    // distances[i * doclen + j] = centroid_distances[i * doclen + j] + pq_distances[i * doclen + j];
                    distances[i * doclen + j] = centroid_distances[j * M + i] + pq_distances[i * doclen + j];
                }
            }

            for (int i = 0; i < M; i++)
            {
                maxs[i] = *std::max_element(&distances[i * doclen], &distances[(i + 1) * doclen]);
            }

            float score = accumulate(maxs.begin(), maxs.end(), 0.0f);

            heap.push(score);
        }

        auto result = heap.sorted_topk();
        for (auto &tuple : result)
        {
            size_t id = get<0>(tuple); // Get the ID from the tuple
            // Call the foo function to modify the ID
            size_t doc_id = doc_ids[id]; // Assuming foo takes an int as an argument
            // Update the first element of the tuple with the modified ID
            get<0>(tuple) = doc_id;
        }
        map_doc_centroid_scores.clear();

        return result;
    }

    inline int *filter_centroids_in_scoring(const float th, const float *current_centroid_scores, const size_t doclen)
    {
        __m512i ids = _mm512_loadu_epi32((const void *)GLOBAL_INDEXES);
        const __m512i SHIFT = _mm512_set1_epi32(16);
        __m512 broad_th = _mm512_set1_ps(th);
        __m512 current_values;
        int *current_buffer = this->buffer_centroids;

        size_t avx_cycle_lenth = (doclen / 16) * 16;

        for (size_t j = 0; j < avx_cycle_lenth; j += 16)
        {
            // load unaligned at the moment.
            current_values = _mm512_loadu_ps((const void *)&current_centroid_scores[j]);
            __mmask16 mask = _mm512_cmp_ps_mask(current_values, broad_th, _CMP_GT_OS);
            _mm512_mask_compressstoreu_epi32((void *)current_buffer, mask, ids);
            auto added_len = popcount(mask);
            current_buffer += added_len;
            ids = _mm512_add_epi32(ids, SHIFT);
        }

        for (size_t j = avx_cycle_lenth; j < doclen; j++)
        {
            *current_buffer = j;
            current_buffer += (size_t)(current_centroid_scores[j] > th);
        }

        return current_buffer;
    }

    inline int *filter_centroids_in_scoring_scalar(const float th, const float *current_centroid_scores, const size_t doclen)
    {
        int *current_buffer = this->buffer_centroids;

        for (size_t j = 0; j < doclen; j++)
        {
            if (current_centroid_scores[j] > th)
            {
                *current_buffer = GLOBAL_INDEXES[j];
                current_buffer++;
            }
        }

        return current_buffer;
    }

    vector<tuple<size_t, float>> compute_topk_documents_selected(const float *queries_data, const globalIdxType q_start, const vector<numDocsType> &doc_ids, const size_t k, const float th)
    {
        auto heap = HeapFloats(k);
        pq.precompute_distance_table(queries_data + q_start, M);

        for (numDocsType doc_id : doc_ids)
        {

            auto doclen = all_doclens[doc_id];
            auto doc_offset = doc_offsets[doc_id];
            vector<float> buffer_for_distances(doclen, 0.0);

            auto centroid_distances = map_doc_centroid_scores[doc_id];

            // auto pq_distances = pq.compute_distances_with_offset(doc_offset, doclen, M);

            vector<float> distances(M * doclen);
            vector<float> maxs(M);

            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < doclen; j++)
                {
                    distances[i * doclen + j] = centroid_distances[j * M + i];
                }

                auto current_indexes = filter_centroids_in_scoring_optimal(th, &distances[i * doclen], doclen);

                if (current_indexes == this->buffer_centroids)
                {
                    this->globalCounter+= doclen;
                    pq.compute_distances_one_qt(doc_offset, doclen, i, buffer_for_distances);

                    for (int j = 0; j < doclen; j++)
                    {
                        distances[i * doclen + j] += buffer_for_distances[j];
                        // cout<< buffer_for_distances[j] << " "<< pq_distances[i * doclen + j] << "\n";
                    }
                    // for (size_t j = 0; j < doclen; j++)
                    // {
                    //     distances[i * doclen + j] += pq_distances[i * doclen + j];
                    // }
                    maxs[i] = *std::max_element(&distances[i * doclen], &distances[(i + 1) * doclen]);
                }
                else
                {
                    this->globalCounter+= (current_indexes - this->buffer_centroids);
                    
                    // TODO: Here, compute only necessary terms
                    //  Probably, we should first compute all the dot product with the codes and then add them to distances
                    //  to avoid the distance table to be pop out of the cache
                    for (int idx = 0; idx < (int)(current_indexes - this->buffer_centroids); idx++)
                    {
                        auto j = this->buffer_centroids[idx];
                        // distances[i * doclen + j] += pq_distances[i * doclen + j];
                        distances[i * doclen + j] += pq.compute_distances_one_qt_one_doc(doc_offset, doclen, i, j);
                    }

                    // TODO: maybe could be improved by extracting the max only in current_indexes
                    maxs[i] = *std::max_element(&distances[i * doclen], &distances[(i + 1) * doclen]);
                }
            }

            float score = accumulate(maxs.begin(), maxs.end(), 0.0f);
            heap.push(score);

        }
        auto result = heap.sorted_topk();
        for (auto &tuple : result)
        {
            size_t id = get<0>(tuple); // Get the ID from the tuple
            // Call the foo function to modify the ID
            size_t doc_id = doc_ids[id]; // Assuming foo takes an int as an argument
            // Update the first element of the tuple with the modified ID
            get<0>(tuple) = doc_id;
        }
        map_doc_centroid_scores.clear();

        return result;

    }

    /// Ending functions for phase 3.

    void load_centroid_to_pids(const string path)
    {
        ifstream file(path);
        string line_centroid_to_pid;

        while (std::getline(file, line_centroid_to_pid))
        {
            vector<size_t> current_line;

            std::stringstream ss(line_centroid_to_pid);
            std::istream_iterator<std::string> begin(ss);
            std::istream_iterator<std::string> end;
            std::vector<std::string> tokens(begin, end);
            for (auto &s : tokens)
            {
                current_line.push_back(stoi(s));
            }
            centroids_to_pid.push_back(current_line);
        }
    }
};