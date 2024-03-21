#include <faiss/impl/ProductQuantizer.h>

template <typename CT>
inline void compute_distance(
    const size_t M,
    const CT *codes,
    const size_t ncodes,
    const float *__restrict dis_table,
    const size_t ksub,
    float *__restrict distances)
{

    for (size_t j = 0; j < ncodes; j++)
    {

        float dis = 0;
        const float *dt = dis_table;

        for (size_t m = 0; m < M; m += 4)
        {
            float dism = 0;
            dism = dt[*codes++];
            dt += ksub;
            dism += dt[*codes++];
            dt += ksub;
            dism += dt[*codes++];
            dt += ksub;
            dism += dt[*codes++];
            dt += ksub;
            dis += dism;
        }

        distances[j] = dis;
    }
}



struct ProductQuantizerX : faiss::ProductQuantizer
{
    uint8_t *codes;
    vector<float> distance_matrix;
//    float * precomputed_distance_table;
    size_t skip_counter = 0;
    vector<size_t> query_indexes;
    vector<size_t> neg_query_indexes;

    ProductQuantizerX(size_t d, size_t M, size_t nbits, uint8_t *codes, const vector<float> centroids)
        : faiss::ProductQuantizer(d, M, nbits), codes(codes)
    {
        this->centroids = centroids;
        distance_matrix.assign(32 * 32, 1.);
    }
    ProductQuantizerX() : ProductQuantizerX(0, 1, 0, 0, vector<float>()) {}

    inline vector<float> compute_distances_with_offset(
        const size_t offset,
        const size_t range,
        const size_t nx)
    {
        vector<float> distances(nx * range);
        // float * dism = (float*) std::aligned_alloc(64, sizeof(float)*M);
        // #pragma omp parallel for
        for (size_t i = 0; i < nx; i++)
        {
            // maybe prefetch here?
            compute_distance<uint8_t>(
                M, codes + M * offset, range, this->precomputed_dis_table + i * ksub * M, ksub, distances.data() + i * range);
        }
        return distances;
    }


    inline void compute_distances_one_qt(
        const size_t offset,
        const size_t range,
        const size_t query_term_index,
        vector<float> &distances)
    {
        compute_distance<uint8_t>(
            M, codes + M * offset, range, this->precomputed_dis_table + query_term_index * ksub * M, ksub, distances.data());
    }

    inline float compute_distances_one_qt_one_doc(
        const size_t offset,
        const size_t range,
        const size_t query_term_index,
        const size_t doc_term_index)
    {
        const float *dt = this->precomputed_dis_table + query_term_index * ksub * M;
        uint8_t *current_codes = codes + M * (offset + doc_term_index)  ;

        float dis = 0;
        for (size_t m = 0; m < M; m += 4)
        {
            float dism = 0;
            dism = dt[*current_codes++];
            dt += ksub;
            dism += dt[*current_codes++];
            dt += ksub;
            dism += dt[*current_codes++];
            dt += ksub;
            dism += dt[*current_codes++];
            dt += ksub;
            dis += dism;
        }

        return dis;
    }

};
