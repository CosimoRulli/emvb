#include <vector>
#include <algorithm> // For sorting

using namespace std;

class HeapIntegers
{
private:
    vector<size_t> distances;

    size_t k;
    numDocsType timestamp;

public:
    vector<numDocsType> ids;

    HeapIntegers(size_t k) : k(k), timestamp(0)
    {
        distances.reserve(k);
        ids.reserve(k);
    }

    inline void add(size_t distance, numDocsType id)
    {
        distances.push_back(distance);
        ids.push_back(id);

        size_t i = distances.size() - 1;
        size_t i_father;

        while (i > 0)
        {
            i_father = ((i + 1) >> 1) - 1;
            // if (distance <= distances[i_father]) {
            if (distance >= distances[i_father])
            {
                break;
            }
            distances[i] = distances[i_father];
            ids[i] = ids[i_father];
            i = i_father;
        }
        distances[i] = distance;
        ids[i] = id;
    }

    inline void replace_top(size_t distance, numDocsType id)
    {
        size_t k = distances.size();
        size_t i = 0;
        size_t i1;
        size_t i2;

        while (true)
        {
            i2 = (i + 1) << 1;
            i1 = i2 - 1;
            if (i1 >= k)
            {
                break;
            }
            // if ((i2 == k) || (distances[i1] >= distances[i2])) {
            //     if (distance >= distances[i1]) {
            if ((i2 == k) || (distances[i1] <= distances[i2]))
            {
                if (distance <= distances[i1])
                {
                    break;
                }
                distances[i] = distances[i1];
                ids[i] = ids[i1];
                i = i1;
            }
            else
            {
                // if (distance >= distances[i2]) {
                if (distance <= distances[i2])
                {
                    break;
                }
                distances[i] = distances[i2];
                ids[i] = ids[i2];
                i = i2;
            }
        }
        distances[i] = distance;
        ids[i] = id;
    }

    size_t top() const
    {
        return distances[0];
    }

    inline void push(size_t distance)
    {
        if (timestamp < k)
        {
            add(distance, timestamp);
            timestamp += 1;
            return;
        }

        // if (distance < top()) {
        if (distance > top())
        {
            replace_top(distance, timestamp);
        }
        timestamp += 1;
    }

    void push_with_id(size_t distance, numDocsType id) {
        if (timestamp < k) {
            add(distance, id);
            timestamp += 1;
            return;
        }

        if (distance < top()) {
            replace_top(distance, id);
        }
        timestamp += 1;
    }


    inline void extend(const std::vector<size_t> &distances)
    {
        auto iter = distances.begin();
        size_t id = 0;

        while (this->distances.size() < k)
        {
            if (iter != distances.end())
            {
                add(*iter, timestamp + id);
                ++iter;
            }
            else
            {
                timestamp += distances.size();
                return;
            }
        }

        for (; iter != distances.end(); ++iter, ++id)
        {
            // if (*iter < top()) {
            if (*iter > top())
            {
                replace_top(*iter, timestamp + id);
            }
        }
        timestamp += distances.size();
    }

    inline vector<numDocsType> arg_topk() const
    {
        return ids;
    }

    // vector<std::pair<size_t, numDocsType>> topk() const
    // {
    //     std::vector<std::pair<size_t, numDocsType>> result;
    //     for (size_t i = 0; i < distances.size(); ++i)
    //     {
    //         result.emplace_back(distances[i], ids[i]);
    //     }
    //     return result;
    // }
};

#include <vector>

using namespace std;

class HeapFloats
{
private:
    vector<float> distances;

    size_t k;
    size_t timestamp;

public:
    vector<size_t> ids;

    HeapFloats(size_t k) : k(k), timestamp(0)
    {
        distances.reserve(k);
        ids.reserve(k);
    }

    inline void add(float distance, size_t id)
    {
        distances.push_back(distance);
        ids.push_back(id);

        size_t i = distances.size() - 1;
        size_t i_father;

        while (i > 0)
        {
            i_father = ((i + 1) >> 1) - 1;
            if (distance >= distances[i_father])
            {
                break;
            }
            distances[i] = distances[i_father];
            ids[i] = ids[i_father];
            i = i_father;
        }
        distances[i] = distance;
        ids[i] = id;
    }

    inline void replace_top(float distance, size_t id)
    {
        size_t k = distances.size();
        size_t i = 0;
        size_t i1;
        size_t i2;

        while (true)
        {
            i2 = (i + 1) << 1;
            i1 = i2 - 1;
            if (i1 >= k)
            {
                break;
            }
            if ((i2 == k) || (distances[i1] <= distances[i2]))
            {
                if (distance <= distances[i1])
                {
                    break;
                }
                distances[i] = distances[i1];
                ids[i] = ids[i1];
                i = i1;
            }
            else
            {
                if (distance <= distances[i2])
                {
                    break;
                }
                distances[i] = distances[i2];
                ids[i] = ids[i2];
                i = i2;
            }
        }
        distances[i] = distance;
        ids[i] = id;
    }

    float top() const
    {
        return distances[0];
    }

    inline void push(float distance)
    {
        if (timestamp < k)
        {
            add(distance, timestamp);
            timestamp += 1;
            return;
        }

        if (distance > top())
        {
            replace_top(distance, timestamp);
        }
        timestamp += 1;
    }

    inline void extend(const std::vector<float> &newDistances)
    {
        auto iter = newDistances.begin();
        size_t id = 0;

        while (this->distances.size() < k)
        {
            if (iter != newDistances.end())
            {
                add(*iter, timestamp + id);
                ++iter;
            }
            else
            {
                timestamp += newDistances.size();
                return;
            }
        }

        for (; iter != newDistances.end(); ++iter, ++id)
        {
            if (*iter > top())
            {
                replace_top(*iter, timestamp + id);
            }
        }
        timestamp += newDistances.size();
    }

    void push_with_id(float distance, size_t id) {
        if (timestamp < k) {
            add(distance, id);
            timestamp += 1;
            return;
        }

        if (distance < top()) {
            replace_top(distance, id);
        }
        timestamp += 1;
    }


    inline vector<size_t> arg_topk() const
    {
        return ids;
    }

    vector<std::pair<float, size_t>> topk() const
    {
        std::vector<std::pair<float, size_t>> result;
        for (size_t i = 0; i < distances.size(); ++i)
        {
            result.emplace_back(distances[i], ids[i]);
        }
        return result;
    }

    
    vector<tuple<size_t, float>> sorted_topk() const {
        // Create a vector of tuples to store IDs and distances
        vector<tuple<size_t, float>> sortedTuples;

        // Copy the data from distances and ids vectors into sortedTuples
        for (size_t i = 0; i < distances.size(); ++i) {
            sortedTuples.emplace_back(ids[i], distances[i]);
        }

        // Sort the vector of tuples based on distances in descending order
        std::sort(sortedTuples.begin(), sortedTuples.end(), [](const auto& a, const auto& b) {
            return get<1>(a) > get<1>(b); // Compare in descending order based on the second element (distance)
        });

        // Return the sorted vector of tuples
        return sortedTuples;
    }
};
