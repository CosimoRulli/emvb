# Efficient Multi-Vector Retrieval with Bit Vectors (EMVB)

This repo contains the code and instructions on how to reproduce the results of the ECIR 2024 paper: Franco Maria Nardini, Cosimo Rulli, Rossano Venturini. "Efficient Multi-vector Dense Retrieval with Bit Vectors." European Conference on Information Retrieval. 2024.

### Requirements

As our code heavily relies on AVX512 instructions, to run it you need a CPU with available AVX512 instructions. 

### Installation

- Make sure to have the Math Kernel Library ([MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.5pn8i4)) installed on your machine and set the ```MKLROOT``` env variable to point to the installation directory.

- Clone the repo using the ```--recursive``` flag.
- Run 
  ```bash
  mkdir build && cd build
  cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF  ..
  make -j
  ```

### Parameters

 By running ```./build/perf_embv --help``` you can see the possible arguments to pass to the script. 

 - ```k``` - number of returned results.
 - ```nprobe``` - number of inverted lists to scan. 
 - ```index-path-dir``` path to the directory containing the files for the inverted index, including the compressed collection. We provide the indexes to replicate the results of the paper and the instructions to build indexes on other collections (see below).
 - ```thresh``` threshold used to filter-our non-relevant centroid scores. Values in $[0.3, 0.6]$ deliver the best results in our experiments. This is referred as *th* in the paper. See Equation 4 and Figure 2 (right).
 - ```thresh-query```. Threshold used to define $\bar{J}_i$. See Equation 6.
 - ```out-second-stage```. The number of documents that move on the centroid-interaction mechanism after the computation of Equation 4.
 - ```n-doc-to-score```. Number of documents that move to the late-interaction scoring phase. 
 - ```queries-id-file```. Path to a file in .tsv that contains the queries ids.
 - ```alldoclens-path```. Path to a .npy file containing the length of each document (i.e., the number of terms per each document).
 - ```out-file```. Path to the file where the results are written. 

### Reproducing Paper Results

Make sure to execute ```export OMP_NUM_THREADS=1``` before running a script, otherwise ```faiss``` and ```MKL``` may run in multithread mode. In this case, intra-query parallelism is not advantageous compared to single-thread execution. In case one wants to parallelize, it would be worthed to parallelize over the queries. 


We provide the parameters configurations to reproduce the results of Table 1 and Table 2 in the scripts ```results_msmarco.sh``` and ```results_lotte.sh```. Modify the script to provide the path to the downloaded indexes. The scripts to compute the metrics are taken from the ColBERT original repo. 

The indexes can be downloaded here [here](http://hpc.isti.cnr.it/~rulli/emvb-ecir2024/). They have the following name pattern ```{n_centroids}k_{M}_m_{dataset}_{compression_mod}.tar.gz```

### Extend Results on Different Collections

To run our index on your collection, you need to provide the ```doclens``` file, the ```query_ids``` file, and the ```index``` directory. 

The ```index``` directory contains the following fields. 

- ```centroids.npy```. A numpy file containing the representations of the centroids. 
- ```centroid_to_pids.txt```. A text file where the $i$-th line contains the list of documents associated with the $i$-th centroid.
- ```index-assignments.npy```. A numpy file containing, for each vector in the collection, the id of its closest centroid.
- ```pq_centroids.npy```. A numpy file containing the representation of the centroids for Product Quantization. 
- ```query_embeddings.npy```. A numpy file containing the representation of queries. 
- ```residuals.npy```. A numpy file containing the codes of the PQ-encoded vectors. 

We will provide soon the scripts to build and convert indexes on a custom collection.  

### Citation License

The source code in this repository is subject to the following citation license:

By downloading and using this software, you agree to cite the undernoted paper in any kind of material you produce where it was used to conduct search or experimentation, whether be it a research paper, dissertation, article, poster, presentation, or documentation. By using this software, you have agreed to the citation licence.

[Efficient Multi-vector Dense Retrieval with Bit Vectors](https://link.springer.com/chapter/10.1007/978-3-031-56060-6_1)

```
@inproceedings{emvb_ecir2024,
  title={Efficient Multi-vector Dense Retrieval with Bit Vectors},
  author={Nardini, Franco Maria and Rulli, Cosimo and Venturini, Rossano},
  booktitle={European Conference on Information Retrieval},
  pages={3--17},
  year={2024},
  organization={Springer}
}
```
