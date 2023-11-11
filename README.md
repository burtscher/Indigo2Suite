# Indigo2 Suite

Indigo2 consists of hundreds of different implementations for 6 graph algorithms in 3 programming models. For full details, please refer to out paper:
* Yiqian Liu, Noushin Azami, Avery Vanausdal, and Martin Burtscher. Choosing the Best Parallelization and Implementation Styles for Graph Analytics Codes: Lessons Learned from 1106 Programs. Proceedings of the 2023 ACM/IEEE International Conference for High Performance Computing, Networking, Storage, and Analysis. November 2023. [[SC'23]](https://userweb.cs.txstate.edu/~burtscher/papers/sc23a.pdf)

This repo provides the codes, small inputs, and scripts to test the codes. The `run_codes.py` take the code directory, input directory, programming model (0 is C++ Threads, 1 is OpenMP, and 2 is CUDA) as inputs to run the codes through a set of inputs. You can also download larger inputs by the download script.

Example command:
* python script/run_codes.py codes/cuda/bfs-cuda small-inputs 2
