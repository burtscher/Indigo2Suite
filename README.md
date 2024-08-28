# Indigo2 Suite

The Indigo2 benchmark suite contains hundreds of parallel implementations of 6 graph algorithms written in CUDA, OpenMP, and C++ Threads.

If you use Indigo2, please cite the following publication.

* Yiqian Liu, Noushin Azami, Avery Vanausdal, and Martin Burtscher. "Choosing the Best Parallelization and Implementation Styles for Graph Analytics Codes: Lessons Learned from 1106 Programs." Proceedings of the 2023 ACM/IEEE International Conference for High Performance Computing, Networking, Storage, and Analysis. Article 92, pp. 1-14. November 2023.
[[doi]](https://dl.acm.org/doi/10.1145/3581784.3607038)
[[pdf]](https://cs.txstate.edu/~burtscher/papers/sc23a.pdf)
[[pptx]](https://cs.txstate.edu/~burtscher/research/Indigo2Suite/Indigo2Suite.pptx)
[[talk]](https://sc23.conference-program.com/presentation/?id=pap178&sess=sess163)

This repo provides the codes, small inputs, and scripts to run the codes. The `run_codes.py` script takes the code directory, input directory, and programming model ('0' is C++ Threads, '1' is OpenMP, and '2' is CUDA) as inputs. You can also download larger inputs using the download script.

Example command:
* python run_codes.py codes/cuda/bfs-cuda/ small-inputs/ 2


**Summary**: Indigo2 is a suite of 1106 versions of 6 key graph algorithms that are based on 13 parallelization and implementation styles as well as meaningful combinations thereof. The paper analyzes these irregular codes on a set of input graphs from different domains to determine which styles should be used, under what circumstances, and with which other styles they should be combined for best performance. The paper suggests a number of programming guidelines for developing efficient parallel programs.

You may also be interested in the predecessor [Indigo](https://cs.txstate.edu/~burtscher/research/IndigoSuite/) and successor [Indigo3](https://github.com/burtscher/Indigo3Suite/) suites as well as in the related [ECL-Suite](https://github.com/burtscher/ECL-Suite/).

*This work has been supported in part by the National Science Foundation under Grant No. 1955367 as well as by an equipment donation from NVIDIA Corporation.*
