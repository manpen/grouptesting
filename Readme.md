# Building
In order to build the code, you may use the following instructions:
```bash
git submodule update --init --recursive
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

*Remark*: Please make sure you have a somewhat recent C++ compiler (tested with gcc8) and cmake (Version 3.10 or higher) installed.

# Executing
The source file `main.cpp` defines some experimental campaings that can be triggered by executing for instance
```
build/grouptesting run1e7
``` 
Observe that they may take very long, as the code is intentend to be used in conjunction with a scheduler (e.g. SLURM) that will preemptively stop the computation after a fixed amount of time (typically 8 to 50h).

# Analysis
The code stores all results to CSV data files in the current working directory.
To facilitate the parallel on multiple machines sharing a common filesystem, the data files are indentified by a meaningful prefix and suffixed by a random string.
The analysis scripts expect the data file to be moved into `./data/`.

The analysis script `Publication.ipynb` is implemented using Python, Pandas and Matplotlib in a Juptyer Lab notebook.