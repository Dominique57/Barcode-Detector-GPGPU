# GPU barecode detector
- Dominique MICHEL
- Léa Masselles
- Louis Guo
- Kenny Fung

## Needed Libraries
- Cmake 3.17+ for building purposes
- Boost : program\_options
- OpenCv: image, video, gui
- Cuda Toolkit: nvcc

## Execution requirements
> VERY IMPORTANT, the "kmeans.database" file must be in the working directory when executing the program

## Generate custom kmeans.database file
See the python code
You can create the lbp matrix of histograms from this binary using the -l or --gen-lbp option

``` bash
$ ./BarcodeDetector -l input_file.jpg
```

> There should be a input\_file.jpg.txt
This file can be import in python using OpenCv as following:

``` python
import cv2
fs = cv2.FileStorage("input_file.jpg.txt", cv2.FILE_STORAGE_READ)
fs.getFirstTopLevelNode().mat()
```

If you wish to use different than default classes (3, 12 ,15), you must edit the following source file function :
```
src/my_opencv/wrapper.cc::isCorrectClass
```

## Benchmarking
You can run a benhmarking binary by running the following command in the build directory:
``` bash
$ make bench
$ ./bench/bench
```