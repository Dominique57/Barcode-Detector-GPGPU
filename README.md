# GPU barcode detector

This repository contains the school project where we used GPGPU to detect barcode.
> End to end application real-time barcode detector.

## Algorithm
The inefficient but nice to parallelise algorithm is separated in multiple steps:
- Compute local binary pattern of input image
- Apply a KNN classifier to extract a class represenigs barcode patterns (ie. a database)
- Parallelize using CUDA the local binary pattern and the neareset neighbour search (ie. finding the appropriate class)
- Post-process using mathematical morphology

## Algorithm project slide
![image](https://user-images.githubusercontent.com/9299438/137043519-0029f332-68d4-4cb1-a688-c7848d6cb851.png)

## Structure
The python folder contains the ipynb notebook that allows the KNN database generation from images
The c++ folder contains a base reference using CPU only
The cuda folder contains the parallelized algorithm and benchmarks
