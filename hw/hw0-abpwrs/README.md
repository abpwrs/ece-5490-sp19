# HW1 MIATT

DUE:  January 31, 2019 @ 4:49pm

ASSIGNMENT: Modify this README.md file that uses github markdown syntax to answers the following questions:

## Short answers:

###  What does the cmake program do?
```
cmake is a compiler and platform independent build manager
```

###  How many regions does an ITK image require to be defined?
```
an ITK image must define 3 regions
```

###  Describe the role of each of the ITK image regions.
```
LargestPossibleRegion:
describes the entire dataset

BufferedRegion:
the portion of the image in physical memory

RequestedRegion:
the portion of the image being processed
````
:w
###  How should the key work "typedef" be used in your homework assignments?
```
the keyword "typedef" should be used to define Types that will be used 
repeatedly in order to simplify and improve code readability

ex:
typedef itk::Image< unsigned char, 3 > ImageType;
```

###  What does it mean to be "const correct" when writing ITK classes? (HINT: http://en.wikipedia.org/wiki/Const-correctness)
```
the keyword "const" just makes a data type read-only.
to this extent, const correctness is not using methods that mutate the state 
of const variables, or modifying const variables yourself. 
```

## Programming assignment

In this part of the assignment you will create a binary program called `HW0.exe`.  The source code for this assignment must be in a directory called `src` with at least the following files:

``` bash
src/CMakeLists.txt
src/HW0.cxx
```

The HW0 binary must instantiate an image that:

* holds `std::complex<double>` values at each voxel location.
* Is a 7 dimensional image
* Has a start index at 0's
* Has 17 voxels in each direction.
* Has an identity direction
* Has spacing of 1.55 units between each voxel in each direction
* Fill the image with the value (2.2+3.3i) 
* Has an Origin as at 0.0's in each direction





