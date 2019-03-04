#HW02 

_Due Monday February 18th, 2019 at 11:59pm._

The intent of this homework is to become familiar with the basic mechanisms for building trivial ITK applications.

1. You must review materials in chapters 1-5 of the ITKSoftwareGuide
1. You must download, configure, and build your own private version of ITK (see Chapter in ITKSoftwareGuide) 
1. You program for this homework must have the following features:

It must convert a 2D image from one format to another, and smooth the image (recursive gaussian) with a specified 
<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /></a>
 Value

```bash
# HW02.exe <image dimension> <sigmaValue> <input image filename> <output image filename>
% HW02.exe 2 3.7811 input2Dimage.png ouput2Dimage.jpg
```

It must convert a 3D image from one format to another, and smooth the image (recursive gaussian) with a specified 
<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /></a>
 Value

```bash
# HW02.exe <image dimension> <sigmaValue> <input image filename> <output image filename>
% HW02.exe 3 4.2312 input3Dimage.nrrd output3Dimage.nii.gz
```

Useful resources:

* [ITK Examples](https://itk.org/ITKExamples/index.html)
* Useful Example files under ITK source: Examples/IO/

##### HINTS : NOTE:  The following code may not be correct!!! It is only intended to suggest one of many approaches this problem 

```cpp

const unsigned int inputDim = std::stoi( argv[1] );
const float sigmaValue = std::stod( argv[2] )
const std::string inputFilename = argv[3];
const std::string outputFilename = argv[4];

template< Dimension >
void DoWorkHere( inFileName, outFileName )
{
  using PixelType = unsigned char;
}

int main( int argc, char ** argv )
{
  if( argc < 5 )
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageDimension sigmaValue inputImageFile outputImageFile" << std::endl;
    return EXIT_FAILURE;
    }
}

try
{
}
catch( itk::ExceptionObject & err )
{
  std::cerr << "ExceptionObject caught !" << std::endl;
  std::cerr << err << std::endl;
  return EXIT_FAILURE;
}
```
