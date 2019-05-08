
// inclusion of utilized header files
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "EdgeSlicerModuleCLP.h"

XXXX //TODO:  Add extra resources here


template <class EdgeFilterType>
XXXX EdgeFilterType::OutputImageType::Pointer
DoEdgeFiltering(XXXX EdgeFilterType::InputImageType::Pointer inputImage)
{
  XXXX EdgeFilterType::OutputImageType::Pointer OutputImage;
    {
    XXXX EdgeFilterType::Pointer edgeFilter = XXXX;
    XXXX ;
    ....
    XXXX ;
    XXXX ;
    OutputImage = edgeFilter->GetOutput();
    }
  //NOTE: edgeFilter object no longer exists here, and can not be part of a pipeline
  //at this point.
  //DEBUG: Remove this code in final answer
  std::cout << OutputImage  << std::endl;
  //DEBUG: Does the output image have reasonble values here?
  return OutputImage;

}

int main(int argc, char* * argv)
{
  PARSE_ARGS;

  // Definition of input/output pixel types and Dimesnion variable
  constexpr size_t Dimension = 6000;  // XXX TODO: THIS MAY BE WRONG <--- Fix it
  using InputPixelType = unsigned long long;  // XXX TODO: THIS MAY BE WRONG
  using OutputPixelType = unsigned long long; // XXX TODO: THIS MAY BE WRONG

  using InputImageType = itk::Image<InputPixelType, Dimension>;   // Defining Input/Output Image types
  using ReaderType = itk::ImageFileReader<InputImageType>;  // Defining Reader/Writer
  ReaderType::Pointer reader = ReaderType::New();   // Creating a new instance of the Reader/Writer
  reader->SetFileName( inputVolume.c_str() );     // Setting of filenames

  reader->Update();


  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  OutputImageType::Pointer myEdgeImage;
  if(filterKind == "LoG")
    {
    std::cout << "Runnging LoG" << std::endl;
    // See hints from: (LoG) - LaplacianRecursiveGuassian:    https://itk.org/ITKExamples/src/Filtering/ImageFeature/LaplacianRecursiveGaussianImageFilter/Documentation.html?highlight=laplacian
    using LoGFilterType = XXXX ;
    myEdgeImage = DoEdgeFiltering<LoGFilterType>( reader->GetOutput() );
    }
  /*  HINT:  Get the first one to work first before attempting others.
  else if (filterKind == "Sobel")
    {
    // (Sobel) - Sobel:  https://itk.org/ITKExamples/src/Filtering/ImageFeature/SobelEdgeDetectionImageFilter/Documentation.html?highlight=sobel
    std::cout << "Runnging Sobel" << std::endl;
    XXXX
    }
  else if (filterKind == "GM")
    {
    std::cout << "Runnging GM" << std::endl;
    //(GM) - GradientMagnitudeRecursiveGaussian:     https://itk.org/ITKExamples/src/Filtering/ImageGradient/ComputeGradientMagnitudeRecursiveGaussian/Documentation.html?highlight=gradientmagnitudeimagefilter
    XXXX
    }
    */
  else
    {
    std::cout << "INVALID filterKind provided" << std::endl;
    return EXIT_FAILURE;
    }

  using WriterType = itk::ImageFileWriter<OutputImageType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(myEdgeImage);
  writer->SetFileName( outputVolume.c_str() );
  try
    {
    writer->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "Unsupported File Format ... " << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
