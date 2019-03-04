#include "itkImageRegistrationMethodv4.h"
#include "itkTranslationTransform.h"
#include "itkEuler3DTransform.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSubtractImageFilter.h"

#include "HW2IterationLogger.h"

#include "registerCLP.h"

#include <iostream>

int main( int argc, char *argv[] )
{
  PARSE_ARGS;

  std::cout << "fixedImageFile: " << fixedImageFile << std::endl;
  std::cout << "movingImageFile: " << movingImageFile << std::endl;
  std::cout << "outputImageFile: " << outputImageFile << std::endl;
  std::cout << "differenceImageAfterFile: " << differenceImageAfterFile << std::endl;

  constexpr unsigned int Dimension = 3;
  using PixelType = float;


    // define input image type
    using InputPixelType = unsigned char;
    using InputImageType = itk::Image<InputPixelType, Dimension>;

    // define output image type
    using OutputPixelType =  unsigned char;
    using OutputImageType = itk::Image<OutputPixelType, Dimension>;

    using ReaderType = itk::ImageFileReader<InputImageType >;

    // fixed image reader
    ReaderType::Pointer fixedImage = ReaderType::New();
    fixedImage->SetFileName(fixedImageFile);

    // moving image reader
    ReaderType::Pointer movingImage = ReaderType::New();
    movingImage->SetFileName(movingImageFile);

    fixedImage->Update();
    movingImage->Update();

    std::cout<<*fixedImage->GetOutput()<<std::endl;
    std::cout<<*movingImage->GetOutput()<<std::endl;


    return EXIT_SUCCESS;
}
