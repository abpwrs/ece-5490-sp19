// Author: abpwrs
// Date: 03-14-2019
// inclusion of utilized header files
#include <itkRecursiveGaussianImageFilter.h>
#include <itkLaplacianRecursiveGaussianImageFilter.h>
#include <itkSobelEdgeDetectionImageFilter.h>
#include <itkGradientMagnitudeRecursiveGaussianImageFilter.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "EdgeSlicerModuleCLP.h"


template<class EdgeFilterType>
typename EdgeFilterType::OutputImageType::Pointer
DoEdgeFiltering(typename EdgeFilterType::InputImageType::Pointer inputImage) {
    typename EdgeFilterType::OutputImageType::Pointer OutputImage = EdgeFilterType::OutputImageType::New();
    {
        typename EdgeFilterType::Pointer edgeFilter = EdgeFilterType::New();
        edgeFilter->SetInput(inputImage);
        edgeFilter->Update();
        OutputImage = edgeFilter->GetOutput();
    }

    // std::cout << OutputImage << std::endl;
    return OutputImage;

}

int main(int argc, char **argv) {
    PARSE_ARGS;

    // Definition of input/output pixel types and Dimesnion variable
    constexpr unsigned int Dimension = 3;
    using InputPixelType = float;
    using OutputPixelType = float;

    using InputImageType = itk::Image<InputPixelType, Dimension>;   // Defining Input/Output Image types
    using ReaderType = itk::ImageFileReader<InputImageType>;  // Defining Reader/Writer
    ReaderType::Pointer reader = ReaderType::New();   // Creating a new instance of the Reader/Writer
    reader->SetFileName(inputVolume);     // Setting of filenames

    reader->Update(); // this my not be neccesary, we will see


    using OutputImageType = itk::Image<OutputPixelType, Dimension>;
    OutputImageType::Pointer myEdgeImage;
    if (filterKind == "LoG") {
        std::cout << "Runnging LoG" << std::endl;
        // See hints from: (LoG) - LaplacianRecursiveGuassian:    https://itk.org/ITKExamples/src/Filtering/ImageFeature/LaplacianRecursiveGaussianImageFilter/Documentation.html?highlight=laplacian
        using LoGFilterType = itk::LaplacianRecursiveGaussianImageFilter<InputImageType, OutputImageType>;
        myEdgeImage = DoEdgeFiltering<LoGFilterType>(reader->GetOutput());
    }
        //    HINT:  Get the first one to work first before attempting others.
    else if (filterKind == "Sobel") {
        // (Sobel) - Sobel:  https://itk.org/ITKExamples/src/Filtering/ImageFeature/SobelEdgeDetectionImageFilter/Documentation.html?highlight=sobel
        std::cout << "Runnging Sobel" << std::endl;
        using SobelFilterType = itk::SobelEdgeDetectionImageFilter<InputImageType, OutputImageType>;
        myEdgeImage = DoEdgeFiltering<SobelFilterType>(reader->GetOutput());
    } else if (filterKind == "GM") {
        std::cout << "Runnging GM" << std::endl;
        //(GM) - GradientMagnitudeRecursiveGaussian:     https://itk.org/ITKExamples/src/Filtering/ImageGradient/ComputeGradientMagnitudeRecursiveGaussian/Documentation.html?highlight=gradientmagnitudeimagefilter
        using GMFilterType = itk::GradientMagnitudeRecursiveGaussianImageFilter<InputImageType, OutputImageType>;
        myEdgeImage = DoEdgeFiltering<GMFilterType>(reader->GetOutput());
    } else {
        std::cout << "INVALID filterKind provided" << std::endl;
        return EXIT_FAILURE;
    }

    using WriterType = itk::ImageFileWriter<OutputImageType>;
    WriterType::Pointer writer = WriterType::New();
    writer->SetInput(myEdgeImage);
    writer->SetFileName(outputVolume.c_str());
    try {
        writer->Update();
    }
    catch (itk::ExceptionObject &err) {
        std::cerr << "Unsupported File Format ... " << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
