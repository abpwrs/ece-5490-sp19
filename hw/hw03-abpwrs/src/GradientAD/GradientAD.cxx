//
// Created by Alex Powers on 21-02-2019.
//
// https://itk.org/Doxygen/html/classitk_1_1GradientAnisotropicDiffusionImageFilter.html

#include "GradientADCLP.h" // <-- NOTE THIS FILE WAS AUTO GENERATED by GENERATECLP()

// ITK Imports
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkGradientAnisotropicDiffusionImageFilter.h"

int main(int argc, char *argv[]) {
    PARSE_ARGS;
    std::cout << "The input volume is: " << inputVolume << std::endl;
    std::cout << "The output volume is: " << outputVolume << std::endl;
    std::cout << "The conductance value is: " << conductance << std::endl;
    std::cout << "The timestep  is: " << timeStep << std::endl;
    std::cout << "Number of iterations is: " << numberOfIterations << std::endl;

    // define dimension
    constexpr unsigned int dimension = 3;

    // define input image type
    using InputPixelType = float; // must be a float/double for anisotropic diffusion
    using InputImageType = itk::Image<InputPixelType, dimension>;

    // define output image type
    using OutputPixelType = float; // must be a float/double for anisotropic diffusion
    using OutputImageType = itk::Image<OutputPixelType, dimension>;

    // define reader
    using ReaderType = itk::ImageFileReader<InputImageType>;
    typename ReaderType::Pointer reader = ReaderType::New();
    // set reader filename
    reader->SetFileName(inputVolume);

    // define filter
    using FilterType = itk::GradientAnisotropicDiffusionImageFilter<InputImageType, OutputImageType>;
    typename FilterType::Pointer filter = FilterType::New();
    // define filter parameters
    filter->SetConductanceParameter(conductance);
    filter->SetNumberOfIterations(numberOfIterations);
    filter->SetTimeStep(timeStep);
    filter->SetInput(reader->GetOutput());

    // define writer
    using WriterType = itk::ImageFileWriter<OutputImageType>;
    typename WriterType::Pointer writer = WriterType::New();

    // set writer output name
    writer->SetFileName(outputVolume);
    // link filter output to writer input
    writer->SetInput(filter->GetOutput());

    try {
        writer->Update();
    }
    catch (itk::ExceptionObject &err) {
        std::cerr << "EXCEPTION! OH MY!" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}