//
// Created by Hans Johnson
// Authored by abpwrs
//
#include "itkImage.h"
#include "itkImageIOBase.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRecursiveGaussianImageFilter.h"
#include <iostream>

// modeled after:
// https://itk.org/ITKExamples/src/Filtering/ImageFeature/LaplacianRecursiveGaussianImageFilter/Documentation.html
template<unsigned int dimension>
int blur(const std::string &inFileName, const std::string &outFileName, const float sigma = 3.0f) {
    // define input image type
    using InputPixelType = unsigned char;
    using InputImageType = itk::Image<InputPixelType, dimension>;

    // define reader
    using ReaderType = itk::ImageFileReader<InputImageType>;
    typename ReaderType::Pointer reader = ReaderType::New();
    // set reader filename
    reader->SetFileName(inFileName);

    // define output image type
    using OutputPixelType = unsigned char;
    using OutputImageType = itk::Image<OutputPixelType, dimension>;

    // define filter
    using FilterType = itk::RecursiveGaussianImageFilter<InputImageType, OutputImageType>;
    typename FilterType::Pointer filter = FilterType::New();
    // define filter parameters
    filter->SetSigma(sigma);
    filter->SetInput(reader->GetOutput());

    // define writer
    using WriterType = itk::ImageFileWriter<OutputImageType>;
    typename WriterType::Pointer writer = WriterType::New();

    // set writer output name
    writer->SetFileName(outFileName);
    // link filter output to writer input
    writer->SetInput(filter->GetOutput());

    try {
        writer->Update();
    }
    catch (itk::ExceptionObject &err) {
        std::cerr << "ExceptionObject caught!" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}


int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0];
        std::cerr << " <image dimension> <sigma> <input image filename> <output image filename>";
        std::cerr << std::endl;
        return EXIT_FAILURE;
    }

    const unsigned int inputDim = std::stoi(argv[1]);
    const float sigmaValue = std::stod(argv[2]);
    const std::string inputFilename = argv[3];
    const std::string outputFilename = argv[4];

    int exit_status = EXIT_FAILURE;
    switch (inputDim) {
        case 1:
            exit_status = blur<1>(inputFilename, outputFilename, sigmaValue);
            break;
        case 2:
            exit_status = blur<2>(inputFilename, outputFilename, sigmaValue);
            break;
        case 3:
            exit_status = blur<3>(inputFilename, outputFilename, sigmaValue);
            break;
        case 4:
            exit_status = blur<4>(inputFilename, outputFilename, sigmaValue);
            break;
        case 5:
            exit_status = blur<5>(inputFilename, outputFilename, sigmaValue);
            break;
        case 6:
            exit_status = blur<6>(inputFilename, outputFilename, sigmaValue);
            break;
        case 7:
            exit_status = blur<7>(inputFilename, outputFilename, sigmaValue);
            break;
        case 8:
            exit_status = blur<8>(inputFilename, outputFilename, sigmaValue);
            break;
        case 9:
            exit_status = blur<9>(inputFilename, outputFilename, sigmaValue);
            break;
        case 10:
            exit_status = blur<10>(inputFilename, outputFilename, sigmaValue);
            break;
        default:
            std::cerr << "Unsupported dimension: " << inputDim << std::endl;
    }

    return exit_status;
}