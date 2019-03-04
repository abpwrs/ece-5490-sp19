#include "itkImage.h"
#include "itkImageFileReader.h"
#include <iostream>

// Based off of
// https://itk.org/ITKExamples/src/IO/ImageBase/ReadAnImage/Documentation.html
int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << " myFile.nii.gz";
        std::cerr << std::endl;
        return EXIT_FAILURE;
    }

    const std::string file_name = argv[1];
    const unsigned int Dimension = 3;
    using PixelType = unsigned char;
    using ImageType = itk::Image<PixelType, Dimension>;
    using ReaderType = itk::ImageFileReader<ImageType>;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(file_name);
    reader->Update();

    ImageType::Pointer image = reader->GetOutput();
    std::cout << "Origin: " << image->GetOrigin() << std::endl;
    std::cout << "Spacing: " << image->GetSpacing() << std::endl;
    ImageType::RegionType region = image->GetLargestPossibleRegion();
    std::cout << "Size: " << region.GetSize() << std::endl;


    return EXIT_SUCCESS;
}

