//
// Created by abpwrs on 1/22/19.
//


#include <iostream>
#include "itkImage.h"

int main(int argc, char *argv[]) {
    using PixelType = unsigned int;
    const int Dimension = 2;
    using ImageType = itk::Image<PixelType, Dimension>;
    ImageType::Pointer image = ImageType::New();


    ImageType::SizeType size;
    size[0] = 5;// x direction//
    size[1] = 5;// y direction


    ImageType::IndexType start;
    start[0] = 10; // x direction
    start[1] = 10; // y direction

    ImageType::RegionType region;
    region.SetSize( size );
    region.SetIndex( start );

    ImageType::SpacingType spacing;
    spacing[0]=2;
    spacing[1]=2;

    ImageType::PointType origin;
    origin[0]=0;
    origin[1]=0;

    image->SetRegions( region );
    image->SetOrigin(origin);
    image->SetSpacing(spacing);
    image->Allocate();
    image->FillBuffer(7);




    std::cout << "ITK Hello World" << std::endl;

    return EXIT_SUCCESS;
}
