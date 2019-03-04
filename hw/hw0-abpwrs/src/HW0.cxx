//
// abpwrs @ 01-28-2019
//

#include "itkImage.h"

int main(int argc, char *argv[]) {

    using PixelType = std::complex<double>;

    const int DIMENSION = 7;

    using ImageType = itk::Image<PixelType, DIMENSION>;

    const std::complex<double> DEFAULT_VALUE(2.2, 3.3);

    ImageType::Pointer image = ImageType::New();

    // Initialize Region Size
    ImageType::SizeType size;
    size.Fill(17);

    // Initialize Region Spacing
    ImageType::IndexType start;
    start.Fill(0);

    // Create Image Region
    ImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

    // Allocate Image Memory from Region
    image->SetRegions(region);
    image->Allocate();
    image->FillBuffer(DEFAULT_VALUE);

    // Initialize spacing
    ImageType::SpacingType spacing;
    spacing.Fill(1.55);
    image->SetSpacing(spacing);

    // Initialize Origin
    ImageType::PointType origin;
    origin.Fill(0.0);
    image->SetOrigin(origin);

    return EXIT_SUCCESS;
}
