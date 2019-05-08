/*=========================================================================

  Assignment:   BME186 Homework 4
// Author: abpwrs
// Date: 03-14-2019
  Info:		Difference of Gaussians Filter - Filter/Subtract/Rescale/Write

=========================================================================*/

// inclusion of utlized header files
#include "itkImageFileReader.h"
#include "ComputeInterPupilaryDistanceCLP.h"

int main(int argc, char **argv) {
    PARSE_ARGS;

    // Definition of input/output pixel types and Dimension variable
    const unsigned int Dimension = 3;
    using InputPixelType = unsigned char;

    using InputImageType = itk::Image<InputPixelType, Dimension>;   // Defining Input/Output Image types

    using ReaderType = itk::ImageFileReader<InputImageType>;  // Defining Reader/Writer
    ReaderType::Pointer reader = ReaderType::New();   // Creating a new instance of the Reader/Writer
    reader->SetFileName(inputVolume);     // Setting of filenames --> don't need to use c_str()
    reader->Update();

    InputImageType::Pointer subjectImage = reader->GetOutput();

    //These are the pixel locations of the Left and Right Eye.
    InputImageType::IndexType LeftEyeIndex;
    LeftEyeIndex[0] = 60;
    LeftEyeIndex[1] = 127;
    LeftEyeIndex[2] = 93;

    InputImageType::IndexType RightEyeIndex;
    RightEyeIndex[0] = 59;
    RightEyeIndex[1] = 129;
    RightEyeIndex[2] = 41;

    // get the physical location of the left eye
    InputImageType::PointType leftEyePhysical;
    subjectImage->TransformIndexToPhysicalPoint(LeftEyeIndex, leftEyePhysical);

    // get the physical location of the right eye
    InputImageType::PointType rightEyePhysical;
    subjectImage->TransformIndexToPhysicalPoint(RightEyeIndex, rightEyePhysical);

    // compute the euclidean distance between the two points
    float distance = 0;
    for (int i = 0; i < Dimension; ++i) {
        distance += std::pow(rightEyePhysical[i] - leftEyePhysical[i], 2);
    }

    distance = std::sqrt(distance);


    //NOTE:  The distance between the eyes is a single value.  It is NOT a vector.
    std::cout << "===========================================" << std::endl;
    std::cout << "The euclidean distance between the eyes is " << distance << std::endl;

    return EXIT_SUCCESS;
}
