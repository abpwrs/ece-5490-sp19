/*=========================================================================

  Assignment:   BME186 Homework 4
  Author:
  Date:
  Info:		Difference of Gaussians Filter - Filter/Subtract/Rescale/Write

=========================================================================*/

// inclusion of utlized header files
#include "itkImageFileReader.h"
//TODO:  XXXX

int main(int argc, char* * argv)
{
  PARSE_ARGS;

  // Definition of input/output pixel types and Dimension variable
  const unsigned int Dimension = 6;
  using InputPixelType = XXXX ;

  using InputImageType = itk::Image<       TODO: XXXX                 >  ;   // Defining Input/Output Image types

  using ReaderType = itk::ImageFileReader<InputImageType>;  // Defining Reader/Writer
  ReaderType::Pointer reader = ReaderType::New();   // Creating a new instance of the Reader/Writer
  reader->SetFileName( inputVolume.c_str() );     // Setting of filenames
  reader->Update();

  InputImageType::Pointer subjectImage = reader->GetOutput();

  //These are the pixel locations of the Left and Right Eye.
  InputImageType::IndexType LeftEyeIndex;
  LeftEyeIndex[0]=60;
  LeftEyeIndex[1]=127;
  LeftEyeIndex[2]=93;

  InputImageType::IndexType RightEyeIndex;
  RightEyeIndex[0]=59;
  RightEyeIndex[1]=129;
  RightEyeIndex[2]=41;

  //TODO: XXXX

  //NOTE:  The distance between the eyes is a single value.  It is NOT a vector.
  std::cout << "===========================================" << std::endl;
  std::cout << "The distance between the eyes is " << XXXX << std::endl;
  
  return EXIT_SUCCESS;
}
