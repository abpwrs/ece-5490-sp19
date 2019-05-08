//
// Author: abpwrs
// Due: 03-13-2019
//
// Based on ITK Example:
// https://itk.org/Doxygen/html/Examples_2RegistrationITKv4_2ImageRegistration8_8cxx-example.html

#include "itkImageRegistrationMethodv4.h"
#include "itkTranslationTransform.h"
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
#include <itkCenteredTransformInitializer.h>
#include <itkVersorRigid3DTransform.h>

#include "itkCommand.h"

class CommandIterUpper : public itk::Command {
public:
    using Self = CommandIterUpper;
    using Superclass = itk::Command;
    using Pointer = itk::SmartPointer<Self>;

    itkNewMacro(Self);
protected:
    CommandIterUpper() = default;

public:
    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
    using OptimizerPointer = const OptimizerType *;

    void Execute(itk::Object *caller, const itk::EventObject &event) override {
        Execute((const itk::Object *) caller, event);
    }

    void Execute(const itk::Object *object, const itk::EventObject &event) override {
        auto optimizer = static_cast< OptimizerPointer >( object );
        if (!itk::IterationEvent().CheckEvent(&event)) {
            return;
        }
        std::cout << optimizer->GetCurrentIteration() << "   ";
        std::cout << optimizer->GetValue() << "   ";
        std::cout << optimizer->GetCurrentPosition() << std::endl;
    }
};


int main(int argc, char *argv[]) {
    // parse CLI args
    PARSE_ARGS;

    // print all args
    std::cout << "fixedImageFile: " << fixedImageFile << std::endl;
    std::cout << "movingImageFile: " << movingImageFile << std::endl;
    std::cout << "outputImageFile: " << outputImageFile << std::endl;
    std::cout << "differenceImageAfterFile: " << differenceImageAfterFile << std::endl;

    // Define All Types
    // ////////////////

    // const define dimension
    constexpr unsigned int DIMENSION = 3;


    //define pixel type
    using FixedPixelType = float;
    using MovingPixelType = float;

    // define image type
    using FixedImageType = itk::Image<FixedPixelType, DIMENSION>;
    using MovingImageType = itk::Image<MovingPixelType, DIMENSION>;

    // define reader type
    using FixedReaderType = itk::ImageFileReader<FixedImageType>;
    using MovingReaderType = itk::ImageFileReader<MovingImageType>;
    // fixed image reader
    auto fixedImageReader = FixedReaderType::New();
    fixedImageReader->SetFileName(fixedImageFile);
    // moving image reader
    auto movingImageReader = MovingReaderType::New();
    movingImageReader->SetFileName(movingImageFile);


    // affine transform
    using TransformType = itk::VersorRigid3DTransform<double>;
    auto initialTransform = TransformType::New();

    // define optimizer type
    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
    auto optimizer = OptimizerType::New();

    // define metric type
    using MetricType = itk::MeanSquaresImageToImageMetricv4<FixedImageType, MovingImageType>;
    auto metric = MetricType::New();


    // define registration type
    using RegistrationType = itk::ImageRegistrationMethodv4<FixedImageType, MovingImageType, TransformType>;
    auto registration = RegistrationType::New();
    // set input image parameters of transform
    registration->SetFixedImage(fixedImageReader->GetOutput());
    registration->SetMovingImage(movingImageReader->GetOutput());
    // set registration components
    registration->SetMetric(metric);
    registration->SetOptimizer(optimizer);

    // set up initial transform to align the centers of the images
    using TransformInitializerType = itk::CenteredTransformInitializer<TransformType, FixedImageType, MovingImageType>;
    auto initializer = TransformInitializerType::New();

    initializer->SetTransform(initialTransform);
    initializer->SetFixedImage(fixedImageReader->GetOutput());
    initializer->SetMovingImage(movingImageReader->GetOutput());
    initializer->GeometryOn(); // calculate geometric center or center of mass with MomentOn
    initializer->InitializeTransform();

    using VersorType = TransformType::VersorType;
    using VectorType = VersorType::VectorType;
    VersorType rotation;
    VectorType axis;
    axis[0] = 0.0;
    axis[1] = 0.0;
    axis[2] = 1.0;
    constexpr double angle = 0;
    rotation.Set(axis, angle);
    // set initial rotation parameters
    initialTransform->SetRotation(rotation);
    // get the center of the fixed image to set as the center of rotation
    fixedImageReader->Update();
    auto fixed = fixedImageReader->GetOutput();
    FixedImageType::IndexType centerVoxel;
    for (int i = 0; i < DIMENSION; ++i) {
        centerVoxel[i] = fixed->GetLargestPossibleRegion().GetSize()[i]/2;
    }
    FixedImageType::PointType centerPhysical;
    fixed->TransformIndexToPhysicalPoint(centerVoxel, centerPhysical);
    initialTransform->SetCenter(centerPhysical);

    std::cout << "Initial Transform: " << *initialTransform << std::endl;

    registration->SetInitialTransform(initialTransform); // sets the initial transform that was calculated above


    // configure the optimizer
    using OptimizerScalesType = OptimizerType::ScalesType;
    OptimizerScalesType optimizerScales(initialTransform->GetNumberOfParameters());
    const double translationScale = 1.0 / 1000.0;
    optimizerScales[0] = 1.0;
    optimizerScales[1] = 1.0;
    optimizerScales[2] = 1.0;
    // these translations could and probably should be scaled by spacing
    optimizerScales[3] = translationScale;
    optimizerScales[4] = translationScale;
    optimizerScales[5] = translationScale;

    optimizer->SetLearningRate(0.01);
    optimizer->SetRelaxationFactor(0.5);
    optimizer->SetNumberOfIterations(500);
    optimizer->SetMinimumStepLength(0.001); // MY MINIMUM STEP LENGTH WAS TOO BIG!!!!!!
    optimizer->SetScales(optimizerScales);
    optimizer->SetReturnBestParametersAndValue(true);

    // links observer to the optimizer
    auto observer = CommandIterUpper::New();
    optimizer->AddObserver(itk::IterationEvent(), observer);

    //const define number of layers for multi-resolution
    constexpr unsigned int MULTI_RES_LAYERS = 3;

    // sets up scaling for multi-res process
    RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    shrinkFactorsPerLevel.SetSize(MULTI_RES_LAYERS);
    shrinkFactorsPerLevel[0] = 16; // 1/16
    shrinkFactorsPerLevel[1] = 4; // 1/4
    shrinkFactorsPerLevel[2] = 1; // 1/1

    RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize(MULTI_RES_LAYERS);
    smoothingSigmasPerLevel[0] = 8;
    smoothingSigmasPerLevel[1] = 4;
    smoothingSigmasPerLevel[2] = 0;

    registration->SetNumberOfLevels(MULTI_RES_LAYERS);
    registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);
    registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);

    // in case an error occurs, the exception can be caught
    try {
        // updating the registration will also update all un-updated links behind it
        registration->Update();
        std::cout << "Optimizer stop condition: "
                  << registration->GetOptimizer()->GetStopConditionDescription()
                  << std::endl;
    }
    catch (itk::ExceptionObject &err) {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    // get the final (best as we set a bool above) parameters of the registration
    const TransformType::ParametersType finalParameters = registration->GetOutput()->Get()->GetParameters();
    const double versorX = finalParameters[0];
    const double versorY = finalParameters[1];
    const double versorZ = finalParameters[2];
    const double finalTranslationX = finalParameters[3];
    const double finalTranslationY = finalParameters[4];
    const double finalTranslationZ = finalParameters[5];
    const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
    const double bestValue = optimizer->GetValue();

    // print final parameters
    // it'd probably be worthwhile to write this to a file
    std::cout << std::endl << std::endl;
    std::cout << "Result = " << std::endl;
    std::cout << "  versor X = " << versorX << std::endl;
    std::cout << "  versor Y = " << versorY << std::endl;
    std::cout << "  versor Z = " << versorZ << std::endl;
    std::cout << "  Translation X = " << finalTranslationX << std::endl;
    std::cout << "  Translation Y = " << finalTranslationY << std::endl;
    std::cout << "  Translation Z = " << finalTranslationZ << std::endl;
    std::cout << "  Iterations = " << numberOfIterations << std::endl;
    std::cout << "  Metric value = " << bestValue << std::endl;


    // create final versor3d transform from fixed and optimized parameters
    auto finalTransform = TransformType::New();
    finalTransform->SetFixedParameters(registration->GetOutput()->Get()->GetFixedParameters());
    finalTransform->SetParameters(finalParameters);


    // resample everything and write output and difference images
    using ResampleFilterType = itk::ResampleImageFilter<MovingImageType, FixedImageType>;

    auto resampler = ResampleFilterType::New();
    auto fixedImage = fixedImageReader->GetOutput();

    resampler->SetTransform(finalTransform);
    resampler->SetInput(movingImageReader->GetOutput());
    resampler->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
    resampler->SetOutputOrigin(fixedImage->GetOrigin());
    resampler->SetOutputSpacing(fixedImage->GetSpacing());
    resampler->SetOutputDirection(fixedImage->GetDirection());
    resampler->SetDefaultPixelValue(100);


    // define output image type
    using OutputPixelType = unsigned char;
    using OutputImageType = itk::Image<OutputPixelType, DIMENSION>;
    // cast from fixed type float to output type unsigned char
    using CastFilterType = itk::CastImageFilter<FixedImageType, OutputImageType>;
    // your friendly neighborhood itkImageFileWriter
    using WriterType = itk::ImageFileWriter<OutputImageType>;

    // instantiate caster and writer
    auto caster = CastFilterType::New();
    auto writer = WriterType::New();

    // link caster to resampler
    caster->SetInput(resampler->GetOutput());
    // link writer to caster
    writer->SetInput(caster->GetOutput());
    // set where to write output image to
    writer->SetFileName(outputImageFile);
    // write output image
    writer->Update();


    // create difference image
    // define types for output
    using SubtractImageFilterType = itk::SubtractImageFilter<FixedImageType, FixedImageType, FixedImageType>;
    using RescaleImageFilterType = itk::RescaleIntensityImageFilter<FixedImageType, OutputImageType>;

    // instantiate difference and rescale
    auto subtractFilter = SubtractImageFilterType::New();
    auto rescaleFilter = RescaleImageFilterType::New();

    subtractFilter->SetInput1(fixedImageReader->GetOutput());
    subtractFilter->SetInput2(resampler->GetOutput());
    resampler->SetDefaultPixelValue(0);

    rescaleFilter->SetInput(subtractFilter->GetOutput());
    rescaleFilter->SetOutputMinimum(0);
    rescaleFilter->SetOutputMaximum(255);

    auto differenceWriter = WriterType::New();
    differenceWriter->SetInput(rescaleFilter->GetOutput());
    differenceWriter->SetFileName(differenceImageAfterFile);
    differenceWriter->Update();

    return EXIT_SUCCESS;
}
