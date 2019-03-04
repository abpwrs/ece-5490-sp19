// Write a program to register two 3D images that only differ
// by a translation transform.

// Program command line arguments
/*
 --fixedImageFile /nfsscratch/opt/ece5490/data/fixed_t1.nii.gz
 --movingImageFile /nfsscratch/opt/ece5490/data/moving_t1.nii.gz
 --outputImageFile /tmp/test.nii.gz
 --differenceImageBeforeFile /tmp/diffbefore.nii.gz
 --differenceImageAfterFile /tmp/diffafter.nii.gz
 */

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
// TODO: Add auto-generated SlicerExecutionModel header here
#include "registerCLP.h"

int main(int argc, char *argv[]) {
    //TODO: Add the 1 line SlicerExecutionModel macro that is parses the arguments
    PARSE_ARGS;
    std::cout << "fixedImageFile: " << fixedImageFile << std::endl;
    std::cout << "movingImageFile: " << movingImageFile << std::endl;
    std::cout << "outputImageFile: " << outputImageFile << std::endl;
    std::cout << "differenceImageBeforeFile: " << differenceImageBeforeFile << std::endl;
    std::cout << "differenceImageAfterFile: " << differenceImageAfterFile << std::endl;

    // TODO:  Your program must work in 3D, and only with float pixel types.
    //        You program only needs to register images from the same modality for translation transforms.
    using PixelType = float;
    constexpr unsigned int Dimension = 3;

    using FixedImageType = itk::Image<PixelType, Dimension>;
    using MovingImageType = itk::Image<PixelType, Dimension>;

    using FixedImageReaderType = itk::ImageFileReader<FixedImageType>;
    auto fixedImageReader = FixedImageReaderType::New();
    fixedImageReader->SetFileName(fixedImageFile);

    using MovingImageReaderType = itk::ImageFileReader<MovingImageType>;
    auto movingImageReader = MovingImageReaderType::New();
    movingImageReader->SetFileName(movingImageFile);

    //TODO:  Set the proper transform type -- DONE
    using TransformType = itk::TranslationTransform<PixelType, Dimension>;
    auto initialTransform = TransformType::New();


    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
    auto optimizer = OptimizerType::New();
    optimizer->SetLearningRate(4);
    optimizer->SetMinimumStepLength(0.01);
    optimizer->SetRelaxationFactor(0.5);
    optimizer->SetNumberOfIterations(200);


    //TODO: Set the proper metric type -- DONE
    using MetricType = itk::MeanSquaresImageToImageMetricv4<FixedImageType, MovingImageType>;
    auto metric = MetricType::New();


    using RegistrationType = itk::ImageRegistrationMethodv4<FixedImageType, MovingImageType>;
    auto registration = RegistrationType::New();

    //TODO: Set the registration state:
    registration->SetInitialTransform(initialTransform);
    registration->SetMetric(metric);
    registration->SetOptimizer(optimizer);
    registration->SetFixedImage(fixedImageReader->GetOutput());
    registration->SetMovingImage(movingImageReader->GetOutput());

    auto movingInitialTransform = TransformType::New();
    TransformType::ParametersType initialParameters(movingInitialTransform->GetNumberOfParameters());

    //TODO: Manually set the initial parameter staring values.  i.e. where should the optimization start?
    initialParameters[0] = 0.0;
    initialParameters[1] = 0.0;
    initialParameters[2] = 0.0;

    movingInitialTransform->SetParameters(initialParameters);
    registration->SetMovingInitialTransform(movingInitialTransform);


    auto identityTransform = TransformType::New();
    identityTransform->SetIdentity();
    registration->SetFixedInitialTransform(identityTransform);

    constexpr unsigned int numberOfLevels = 1;
    registration->SetNumberOfLevels(numberOfLevels);

    RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    shrinkFactorsPerLevel.SetSize(1);
    shrinkFactorsPerLevel[0] = 1;
    registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);

    RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize(1);
    smoothingSigmasPerLevel[0] = 0;
    registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);


    try {
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

    auto transform = registration->GetTransform();
    auto finalParameters = transform->GetParameters();
    auto translationAlongX = finalParameters[0];
    //TODO: Get all parameters
    auto translationAlongY = finalParameters[1];
    auto translationAlongZ = finalParameters[2];

    auto numberOfIterations = optimizer->GetCurrentIteration();
    auto bestValue = optimizer->GetValue();

    std::cout << "Result = " << std::endl;
    std::cout << " Translation X = " << translationAlongX << std::endl;
    //TODO: Print other parameters
    std::cout << " Translation Y = " << translationAlongY << std::endl;
    std::cout << " Translation Z = " << translationAlongZ << std::endl;

    std::cout << " Iterations    = " << numberOfIterations << std::endl;
    std::cout << " Metric value  = " << bestValue << std::endl;


    using CompositeTransformType = itk::CompositeTransform<
            double,
            Dimension>;
    auto outputCompositeTransform = CompositeTransformType::New();
    outputCompositeTransform->AddTransform(movingInitialTransform);
    outputCompositeTransform->AddTransform(
            registration->GetModifiableTransform());

    using ResampleFilterType = itk::ResampleImageFilter<
            MovingImageType,
            FixedImageType>;
    auto resampler = ResampleFilterType::New();
    resampler->SetInput(movingImageReader->GetOutput());
    resampler->SetTransform(outputCompositeTransform);
    auto fixedImage = fixedImageReader->GetOutput();
    resampler->SetUseReferenceImage(true);
    resampler->SetReferenceImage(fixedImage);
    resampler->SetDefaultPixelValue(100);

    using OutputPixelType = unsigned char;
    using OutputImageType = itk::Image<OutputPixelType, Dimension>;

    using CastFilterType = itk::CastImageFilter<
            FixedImageType,
            OutputImageType>;
    auto caster = CastFilterType::New();
    caster->SetInput(resampler->GetOutput());

    using WriterType = itk::ImageFileWriter<OutputImageType>;
    auto writer = WriterType::New();
    writer->SetFileName(outputImageFile);
    writer->SetInput(caster->GetOutput());
    writer->Update();

    using DifferenceFilterType = itk::SubtractImageFilter<
            FixedImageType,
            FixedImageType,
            FixedImageType>;
    auto difference = DifferenceFilterType::New();
    difference->SetInput1(fixedImageReader->GetOutput());
    difference->SetInput2(resampler->GetOutput());

    using RescalerType = itk::RescaleIntensityImageFilter<
            FixedImageType,
            OutputImageType>;
    auto intensityRescaler = RescalerType::New();
    intensityRescaler->SetInput(difference->GetOutput());
    intensityRescaler->SetOutputMinimum(itk::NumericTraits<OutputPixelType>::min());
    intensityRescaler->SetOutputMaximum(itk::NumericTraits<OutputPixelType>::max());

    resampler->SetDefaultPixelValue(1);

    writer->SetInput(intensityRescaler->GetOutput());
    writer->SetFileName(differenceImageAfterFile);
    writer->Update();

    resampler->SetTransform(identityTransform);
    writer->SetFileName(differenceImageBeforeFile);
    writer->Update();


    return EXIT_SUCCESS;
}
