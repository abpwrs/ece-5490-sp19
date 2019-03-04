# HW04 - Registration Assignment

DUE: Wednesday March 13, 2019 at 11:59pm

## Description

Imagine that the local hospital purchased two identical MRI scanning machines and installed them
in different departments.  You must validate that the two scanners produce image datasets that are
similar.   A volunteer has agreed to have 3D scan of their brain taken in the first scanner, and then
again in the second scanner.  Your task is to register these two scans so that a clinician can 
determining if the two scanners have any systematic differences in their scan appearance or qualtiy.

The two scans resulting from this experiment are listed below:
```
/nfsscratch/opt/ece5490/data/fixed_t1.nii.gz
/nfsscratch/opt/ece5490/data/moving_rot_newscanner_t1.nii.gz
```

### Part 1
Compute the physical dimensions of the cube occupied by each image.


> TODO: Place your answers here

> fixed_t1.nii has physical dimensions ( 240mm x 186mm  x 240mm)

> moving_rot_newscanner_t1.nii has physical dimensions ( 240mm x 186mm  x 240mm)

Compute the value of the center voxel in the image for both the moving and the fixed image:


> TODO: Place your answers here:

> fixed_t1.nii center voxel physical location:                   (120.0, 64.5, 93.0)mm

> moving_rot_newscanner_t1.nii center voxel physical location:   (137.7, -202.3, 90.6)mm

HINT: `GetSize()[0]/2;`
HINT: Index types and Size types are different! `ImageType::SizeType` != `ImageType::IndexType`
HINT: Filters are not run until the "Update()" member function is called.  
Before Update() is called, the image objects are often default values with no pixels allocated.

### Part 2

Develop a program for registering the 3D images.  

The command line for running the registration should be runnable similar to the following:

```bash
register_hawkid \
   --fixedImageFile   /nfsscratch/opt/ece5490/data/fixed_t1.nii.gz \
   --movingImageFile  /nfsscratch/opt/ece5490/data/moving_rot_newscanner_t1.nii.gz \
   --outputImageFile  /tmp/deformed_moving.nii.gz \
   --differenceImageAfterFile /tmp/diff_after.nii.gz
```


HINT: `itkEuler3DTransform` The serialization of the optimizable parameters is an array of 6 elements. 
The first 3 represents three euler angle of rotation respectively about the X, Y and Z axis.
The last 3 parameters defines the translation in each dimension.


HINT: Optimizer state parameters (500, .1, 0.01, 0.5 ) that worked for me for SetLearningRate, SetMinimumStepLength, SetRelaxationFactor,
SetNumberOfIterations. (not in this order, you need to figure out the ordering.) If you have a different solution, you may need different values.


HINT: The parameter space to be optimized over is NOT isotropic!!! Changes of 1.0 in 𝞀,Φ,θ have drastically different effects
on the metric than do changes of 1.0 in translations.  ScalesType. This array defines scale to be applied
 to parameters before being evaluated in the cost function. This allows to map to a more convenient space.
 In particular this is used to normalize parameter spaces in which some parameters have a different dynamic range.

```
  OptimizerType::ScalesType optimizerScales(initialTransform->GetNumberOfParameters());
  const double translationScale = SOME_SCALE_FACTOR;
  optimizerScales.Fill(1.0);
  optimizerScales[XXX]  = translationScale;
```

> TODO: What translationScale value did you use? = SOME_SCALE_FACTOR
> TODO: Why did you choose this value?

HINT: Develop a multi-resolution registration approach.  
1/8 scale, 1/4 scale, full scale, (Smooth by alot, a little, nosmoothing).  
You may find your programs from HW2 useful in understanding the smoothing.

HINT: Set the center of the `movingInitialTransform->SetCenter(  CNTR  );`

### Part 3

Make a plot of the metric values at each iteration vs interation for all the iterations in the multi-resolution run.
You will need to capture those values somehow from the registration process.
HINT:
```cxx
  using ObserverType = CommandIterationUpdate< OptimizerType  >;
  ObserverType::Pointer observer = ObserverType::New();
  MYOBJECT_NAME->AddObserver(itk::IterationEvent(),observer);
```


> TODO: Insert plot here.


> TODO: Describe why there are discontinuities in the metric plot when the resolution value chagnes:
HERE



### Part 4
In this section we will examine the importance of transform fixed parameters.

#### 4.1
> TODO: Fill out the estimated euler transform parameter values from the registration, (it should have fixed parameters set).

```asm
Result = 
 rho = ???? degrees
 phi = ???? degrees
 theta = ????? degrees
 Translation X = ?????
 Translation Y = ?????
 Translation Z = ?????
 Metric value  = ?????
```  

#### 4.2
Comment out the line in your code that set the fixed parameters for the Euler transform, re-run the same experiment.

> TODO: Fill out the estimated euler transform parameter values from the registration, without setting the transform fixed parameters.

```asm
Result = 
 rho = ???? degrees
 phi = ???? degrees
 theta = ????? degrees
 Translation X = ?????
 Translation Y = ?????
 Translation Z = ?????
 Metric value  = ?????
```  

> TODO: Describe why the results are different.