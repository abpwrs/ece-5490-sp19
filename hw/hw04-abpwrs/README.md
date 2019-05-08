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


> fixed_t1.nii has physical dimensions ( 240mm x 186mm  x 240mm)

> moving_rot_newscanner_t1.nii has physical dimensions ( 240mm x 186mm  x 240mm)

Compute the value of the center voxel in the image for both the moving and the fixed image:

> fixed_t1.nii center voxel physical location:                   (120.0, 64.5, 93.0)mm

> moving_rot_newscanner_t1.nii center voxel physical location:   (137.7, -202.3, 90.6)mm

### Part 2

Develop a program for registering the 3D images.  

The command line for running the registration should be runnable similar to the following:

```bash
register_abpwrs \
   --fixedImageFile   /nfsscratch/opt/ece5490/data/fixed_t1.nii.gz \
   --movingImageFile  /nfsscratch/opt/ece5490/data/moving_rot_newscanner_t1.nii.gz \
   --outputImageFile  /tmp/deformed_moving.nii.gz \
   --differenceImageAfterFile /tmp/diff_after.nii.gz
```

```
  OptimizerType::ScalesType optimizerScales(initialTransform->GetNumberOfParameters());
  const double translationScale = 1.0/1000.0;
  optimizerScales.Fill(1.0);
  optimizerScales[3]  = translationScale;
  optimizerScales[4]  = translationScale;
  optimizerScales[5]  = translationScale;
```

#### Reason for scaling factor choice: 
I used a translation scale of 1/1000 because we have already done most of the work involved for translation with the fixed parameters that center the two images. Thus the changes to these parameters by the optimizer should be much smaller.

### Part 3

Make a plot of the metric values at each iteration vs interation for all the iterations in the multi-resolution run.
You will need to capture those values somehow from the registration process.



![loss image](https://github.com/UIOWAECE5490SP19/hw04-abpwrs/blob/master/MIAT-HW4-Loss.png)


#### Reason for Discontinuity:
The resolution of the image increases as a step function once the previous resolutions optimization has converged.  
Because the number of pixels being evaluated by the metric increases suddenly, so does the loss.


### Part 4
In this section we will examine the importance of transform fixed parameters.

#### 4.1

```txt
Result: 
 versor X      = -0.0378825
 versor Y      = 0.0418052
 versor Z      = 0.044935
 Translation X = 17.577
 Translation Y = -266.997
 Translation Z = -2.34762
 Iterations    = 3
 Metric value  = 0.022082
```  

#### 4.2
Comment out the line in your code that set the fixed parameters for the Versor3D transform, re-run the same experiment.

```txt
Result = 
 versor X      = 0
 versor Y      = 0
 versor Z      = 0
 Translation X = 0
 Translation Y = 0
 Translation Z = 0
 Iterations    = 1
 Metric value  = 1.79769e+308
```  

#### Explanation of Difference:
The  reason that 4.2 does not work, is because the images are in different physical spaces. 
The fixed parameters of the transform roughly align the images in physical space.     
It is important for the two images to occupy the same physical space, 
because if they don't, the mean squared error will always be the same(square sum of all intensity values), 
and the gradient will not be meaningful.

### Difference Image:
![difference-image](https://github.com/UIOWAECE5490SP19/hw04-abpwrs/blob/master/difference-image.png)
