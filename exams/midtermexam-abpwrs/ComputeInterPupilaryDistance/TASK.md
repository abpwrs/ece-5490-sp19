For this task, imagine that you must find the distance
between the left and right eye of an MRI image.

Pretend that the mouse can provide the index location
of when you select each eye and the resulting index locations
are stored in variables: LeftEyeIndex and RightEyeIndex respectively.

For the image `/nfsscratch/opt/ece5490/data/Eyes.nrrd` the process of selecting the left and right eyes
results in the following index values:

```cpp
  InputImageType::IndexType LeftEyeIndex;
  LeftEyeIndex[0]=60;
  LeftEyeIndex[1]=127;
  LeftEyeIndex[2]=93;

  InputImageType::IndexType RightEyeIndex;
  RightEyeIndex[0]=59;
  RightEyeIndex[1]=129;
  RightEyeIndex[2]=41;
```

Your task is to compute the interpupilary distance in physical space and
print that distance to the screen.  Assume that the medical image 
`/nfsscratch/opt/ece5490/data/Eyes.nrrd` has physical units of millimeters.
