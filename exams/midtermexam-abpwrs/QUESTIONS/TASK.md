# Free Response

Answer the following questions my modifying this file directly.

##### 1 . Does cmake compile your program?

 - [ ] Yes
 - [x] No

##### 2 . Explain your answer to question #1 by describing the primary utility of CMake (i.e. why does the cmake exist, what common problem is the cmake tool well suited to solve)?

 > CMake is a build environment generator. CMake simplifies the problem of building the same software on different operating systems with different compilers and versions of compilers.

##### 3 . How should the `typedef` or `using` directives be used in your ITK based homework assignments?

 > typedef, and using directives should be used to define the types of different objects. It is not neccessary to use these in order for your code to compile and run, but using typedefs improves the readability and flexibility of the code.

##### 4 . For an ITK image, what are the 3 state variables that are required to define the physcial representation of an image? (NOTE: Regions, Size, Dimension, and Pixel types are incorrect answers.)

 > Origin, Spacing, and Direction Cosine
 
##### 5 . For an `itk::Euler3DTransform<float>` what do the first 3 values of the FixedParameters represent in the transform model?

 > FixedParameter[0] is the euler angle of rotation about the X axis
 
 > FixedParameter[1] is the euler angle of rotation about the Y axis
 
 > FixedParameter[2] is the euler angle of rotation about the Z axis
 
