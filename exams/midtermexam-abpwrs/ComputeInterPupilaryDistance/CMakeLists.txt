
set(ComputeInterPupilaryDistance_SRC ComputeInterPupilaryDistance.cxx)
generateclp(ComputeInterPupilaryDistance_SRC ComputeInterPupilaryDistance.xml)

## Build stand-alone command line module.
add_executable(${HAWKID}ComputeInterPupilaryDistance ${ComputeInterPupilaryDistance_SRC})
target_link_libraries(${HAWKID}ComputeInterPupilaryDistance ${ITK_LIBRARIES})
