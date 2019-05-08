
set(EdgeSlicerModule_SRC EdgeSlicerModule.cxx)
generateclp(EdgeSlicerModule_SRC EdgeSlicerModule.xml)

## Build stand-alone command line module.
add_executable(${HAWKID}EdgeSlicerModule ${EdgeSlicerModule_SRC})
target_link_libraries(${HAWKID}EdgeSlicerModule ${ITK_LIBRARIES})
