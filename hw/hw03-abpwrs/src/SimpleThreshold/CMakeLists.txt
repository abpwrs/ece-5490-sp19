
set(SimpleThreshold_SOURCE SimpleThreshold.cxx)
GENERATECLP(SimpleThreshold_SOURCE SimpleThreshold.xml)


add_executable(SimpleThreshold_${MYHAWKID} ${SimpleThreshold_SOURCE})
target_link_libraries(SimpleThreshold_${MYHAWKID} ${ITK_LIBRARIES})
