
set(DOG_SOURCE DOG.cxx)
GENERATECLP(DOG_SOURCE DOG.xml)


add_executable(DOG_${MYHAWKID} ${DOG_SOURCE})
target_link_libraries(DOG_${MYHAWKID} ${ITK_LIBRARIES})
