

set(register_SOURCE register.cxx)
GENERATECLP(register_SOURCE register.xml)


add_executable(register_${MYHAWKID} ${register_SOURCE})
target_link_libraries(register_${MYHAWKID} ${ITK_LIBRARIES})

