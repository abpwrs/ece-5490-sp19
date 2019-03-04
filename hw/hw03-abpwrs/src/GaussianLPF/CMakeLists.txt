
set(GaussianLPF_SOURCE GaussianLPF.cxx)
GENERATECLP(GaussianLPF_SOURCE GaussianLPF.xml)


add_executable(GaussianLPF_${MYHAWKID} ${GaussianLPF_SOURCE})
target_link_libraries(GaussianLPF_${MYHAWKID} ${ITK_LIBRARIES})
