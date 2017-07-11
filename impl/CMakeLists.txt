CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

foreach(TARGET
	Kparser
	)
  ADD_EXECUTABLE(${TARGET} ${TARGET}.cc eval.cc pretrained.cc oracle.cc)
  if (WITH_CUDA_BACKEND)
    target_link_libraries(${TARGET} gdynet ${LIBS} boost_iostreams)
    add_dependencies(${TARGET} dynetcuda)
    target_link_libraries(${TARGET} dynetcuda)
    CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
  else()
    target_link_libraries(${TARGET} dynet ${LIBS} boost_iostreams)
  endif (WITH_CUDA_BACKEND)
  if(UNIX AND NOT APPLE)
    target_link_libraries(${TARGET} rt)
  endif()
endforeach()

