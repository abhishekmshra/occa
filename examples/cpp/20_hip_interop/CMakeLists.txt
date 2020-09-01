if (WITH_HIP)
  compile_cpp_example(20_hip_interop main.cpp)

  add_custom_target(cpp_example_20_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
  add_dependencies(examples_cpp_20_hip_interop cpp_example_20_addVectors_okl)
endif()
