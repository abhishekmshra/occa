if (WITH_HIP)
  compile_cpp_example(21_native_hip_kernels main.cpp)

  add_custom_target(cpp_example_21_addVectors_hip ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.hip addVectors.hip)
  add_dependencies(examples_cpp_21_native_hip_kernels cpp_example_21_addVectors_hip)
endif()
