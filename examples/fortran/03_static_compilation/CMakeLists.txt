compile_fortran_example_with_modes(03_static_compilation main.f90)

add_custom_target(fortran_example_03_addVectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
add_dependencies(examples_fortran_03_static_compilation fortran_example_03_addVectors_okl)
