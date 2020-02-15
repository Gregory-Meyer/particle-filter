cmake_minimum_required(VERSION 3.11)

project(particle-filter VERSION 0.1.0 LANGUAGES C)

set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_library(pf STATIC
    src/correct.c
    src/drop.c
    src/error_description.c
    src/memory.c
    src/new.c
    src/particles_function.c
    src/predict.c
    src/random.c
    src/util.c
)
target_include_directories(pf PUBLIC include)
target_compile_features(pf PUBLIC c_std_11)
target_link_libraries(pf PRIVATE m openblas)

option(PARTICLE_FILTER_BUILD_EXAMPLES "Build examples for particle-filter" OFF)
if(PARTICLE_FILTER_BUILD_EXAMPLES)
    add_executable(odometry
        examples/alloc.c
        examples/l2dm.c
        examples/odometry.c
        examples/se2.c
        examples/trig.c
        examples/v2dp.c
    )
    target_compile_features(odometry PRIVATE c_std_11)
    target_link_libraries(odometry PRIVATE pf pcg_random)
endif()
