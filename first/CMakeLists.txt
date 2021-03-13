cmake_minimum_required(VERSION 2.8.3)
project(Task)

set(CMAKE_BUILD_TYPE "release")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_PREFIX_PATH "/home/sentinel/opencv-4.5.1/")

find_package(OpenCV 4.5.1 REQUIRED)
if (NOT OpenCV_FOUND)
    message("No appropriate opencv package found.")
endif()

find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
   message(FATAL_ERROR "Eigen3 not found.")
endif()

set(OpenCV_INCLUDE_DIRS "/home/sentinel/opencv-4.5.1/include/")
set(${OpenCV_LIBS} "/home/sentinel/opencv-4.5.1/lib/")
set(CUDA_PROJECT TRUE)

if (CUDA_PROJECT)
    find_package(CUDA REQUIRED)
    set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3 --use_fast_math
        -gencode=arch=compute_30,code=sm_30
        -gencode=arch=compute_35,code=sm_35
        -gencode=arch=compute_50,code=sm_50
        -gencode=arch=compute_52,code=sm_52
        -gencode=arch=compute_61,code=sm_61
        -gencode=arch=compute_70,code=sm_70
    )

    aux_source_directory(
        ./src SRC
    )

    cuda_add_executable(Task
        transInterp.cu
        ${SRC}
    )

    cuda_add_executable(Task2
        level8.cu
        ${SRC}
    )

    target_link_libraries(Task2
        ${OpenCV_LIBS}
    )

else()
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()

    link_directories(
        /usr/local/lib
    )

    add_executable(Task homework1.cc
        
    )
endif()

include_directories(
    .
    ./include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    /usr/local/include/
)

link_directories(
    /usr/local/lib
)

target_link_libraries(Task
    ${OpenCV_LIBS}
)