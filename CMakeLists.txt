cmake_minimum_required(VERSION 3.1)
project(tesi_watermarking)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    src/main.cpp
    src/dataset/dataset.hpp
    src/dataset/msm_middlebury.hpp
    src/dataset/tsukuba_dataset.h
    src/dataset/util.hpp
    src/disparity_computation/stereo_matching.h
    src/logger/log.h
    src/disparity_computation/stereo_matching.cpp
    src/dataset/dataset.cpp
    src/dataset/msm_middlebury.cpp
    src/dataset/tsukuba_dataset.cpp
    src/dataset/util.cpp

)




#################################### Locate OpenCV ############################################

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
message(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBS})

add_executable(tesi_watermarking ${SOURCE_FILES} src/disparity_computation/stereo_matching.h src/disparity_computation/stereo_matching.h src/disparity_computation/stereo_matching.cpp src/dataset/tsukuba_dataset.cpp src/dataset/tsukuba_dataset.h src/disparity_computation/sift_computation.h src/disparity_computation/sift_computation.cpp src/disparity_optimization/occlusions_handler.h src/disparity_optimization/occlusions_handler.cpp)
target_link_libraries(tesi_watermarking ${LIBS})