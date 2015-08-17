cmake_minimum_required(VERSION 3.1)
project(tesi_watermarking)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


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
    src/img_watermarking/fft2d.cpp
    src/img_watermarking/watermarking.cpp
    src/img_watermarking/allocim.cpp
    src/img_watermarking/bch.cpp
    src/img_watermarking/fft2d.h
    src/img_watermarking/watermarking.h
    src/img_watermarking/bch.h
    src/img_watermarking/allocim.h
    src/config/config.cpp
    src/config/config.hpp
)
   ###graph cuts classes###

set(SRC_C
    src/graphcuts/io_tiff.c
    src/graphcuts/io_tiff.h
    src/graphcuts/io_png.c
    src/graphcuts/io_png.h)

set(SRC
    src/graphcuts/cmdLine.h
    src/graphcuts/data.cpp
    src/graphcuts/image.cpp
    src/graphcuts/image.h
    src/graphcuts/kz2.cpp
    src/graphcuts/match.cpp
    src/graphcuts/match.h
    src/graphcuts/nan.h
    src/graphcuts/statistics.cpp
    src/graphcuts/utils.cpp
    src/graphcuts/utils.h
    )

set(SRC_ENERGY
    src/graphcuts/energy/energy.h)

set(SRC_MAXFLOW
    src/graphcuts/maxflow/graph.cpp
    src/graphcuts/maxflow/graph.h
    src/graphcuts/maxflow/maxflow.cpp)


 FIND_PACKAGE(PNG)
 FIND_PACKAGE(TIFF)

 IF(NOT PNG_FOUND)
     FIND_PACKAGE(ZLIB)
     IF(NOT ZLIB_FOUND)
         ADD_SUBDIRECTORY(src/graphcuts/third_party/zlib-1.2.8)
     ENDIF(NOT ZLIB_FOUND)
     INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIRS})
     ADD_SUBDIRECTORY(src/graphcuts/third_party/libpng-1.6.12)
     INCLUDE_DIRECTORIES(${PNG_INCLUDE_DIRS})
 ENDIF(NOT PNG_FOUND)

 IF(NOT TIFF_FOUND)
     FIND_PACKAGE(JPEG)
     IF(NOT JPEG_FOUND)
         ADD_SUBDIRECTORY(src/graphcuts/third_party/jpeg-9a)
     ENDIF(NOT JPEG_FOUND)
     INCLUDE_DIRECTORIES(${JPEG_INCLUDE_DIR})
     ADD_SUBDIRECTORY(src/graphcuts/third_party/tiff-4.0.3)
     INCLUDE_DIRECTORIES(${TIFF_INCLUDE_DIR})
 ENDIF(NOT TIFF_FOUND)

 ADD_DEFINITIONS(${PNG_DEFINITIONS} -DHAS_PNG)
 ADD_DEFINITIONS(${TIFF_DEFINITIONS} -DHAS_TIFF)


set(SOURCE_FILES ${SOURCE_FILES} ${SRC_ENERGY})
set(SOURCE_FILES ${SOURCE_FILES} ${SRC_MAXFLOW})
set(SOURCE_FILES ${SOURCE_FILES} ${SRC_C})
set(SOURCE_FILES ${SOURCE_FILES} ${SRC})

set(LIBS ${LIBS} ${TIFF_LIBRARIES})
set(LIBS ${LIBS} ${PNG_LIBRARIES})

#################################### libconfig  ############################################

#INCLUDE(${CMAKE_SOURCE_DIR}/cmake/FindLibConfig.cmake)
find_package(LibConfig REQUIRED)
#
#link_directories(${LIBCONFIG_LIBRARIES})
include_directories(${LIBCONFIG_INCLUDE_DIR})
set(LIBS ${LIBS} ${LIBCONFIGPP_LIBRARY})


#################################### Locate OpenCV ############################################

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
message(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBS})

add_executable(tesi_watermarking ${SOURCE_FILES} src/disparity_computation/stereo_matching.h src/disparity_computation/stereo_matching.h src/disparity_computation/stereo_matching.cpp src/dataset/tsukuba_dataset.cpp src/dataset/tsukuba_dataset.h src/disparity_computation/sift_computation.h src/disparity_computation/sift_computation.cpp src/disparity_optimization/occlusions_handler.h src/disparity_optimization/occlusions_handler.cpp src/right_view_computation/right_view.cpp src/right_view_computation/right_view.h src/disparity_optimization/disp_opt.cpp src/disparity_optimization/disp_opt.h src/img_watermarking/bch.h src/img_watermarking/bch.cpp src/img_watermarking/fft2d.h src/img_watermarking/fft2d.cpp src/img_watermarking/imgwat.h src/img_watermarking/imgwat.cpp src/img_watermarking/allocim.h src/img_watermarking/allocim.cpp src/img_watermarking/watermarking.h src/img_watermarking/watermarking.cpp src/graphcuts/utils.h src/graphcuts/utils.cpp src/utils.h src/utils.cpp)
target_link_libraries(tesi_watermarking ${LIBS})

