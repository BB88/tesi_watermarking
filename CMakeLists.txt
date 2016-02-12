cmake_minimum_required(VERSION 3.1)
project(tesi_watermarking)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

#set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


IF(UNIX)
    SET(BUILD_SHARED_LIBS OFF CACHE BOOL "Dynamic libraries")
ENDIF(UNIX)

SET(BASE_DIR_SRC ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "Base src dir")
SET(BASE_DIR_BIN ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "Base bin dir")

# Use the local libraries and do not check the system libaries
SET(WITH_INCLUDED_ZLIB OFF CACHE BOOL "Use local zlib")
SET(WITH_INCLUDED_LIBPNG OFF CACHE BOOL "Use local libpng")
SET(WITH_INCLUDED_LIBJPEG OFF CACHE BOOL "Use local libjpeg")
SET(WITH_INCLUDED_LIBTIFF OFF CACHE BOOL "Use local libtiff")

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)


#ADD_SUBDIRECTORY(./src/rectify)


set(SOURCE_FILES
    src/main.cpp
    src/dataset/dataset.hpp
    src/dataset/msm_middlebury.hpp
    src/dataset/tsukuba_dataset.h
    src/dataset/util.hpp
    src/logger/log.h
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
    src/spatialWatermarking/gaussianNoise.h
    src/spatialWatermarking/gaussianNoise.cpp
    src/DFTwatermarking/frequencyWatermarking.h
    src/DFTwatermarking/frequencyWatermarking.cpp

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

set(SRC_QM
    src/quality_metrics/quality_metrics.cpp
    src/quality_metrics/quality_metrics.h
  )

set(SRC_C
    src/graphcuts/io_tiff.c
    src/graphcuts/io_tiff.h
    src/graphcuts/io_png.c
    src/graphcuts/io_png.h)

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
set(SOURCE_FILES ${SOURCE_FILES} ${SRC_QM})

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

message(${LIBS})

add_executable(tesi_watermarking ${SOURCE_FILES} src/dataset/tsukuba_dataset.cpp src/dataset/tsukuba_dataset.h  src/right_view_computation/right_view.cpp src/right_view_computation/right_view.h src/disparity_optimization/disp_opt.cpp src/disparity_optimization/disp_opt.h src/img_watermarking/bch.h src/img_watermarking/bch.cpp src/img_watermarking/fft2d.h src/img_watermarking/fft2d.cpp src/img_watermarking/allocim.h src/img_watermarking/allocim.cpp src/img_watermarking/watermarking.h src/img_watermarking/watermarking.cpp src/graphcuts/utils.h src/graphcuts/utils.cpp src/utils.h src/utils.cpp src/quality_metrics/quality_metrics.h src/quality_metrics/quality_metrics.cpp  src/spatialWatermarking/gaussianNoise.h src/spatialWatermarking/gaussianNoise.cpp src/DFTwatermarking/frequencyWatermarking.h src/DFTwatermarking/frequencyWatermarking.cpp src/quality_metrics/RRQualityMetrics.h src/quality_metrics/RRQualityMetrics.cpp src/roc/roc.h src/roc/roc.cpp)

target_link_libraries(tesi_watermarking ${LIBS})


