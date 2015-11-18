INCLUDE_DIRECTORIES(..)

SET(SRC ccmath_svd.cpp
        homography.cpp homography.h
        matrix.cpp matrix.h
        numerics.cpp numerics.h
        vector.cpp)

ADD_LIBRARY(Numerics ${SRC})

IF(UNIX)
    SET_TARGET_PROPERTIES(Numerics PROPERTIES
                          COMPILE_FLAGS "-std=c++98 -Wall -Wextra -Werror")
ENDIF(UNIX)
