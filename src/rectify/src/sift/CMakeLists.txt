SET(SRC library.cpp library.h
        domain.cpp domain.h
        splines.cpp splines.h
        filter.cpp filter.h
        demo_lib_sift.cpp demo_lib_sift.h)

ADD_LIBRARY(SIFT ${SRC})
ADD_EXECUTABLE(siftMatch demo_sift.cpp)

INCLUDE_DIRECTORIES(..)

IF(UNIX)
    SET_TARGET_PROPERTIES(SIFT PROPERTIES
                          COMPILE_FLAGS "-std=c++98 -Wall -Wextra -Werror")
    SET_TARGET_PROPERTIES(siftMatch PROPERTIES
                          COMPILE_FLAGS "-std=c++98 -Wall -Wextra -Werror")
ENDIF(UNIX)

TARGET_LINK_LIBRARIES(SIFT Numerics Match)
TARGET_LINK_LIBRARIES(siftMatch SIFT Numerics Match IO ${IO_LIBRARIES})
