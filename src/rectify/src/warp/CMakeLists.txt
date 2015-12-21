INCLUDE_DIRECTORIES(..)

ADD_EXECUTABLE(warp warp.cpp)

IF(UNIX)
    SET_TARGET_PROPERTIES(warp PROPERTIES
                          COMPILE_FLAGS "-std=c++98 -Wall -Wextra -Werror")
ENDIF(UNIX)

TARGET_LINK_LIBRARIES(warp Transform Numerics IO ${IO_LIBRARIES})
