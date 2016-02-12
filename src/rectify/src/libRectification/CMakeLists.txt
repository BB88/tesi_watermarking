INCLUDE_DIRECTORIES(..)

ADD_LIBRARY(Rectification rectify.cpp rectify.h)
ADD_EXECUTABLE(libRectification main.cpp)

IF(UNIX)
    SET_TARGET_PROPERTIES(Rectification PROPERTIES
                          COMPILE_FLAGS "-std=c++98 -Wall -Wextra -Werror")
    SET_TARGET_PROPERTIES(libRectification PROPERTIES
                          COMPILE_FLAGS "-std=c++98 -Wall -Wextra -Werror")
ENDIF(UNIX)

TARGET_LINK_LIBRARIES(Rectification Numerics Match)
TARGET_LINK_LIBRARIES(libRectification Rectification Numerics Match)
