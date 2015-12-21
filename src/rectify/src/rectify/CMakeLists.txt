INCLUDE_DIRECTORIES(..)

ADD_LIBRARY(Rectification rectify.cpp rectify.h)
ADD_EXECUTABLE(rectify main.cpp)

IF(UNIX)
    SET_TARGET_PROPERTIES(Rectification PROPERTIES
                          COMPILE_FLAGS "-std=c++98 -Wall -Wextra -Werror")
    SET_TARGET_PROPERTIES(rectify PROPERTIES
                          COMPILE_FLAGS "-std=c++98 -Wall -Wextra -Werror")
ENDIF(UNIX)

TARGET_LINK_LIBRARIES(Rectification Numerics Match)
TARGET_LINK_LIBRARIES(rectify Rectification Numerics Match)
