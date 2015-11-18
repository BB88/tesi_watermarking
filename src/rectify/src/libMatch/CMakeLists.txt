SET(SRC imgmatch.cpp imgmatch.h)
ADD_LIBRARY(Match ${SRC})

IF(UNIX)
    SET_TARGET_PROPERTIES(Match PROPERTIES
                          COMPILE_FLAGS "-std=c++98 -Wall -Wextra -Werror")
ENDIF(UNIX)
