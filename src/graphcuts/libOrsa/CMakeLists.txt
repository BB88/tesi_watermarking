INCLUDE_DIRECTORIES(..)

SET(SRC conditioning.hpp conditioning.cpp
        orsa_model.hpp orsa_model.cpp
        fundamental_model.hpp fundamental_model.cpp)

ADD_LIBRARY(Orsa ${SRC})

IF(UNIX)
    SET_TARGET_PROPERTIES(Orsa PROPERTIES
                          COMPILE_FLAGS "-std=c++98 -Wall -Wextra -Werror")
ENDIF(UNIX)

TARGET_LINK_LIBRARIES(Orsa Numerics)
