include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/Application)
include_directories(${CMAKE_SOURCE_DIR}/Groom)
include_directories(${CMAKE_SOURCE_DIR}/ParticleShapeworks/include)

FILE(GLOB SRCS "*.h" "*.cxx")

ADD_LIBRARY(ShapeWorksOptimize STATIC ${SRCS})
TARGET_LINK_LIBRARIES(ShapeWorksOptimize ShapeWorksGroom 
  ${ITK_LIBRARIES} Optimize tinyxml)
