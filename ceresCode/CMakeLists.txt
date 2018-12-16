cmake_minimum_required(VERSION 2.8)
project(ShapeAndPoseAdjuster)

set(Ceres_DIR "/home/krishna/code/ceres-solver/build/lib")
message("-- Check for Ceres solver")
find_package(Ceres REQUIRED)
if(CERES_FOUND)
	message("-- Found Ceres Solver: ${CERES_INCLUDE_DIRS}")
else(CERES_FOUND)
	message(FATAL_ERROR "Cannot find Ceres Solver. Plese set CERES_INCLUDE_DIRS & CERES_LIBRARY")
endif(CERES_FOUND)

include_directories(${CMAKE_INCLUDE_PATH} ${CERES_INCLUDE_DIRS})

# Add the executable
add_executable(singleViewPoseAdjuster singleViewPoseAdjuster.cc)
target_link_libraries(singleViewPoseAdjuster ${CERES_LIBRARIES})


add_executable(singleViewShapeAdjuster singleViewShapeAdjuster.cc)
target_link_libraries(singleViewShapeAdjuster ${CERES_LIBRARIES})
