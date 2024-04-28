# cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
# project(inference)

# find_package(Torch REQUIRED)

# add_executable(inference inference.cpp)
# target_link_libraries(inference "${TORCH_LIBRARIES}")
# set_property(TARGET inference PROPERTY CXX_STANDARD 14)



cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

add_definitions(-std=c++11 -g)

# set(CXX_FLAGS "-Wall")
# set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(inference)

# packages required
find_package(PCL 1.2 REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(Torch REQUIRED PATHS "../libtorch" )

# directories 
include_directories( ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
# include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} )
ENDMACRO()

# executable
add_executable(inference src/inference.cpp)
target_link_libraries(inference "${TORCH_LIBRARIES}")
target_link_libraries (inference ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(inference PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference 
    -lopencv_core 
    -lopencv_imgcodecs 
    -lopencv_highgui 
)
# set_property(TARGET inference PROPERTY CXX_STANDARD 14)