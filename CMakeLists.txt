cmake_minimum_required(VERSION 3.10)
project(aruco_robot_navigation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

add_executable(aruco_navigation src/main.cpp)
target_link_libraries(aruco_navigation PRIVATE ${OpenCV_LIBS})
target_include_directories(aruco_navigation PRIVATE ${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs:    ${OpenCV_LIBS}")
