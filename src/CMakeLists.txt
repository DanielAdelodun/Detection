# CMakeLists.txt

# Older versions of CMake are likely to work just fine but, since
# I don't know where to cut off I just use the version I'm using
cmake_minimum_required(VERSION "3.17")

# name of this example project
project(Detection)

# set OpenCV_DIR variable equal to the path to the cmake
# files within the previously installed opencv program
# path like /xxx/yyy/opencv/install/lib/cmake/opencv4
set(OpenCV_DIR ${OpenCV_DIR})

# Tell compiler to use C++ 14 features which is needed because
# Clang version is often behind in the XCode installation
set(CMAKE_CXX_STANDARD 14)

# configure the necessary common CMake environment variables
# needed to include and link the OpenCV program into this
# demo project, namely OpenCV_INCLUDE_DIRS and OpenCV_LIBS
find_package( OpenCV 4.8.0 EXACT PATHS /opt/opencv REQUIRED )

# tell the build to include the headers from OpenCV
# path like /xxx/yyy/opencv/install/include
include_directories( ${OpenCV_INCLUDE_DIRS} )

# specify the executable target to be built
add_executable(yolov6 yolov6.cpp)
add_executable(detection servo.cpp detection.cpp)
add_executable(tests tests.cpp)

# tell it to link the executable target against OpenCV
# path like /xxx/yyy/opencv/install/lib
target_link_libraries(yolov6 ${OpenCV_LIBS} )
target_link_libraries(detection ${OpenCV_LIBS} pigpiod_if2)
target_link_libraries(tests ${OpenCV_LIBS} pigpiod_if2)
