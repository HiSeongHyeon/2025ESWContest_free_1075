cmake_minimum_required(VERSION 3.10)
project(glareTracker)

set(CMAKE_CXX_STANDARD 17)

# OpenCV 찾기
find_package(OpenCV REQUIRED)

# OpenMP 활성화
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# include 디렉토리 추가
include_directories(${OpenCV_INCLUDE_DIRS} include)

# 실행 파일 정의
add_executable(glareTracker
    src/main.cpp
    src/glare_detector.cpp
    src/return_position.cpp
)

# 라이브러리 링크
target_link_libraries(glareTracker PUBLIC
    coordinate_mapping_lib
    serial_communication_lib
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX  # OpenMP 라이브러리 추가
)
