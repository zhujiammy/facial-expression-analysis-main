cmake_minimum_required(VERSION 3.16)
project(FacialExpressionAnalysis)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 启用调试信息
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
# 设置编译选项
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /utf-8")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /DEBUG /utf-8")
    # 添加UTF-8支持
    add_compile_options(/utf-8)
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

# 添加vcpkg工具链文件（如果使用vcpkg）
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 查找必需的包
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找dlib
find_package(dlib QUIET)
if(NOT dlib_FOUND)
    message(STATUS "dlib not found via find_package, trying pkg-config...")
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(DLIB dlib-1)
        if(DLIB_FOUND)
            include_directories(${DLIB_INCLUDE_DIRS})
            link_directories(${DLIB_LIBRARY_DIRS})
        endif()
    endif()
endif()

# ONNX Runtime - 尝试多个可能的路径
set(ONNXRUNTIME_PATHS
    "C:/Program Files/onnxruntime"
    "C:/onnxruntime"
    "/usr/local"
    "/opt/onnxruntime"
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime"
)

foreach(ONNX_PATH ${ONNXRUNTIME_PATHS})
    if(EXISTS ${ONNX_PATH}/include/onnxruntime_cxx_api.h)
        set(ONNXRUNTIME_ROOT_PATH ${ONNX_PATH})
        break()
    endif()
endforeach()

if(ONNXRUNTIME_ROOT_PATH)
    message(STATUS "Found ONNX Runtime at: ${ONNXRUNTIME_ROOT_PATH}")
    include_directories(${ONNXRUNTIME_ROOT_PATH}/include)
    if(WIN32)
        link_directories(${ONNXRUNTIME_ROOT_PATH}/lib)
    else()
        link_directories(${ONNXRUNTIME_ROOT_PATH}/lib)
    endif()
    set(ONNX_AVAILABLE TRUE)
else()
    message(WARNING "ONNX Runtime not found. Please install it or set ONNXRUNTIME_ROOT_PATH")
    set(ONNX_AVAILABLE FALSE)
endif()

# cnpy - 尝试通过vcpkg或系统路径查找
find_path(CNPY_INCLUDE_DIR cnpy.h
    PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cnpy
        /usr/local/include
        /usr/include
        C:/vcpkg/installed/x64-windows/include
        ${VCPKG_INSTALLED_DIR}/x64-windows/include
        "C:/Program Files (x86)/CNPY/include"
        "C:/Program Files (x86)/CNPY"
    DOC "Path to cnpy include directory"
)

find_library(CNPY_LIBRARY
    NAMES cnpy libcnpy
    PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cnpy/build
        /usr/local/lib
        /usr/lib
        C:/vcpkg/installed/x64-windows/lib
        ${VCPKG_INSTALLED_DIR}/x64-windows/lib
        "C:/Program Files (x86)/CNPY/lib"
        "C:/Program Files (x86)/CNPY/build"
        "C:/Program Files (x86)/CNPY/build/Release"
    DOC "Path to cnpy library"
)

if(CNPY_INCLUDE_DIR AND CNPY_LIBRARY)
    message(STATUS "Found cnpy: ${CNPY_LIBRARY}")
    include_directories(${CNPY_INCLUDE_DIR})
    set(CNPY_AVAILABLE TRUE)
else()
    message(WARNING "cnpy not found. Please install it via vcpkg or manually")
    set(CNPY_AVAILABLE FALSE)
endif()

# 通用源文件（不包含main.cpp）
set(COMMON_SOURCES
    src/emotion_analyzer.cpp
    src/facial_landmarks.cpp
    src/model_comparison.cpp
    src/utils.cpp
)

# DLL专用源文件
set(DLL_SOURCES
    ${COMMON_SOURCES}
    src/facial_expression_dll.cpp
)

# 可执行文件源文件
set(EXE_SOURCES
    ${COMMON_SOURCES}
    src/main.cpp
)

# 头文件
set(HEADERS
    include/emotion_analyzer.h
    include/facial_landmarks.h
    include/model_comparison.h
    include/utils.h
    include/facial_expression_dll.h
)

# 创建DLL
add_library(${PROJECT_NAME}DLL SHARED ${DLL_SOURCES} ${HEADERS})

# 创建可执行文件
add_executable(${PROJECT_NAME} ${EXE_SOURCES} ${HEADERS})

# 创建DLL测试程序（直接调用DLL，不链接源文件）
add_executable(${PROJECT_NAME}DLLDirectTest src/test_dll_direct.cpp)

# DLL设置
target_compile_definitions(${PROJECT_NAME}DLL PRIVATE FACIAL_EXPRESSION_EXPORTS)
set_target_properties(${PROJECT_NAME}DLL PROPERTIES
    OUTPUT_NAME "FacialExpressionDLL"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 基本链接库 - DLL
target_link_libraries(${PROJECT_NAME}DLL ${OpenCV_LIBS})

# 基本链接库 - EXE
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# 条件链接其他库 - DLL
if(dlib_FOUND)
    target_link_libraries(${PROJECT_NAME}DLL dlib::dlib)
    target_compile_definitions(${PROJECT_NAME}DLL PRIVATE DLIB_AVAILABLE)
elseif(DLIB_FOUND)
    target_link_libraries(${PROJECT_NAME}DLL ${DLIB_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME}DLL PRIVATE DLIB_AVAILABLE)
else()
    message(WARNING "dlib not found - face detection will not work")
endif()

if(ONNX_AVAILABLE)
    if(WIN32)
        target_link_libraries(${PROJECT_NAME}DLL onnxruntime)
    else()
        target_link_libraries(${PROJECT_NAME}DLL onnxruntime)
    endif()
    target_compile_definitions(${PROJECT_NAME}DLL PRIVATE ONNX_AVAILABLE)
else()
    message(WARNING "ONNX Runtime not available - model inference will not work")
endif()

if(CNPY_AVAILABLE)
    target_link_libraries(${PROJECT_NAME}DLL ${CNPY_LIBRARY})
    if(UNIX)
        target_link_libraries(${PROJECT_NAME}DLL z)  # zlib for compression
    endif()
    target_compile_definitions(${PROJECT_NAME}DLL PRIVATE CNPY_AVAILABLE)
else()
    message(WARNING "cnpy not available - .npy file loading will not work")
endif()

# 条件链接其他库 - EXE
if(dlib_FOUND)
    target_link_libraries(${PROJECT_NAME} dlib::dlib)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DLIB_AVAILABLE)
elseif(DLIB_FOUND)
    target_link_libraries(${PROJECT_NAME} ${DLIB_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE DLIB_AVAILABLE)
else()
    message(WARNING "dlib not found - face detection will not work")
endif()

if(ONNX_AVAILABLE)
    if(WIN32)
        target_link_libraries(${PROJECT_NAME} onnxruntime)
    else()
        target_link_libraries(${PROJECT_NAME} onnxruntime)
    endif()
    target_compile_definitions(${PROJECT_NAME} PRIVATE ONNX_AVAILABLE)
else()
    message(WARNING "ONNX Runtime not available - model inference will not work")
endif()

if(CNPY_AVAILABLE)
    target_link_libraries(${PROJECT_NAME} ${CNPY_LIBRARY})
    if(UNIX)
        target_link_libraries(${PROJECT_NAME} z)  # zlib for compression
    endif()
    target_compile_definitions(${PROJECT_NAME} PRIVATE CNPY_AVAILABLE)
else()
    message(WARNING "cnpy not available - .npy file loading will not work")
endif()

# DLL直接测试程序设置（仅链接基础库）
set_target_properties(${PROJECT_NAME}DLLDirectTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
)

# 设置输出目录
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
)

# Windows特定设置
if(WIN32)
    # 设置Windows子系统为控制台
    set_target_properties(${PROJECT_NAME} PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
      # 复制必要的DLL文件（如果需要）- EXE
    if(ONNX_AVAILABLE AND EXISTS ${ONNXRUNTIME_ROOT_PATH}/lib/onnxruntime.dll)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_ROOT_PATH}/lib/onnxruntime.dll
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    endif()
    
    # 复制必要的DLL文件（如果需要）- DLL
    if(ONNX_AVAILABLE AND EXISTS ${ONNXRUNTIME_ROOT_PATH}/lib/onnxruntime.dll)
        add_custom_command(TARGET ${PROJECT_NAME}DLL POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_ROOT_PATH}/lib/onnxruntime.dll
            $<TARGET_FILE_DIR:${PROJECT_NAME}DLL>)
    endif()
endif()

# 安装规则
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}DLL 
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

# 尝试复制模型文件到输出目录（如果存在）
set(MODEL_FILES
    "../models/model_emotion_pls30.onnx"
    "../models/model_frontalization.npy"
    "../models/shape_predictor_68_face_landmarks.dat"
)

foreach(MODEL_FILE ${MODEL_FILES})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${MODEL_FILE})
        get_filename_component(MODEL_NAME ${MODEL_FILE} NAME)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${MODEL_FILE} 
                      ${CMAKE_BINARY_DIR}/bin/${MODEL_NAME} COPYONLY)
        message(STATUS "Copied model file: ${MODEL_NAME}")
    else()
        message(STATUS "Model file not found: ${MODEL_FILE}")
    endif()
endforeach()

# 显示配置摘要
message(STATUS "=== Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "dlib: ${dlib_FOUND}")
message(STATUS "ONNX Runtime: ${ONNX_AVAILABLE}")
message(STATUS "cnpy: ${CNPY_AVAILABLE}")
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "=============================")

# 警告和建议
if(NOT ONNX_AVAILABLE OR NOT CNPY_AVAILABLE OR NOT dlib_FOUND)
    message(STATUS "")
    message(STATUS "⚠️  Warning: Some dependencies are missing!")
    message(STATUS "To install missing dependencies on Windows with vcpkg:")
    message(STATUS "  vcpkg install opencv[contrib]:x64-windows")
    message(STATUS "  vcpkg install dlib:x64-windows")
    message(STATUS "  vcpkg install cnpy:x64-windows")
    message(STATUS "")
    message(STATUS "For ONNX Runtime, download from:")
    message(STATUS "  https://github.com/microsoft/onnxruntime/releases")
    message(STATUS "")
endif()
