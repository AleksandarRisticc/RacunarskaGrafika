cmake_minimum_required(VERSION 3.20)
project(RGProject LANGUAGES CXX)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.5")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# openGL
if (WIN32)
    set(OPENGL_gl_LIBRARY opengl32)
else()
    find_package(OpenGL REQUIRED)
endif()

include(FetchContent)

# glfw
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.3.9
)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# glew
FetchContent_Declare(
        glew
        GIT_REPOSITORY https://github.com/Perlmint/glew-cmake.git
        GIT_TAG        master
)
FetchContent_MakeAvailable(glew)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_definitions(-DGLEW_STATIC)

# glm
FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.1
)
FetchContent_MakeAvailable(glm)

# stb_image
include_directories(${CMAKE_SOURCE_DIR}/external)

# app-
add_executable(app
        src/Main.cpp
)

# za shadere
target_compile_definitions(app PRIVATE SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders")

target_link_libraries(app PRIVATE glfw libglew_static)
if (WIN32)
    target_link_libraries(app PRIVATE ${OPENGL_gl_LIBRARY} winmm)
else()
    target_link_libraries(app PRIVATE OpenGL::GL)
endif()
target_include_directories(app PRIVATE
        ${glfw_SOURCE_DIR}/include
        ${glm_SOURCE_DIR}
        ${glew_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/external
)
add_custom_target(copy_shaders ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/shaders"
        "${CMAKE_BINARY_DIR}/shaders")
add_dependencies(app copy_shaders)

# --- Copy audio (.wav) pored exe-a ---
add_custom_target(copy_audio ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/audio"
        "${CMAKE_BINARY_DIR}"
)
add_dependencies(app copy_audio)


if (APPLE)
    target_compile_definitions(app PRIVATE GL_SILENCE_DEPRECATION)
endif()
