cmake_minimum_required(VERSION 3.22)
project(isaac_ros_object_detection LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ── ROS2 dependencies ─────────────────────────────────────────────────────────
find_package(ament_cmake         REQUIRED)
find_package(rclcpp              REQUIRED)
find_package(rclcpp_components   REQUIRED)
find_package(std_msgs            REQUIRED)
find_package(sensor_msgs         REQUIRED)
find_package(vision_msgs         REQUIRED)
find_package(visualization_msgs  REQUIRED)

# ── Isaac ROS NITROS ──────────────────────────────────────────────────────────
find_package(isaac_ros_managed_nitros          REQUIRED)
find_package(isaac_ros_nitros_tensor_list_type REQUIRED)
find_package(isaac_ros_nitros_image_type       REQUIRED)

# ── OpenCV ────────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# ── CUDA ──────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# nvJPEG (hardware JPEG decode/encode)
find_library(NVJPEG_LIB nvjpeg
  PATHS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)
if(NOT NVJPEG_LIB)
  message(WARNING "nvjpeg library not found; GPU annotation path will be disabled")
endif()

# ── TensorRT ──────────────────────────────────────────────────────────────────
find_path(TRT_INCLUDE_DIR NvInfer.h
  PATHS /usr/include/aarch64-linux-gnu /usr/local/cuda/include)
find_library(TRT_LIB nvinfer
  PATHS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)
find_library(TRT_PLUGIN_LIB nvinfer_plugin
  PATHS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)

if(NOT TRT_LIB)
  message(FATAL_ERROR "TensorRT nvinfer library not found.")
endif()

# ── NvBufSurface (VIC hardware engine) ───────────────────────────────────────
set(NVBUF_INCLUDE_DIR "/usr/src/jetson_multimedia_api/include")
find_library(NVBUFSURFACE_LIB nvbufsurface
  PATHS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu)
find_library(NVBUFSURFTRANSFORM_LIB nvbufsurftransform
  PATHS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

set(ROS_DEPS
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  vision_msgs
  visualization_msgs
  message_filters
  isaac_ros_managed_nitros
  isaac_ros_nitros_tensor_list_type
  isaac_ros_nitros_image_type
)

set(HW_LIBS
  ${TRT_LIB}
  ${TRT_PLUGIN_LIB}
  ${NVBUFSURFACE_LIB}
  ${NVBUFSURFTRANSFORM_LIB}
  CUDA::cudart
  CUDA::cuda_driver
)

if (NVJPEG_LIB)
  list(APPEND HW_LIBS ${NVJPEG_LIB})
endif()

# ═════════════════════════════════════════════════════════════════════════════
# TARGET 1 — Object Detection component library
# ═════════════════════════════════════════════════════════════════════════════

add_library(${PROJECT_NAME} SHARED
  src/object_detection_node.cpp
  src/annotated_image_publisher_node.cpp
  src/cuda_kernels.cu
  src/annotation_kernels.cu
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_ARCHITECTURES    "87"        # Jetson Orin Nano — Ampere sm_87
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${TRT_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${NVBUF_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} ${HW_LIBS} ${OpenCV_LIBRARIES})
ament_target_dependencies(${PROJECT_NAME} ${ROS_DEPS})

if (NVJPEG_LIB)
  target_compile_definitions(${PROJECT_NAME} PUBLIC WITH_NVJPEG=1)
endif()

# Composable node registration (used by production.launch.py)
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN     "nvidia::isaac_ros::object_detection::ObjectDetectionNode"
  EXECUTABLE object_detection_node_component
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN     "nvidia::isaac_ros::object_detection::AnnotatedImagePublisherNode"
  EXECUTABLE annotated_image_publisher_node_component
)

# ─────────────────────────────────────────────────────────────────────────────
# TARGET 2 — Object Detection standalone executable (dev / ros2 run)
# ─────────────────────────────────────────────────────────────────────────────

add_executable(object_detection_node
  src/main.cpp
)

set_target_properties(object_detection_node PROPERTIES
  CUDA_ARCHITECTURES "87"
)

target_include_directories(object_detection_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${TRT_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${NVBUF_INCLUDE_DIR}
)

# Links against the component library — no code duplication
target_link_libraries(object_detection_node
  ${PROJECT_NAME}
  ${HW_LIBS}
)

ament_target_dependencies(object_detection_node ${ROS_DEPS})

add_executable(annotated_image_publisher_node
  src/annotated_image_publisher_main.cpp
)

target_include_directories(annotated_image_publisher_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${CUDAToolkit_INCLUDE_DIRS}
  ${NVBUF_INCLUDE_DIR}
)

target_link_libraries(annotated_image_publisher_node
  ${PROJECT_NAME}
  ${HW_LIBS}
)

ament_target_dependencies(annotated_image_publisher_node ${ROS_DEPS})



# ═════════════════════════════════════════════════════════════════════════════
# Install
# ═════════════════════════════════════════════════════════════════════════════

install(TARGETS
    ${PROJECT_NAME}
    object_detection_node
    object_detection_node_component
    annotated_image_publisher_node
    annotated_image_publisher_node_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()