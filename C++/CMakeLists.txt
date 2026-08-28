cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(WIN32)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  set(FACE_HAS_VCAMERA ON)
else()
  set(FACE_HAS_VCAMERA OFF)
endif()

get_filename_component(FACE_REGISTRATION_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(THIRDPARTY_DIR "${FACE_REGISTRATION_ROOT}/thirdparty" CACHE PATH "Third-party dependency root")

if(WIN32)
  list(PREPEND CMAKE_PREFIX_PATH
    "${THIRDPARTY_DIR}/opencv"
    "${THIRDPARTY_DIR}/open3d"
    "${THIRDPARTY_DIR}/OrbbecSDK"
    "${THIRDPARTY_DIR}/vtk")
  set(OpenCV_DIR "${THIRDPARTY_DIR}/opencv/x64/vc17/lib" CACHE PATH "OpenCV config directory")
  set(Open3D_DIR "${THIRDPARTY_DIR}/open3d/CMake" CACHE PATH "Open3D config directory")
  set(OrbbecSDK_DIR "${THIRDPARTY_DIR}/OrbbecSDK/lib" CACHE PATH "Orbbec config directory")
  set(VTK_DIR "${THIRDPARTY_DIR}/vtk/lib/cmake/vtk-9.4" CACHE PATH "VTK config directory")
  file(GLOB _ort_roots "${THIRDPARTY_DIR}/onnxruntime-win-x64-*")
  list(GET _ort_roots 0 ONNXRUNTIME_ROOT)
else()
  list(PREPEND CMAKE_PREFIX_PATH
    "${THIRDPARTY_DIR}/opencv"
    "${THIRDPARTY_DIR}/open3d"
    "${THIRDPARTY_DIR}/OrbbecSDK"
    "${THIRDPARTY_DIR}/vtk")
  # Prefer the Linux package configs explicitly. A transferred project may
  # also contain an unpacked Windows OpenCV tree at thirdparty/opencv/.
  set(OpenCV_DIR "${THIRDPARTY_DIR}/opencv/lib/cmake/opencv4" CACHE PATH "OpenCV config directory" FORCE)
  set(Open3D_DIR "${THIRDPARTY_DIR}/open3d/lib/cmake/Open3D" CACHE PATH "Open3D config directory" FORCE)
  file(GLOB _vtk_configs "${THIRDPARTY_DIR}/vtk/lib/cmake/vtk-*/vtk-config.cmake")
  if(_vtk_configs)
    list(GET _vtk_configs 0 _vtk_config)
    get_filename_component(VTK_DIR "${_vtk_config}" DIRECTORY)
  endif()
  set(OrbbecSDK_DIR "${THIRDPARTY_DIR}/OrbbecSDK/lib" CACHE PATH "Orbbec config directory" FORCE)
  set(ONNXRUNTIME_ROOT "${THIRDPARTY_DIR}/onnxruntime")
endif()
set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime root")

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs calib3d dnn highgui)
find_package(Open3D CONFIG REQUIRED)
find_package(OrbbecSDK CONFIG REQUIRED)
find_package(VTK 9.4 CONFIG REQUIRED COMPONENTS
  CommonCore
  CommonDataModel
  FiltersCore
  IOGeometry
  RenderingCore
  RenderingOpenGL2)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS "${ONNXRUNTIME_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)

set(FACE_CAMERA_SOURCES
  cpp/camera/CameraTypes.cpp
  cpp/camera/CameraBase.cpp
  cpp/camera/OrbbecCamera.cpp
  cpp/camera/OrbbecDepthAlignedCamera.cpp)
if(FACE_HAS_VCAMERA)
  list(APPEND FACE_CAMERA_SOURCES cpp/camera/VcameraDepthAlignedCamera.cpp)
endif()
add_library(face_camera STATIC ${FACE_CAMERA_SOURCES})
target_include_directories(face_camera PUBLIC hpp)
target_link_libraries(face_camera PUBLIC ${OpenCV_LIBS} ob::OrbbecSDK)
if(FACE_HAS_VCAMERA)
  target_include_directories(face_camera PUBLIC "${THIRDPARTY_DIR}/VcameraSDK/cpp/include")
  target_compile_definitions(face_camera PUBLIC FACE_HAS_VCAMERA=1)
  target_link_libraries(face_camera PUBLIC
    "$<$<CONFIG:Debug>:${THIRDPARTY_DIR}/VcameraSDK/cpp/win/Debug/lib/ucvcameraapi.lib>"
    "$<$<NOT:$<CONFIG:Debug>>:${THIRDPARTY_DIR}/VcameraSDK/cpp/win/Release/lib/ucvcameraapi.lib>")
else()
  target_compile_definitions(face_camera PUBLIC FACE_HAS_VCAMERA=0)
endif()

add_library(face_detection STATIC
  cpp/detection/FaceDetector.cpp
  cpp/detection/FaceDetectionService.cpp
  cpp/detection/FaceKeypointService.cpp)
target_include_directories(face_detection PUBLIC hpp "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(face_detection PUBLIC face_camera ${OpenCV_LIBS} "${ONNXRUNTIME_LIBRARY}")

add_library(face_segmentation STATIC cpp/segmentation/Sapiens2Segmenter.cpp)
target_include_directories(face_segmentation PUBLIC hpp "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(face_segmentation PUBLIC ${OpenCV_LIBS} "${ONNXRUNTIME_LIBRARY}")

add_library(face_pose STATIC cpp/pose/Sapiens2PoseEstimator.cpp)
target_include_directories(face_pose PUBLIC hpp "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(face_pose PUBLIC ${OpenCV_LIBS} "${ONNXRUNTIME_LIBRARY}")

add_library(face_registration STATIC
  cpp/registration/CameraFaceKeypointExtractionService.cpp
  cpp/registration/FaceCandidateSelectionPolicy.cpp
  cpp/registration/FacePointCloudCropService.cpp
  cpp/registration/HeadSurfaceCache.cpp
  cpp/registration/HeadSurfaceReconstructor.cpp
  cpp/registration/ModelKeypointAnnotationService.cpp
  cpp/registration/StlModelRenderer.cpp
  cpp/registration/PointCloudRegistration.cpp)
target_include_directories(face_registration PUBLIC hpp)
set(FACE_VTK_MODULES
  VTK::CommonCore
  VTK::CommonDataModel
  VTK::FiltersCore
  VTK::IOGeometry
  VTK::RenderingCore
  VTK::RenderingOpenGL2)
target_link_libraries(face_registration PUBLIC face_detection Open3D::Open3D ${FACE_VTK_MODULES})
vtk_module_autoinit(TARGETS face_registration MODULES ${FACE_VTK_MODULES})

add_executable(face_registration_cli app/RegistrationCli.cpp)
target_link_libraries(face_registration_cli PRIVATE face_registration)

add_executable(face_camera_pipeline app/CameraPipeline.cpp)
target_link_libraries(face_camera_pipeline PRIVATE face_registration face_segmentation face_pose)

add_executable(manual_roi app/ManualRoi.cpp)
target_link_libraries(manual_roi PRIVATE face_camera)

add_executable(sapiens_seg app/OrbbecSapiensSeg.cpp)
target_link_libraries(sapiens_seg PRIVATE face_camera face_segmentation face_detection)

add_executable(sapiens_pose app/SapiensPose.cpp)
target_link_libraries(sapiens_pose PRIVATE face_pose)

if(WIN32)
  file(GLOB FACE_RUNTIME_DLLS
    "${THIRDPARTY_DIR}/opencv/x64/vc17/bin/*.dll"
    "${THIRDPARTY_DIR}/OrbbecSDK/bin/*.dll"
    "${THIRDPARTY_DIR}/open3d/bin/*.dll"
    "${ONNXRUNTIME_ROOT}/lib/*.dll")
  foreach(FACE_EXECUTABLE
      face_registration_cli
      face_camera_pipeline
      manual_roi
      sapiens_seg
      sapiens_pose)
    add_custom_command(TARGET ${FACE_EXECUTABLE} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${FACE_RUNTIME_DLLS}
        $<TARGET_FILE_DIR:${FACE_EXECUTABLE}>
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:${FACE_EXECUTABLE}>
        $<TARGET_FILE_DIR:${FACE_EXECUTABLE}>
      COMMAND_EXPAND_LISTS)
  endforeach()

  add_custom_command(TARGET sapiens_seg POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${THIRDPARTY_DIR}/OrbbecSDK/bin/extensions"
      "$<TARGET_FILE_DIR:sapiens_seg>/extensions")

  add_custom_command(TARGET face_camera_pipeline POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${THIRDPARTY_DIR}/OrbbecSDK/bin/extensions"
      "$<TARGET_FILE_DIR:face_camera_pipeline>/extensions"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${THIRDPARTY_DIR}/VcameraSDK/cpp/win/$<CONFIG>/bin/feature_config.xml"
      "${THIRDPARTY_DIR}/VcameraSDK/cpp/win/$<CONFIG>/bin/FeatureMap.xml"
      "$<TARGET_FILE_DIR:face_camera_pipeline>"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${THIRDPARTY_DIR}/VcameraSDK/cpp/win/$<CONFIG>/bin"
      "$<TARGET_FILE_DIR:face_camera_pipeline>"
    COMMAND_EXPAND_LISTS)
endif()

install(TARGETS face_camera face_detection face_segmentation face_pose face_registration
  face_registration_cli face_camera_pipeline manual_roi sapiens_seg sapiens_pose)
install(DIRECTORY hpp/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/face_registration/config)
