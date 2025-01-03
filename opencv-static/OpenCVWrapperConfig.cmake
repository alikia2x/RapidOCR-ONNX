if (WIN32)
    if (CMAKE_CL_64)
        message("Configured Windows OpenCV x64 path: ${CMAKE_CURRENT_LIST_DIR}/windows-x64")
        set(OpenCV_DIR "${CMAKE_CURRENT_LIST_DIR}/windows-x64")
    else ()
        message("Configured Windows OpenCV x86 path: ${CMAKE_CURRENT_LIST_DIR}/windows-x86")
        set(OpenCV_DIR "${CMAKE_CURRENT_LIST_DIR}/windows-x86")
    endif ()
elseif (APPLE)
    message("Configured macOS OpenCV path: ${CMAKE_CURRENT_LIST_DIR}/macos/lib/cmake/opencv4")
    set(OpenCV_DIR "${CMAKE_CURRENT_LIST_DIR}/macos/lib/cmake/opencv4")
elseif (UNIX)
    message("Configured Linux OpenCV path: ${CMAKE_CURRENT_LIST_DIR}/linux/lib/cmake/opencv4")
    set(OpenCV_DIR "${CMAKE_CURRENT_LIST_DIR}/linux/lib/cmake/opencv4")
endif ()
