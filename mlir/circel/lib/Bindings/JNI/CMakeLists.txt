find_package(Java REQUIRED)
find_package(JNI REQUIRED)
include(UseJava)

add_jar(CircelJNI
    NativeObject.java
    GENERATE_NATIVE_HEADERS CircelJNIHeaders)
add_circt_library(CircelJNINative
    SHARED

    NativeObject.cpp
    
    LINK_LIBS PRIVATE
    CIRCTFirtool
    MLIRBindingsSupport
    )
target_link_libraries(CircelJNINative PRIVATE CircelJNIHeaders)
target_include_directories(CircelJNINative PRIVATE ${JNI_INCLUDE_DIRS})
