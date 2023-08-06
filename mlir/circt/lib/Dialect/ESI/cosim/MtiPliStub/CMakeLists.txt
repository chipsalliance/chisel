set(SOURCES ${CAPNP_SRCS} DummySvDpi.cpp )
add_library(MtiPli SHARED ${SOURCES})
target_include_directories(MtiPli PRIVATE ${CIRCT_INCLUDE_DIR})
set_target_properties(MtiPli PROPERTIES CXX_VISIBILITY_PRESET "default")
