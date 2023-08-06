add_mlir_library(MLIRBindingsSupport
    Context.cpp
    Location.cpp
    NativeReference.cpp

    LINK_LIBS PRIVATE
    MLIRIR
    )
