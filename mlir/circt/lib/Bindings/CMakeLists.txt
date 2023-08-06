if(CIRCT_BINDINGS_PYTHON_ENABLED)
  add_subdirectory(Python)
endif()

if(CIRCT_BINDINGS_TCL_ENABLED)
  add_subdirectory(Tcl)
endif()
