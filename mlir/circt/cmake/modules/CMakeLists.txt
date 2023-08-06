# Generate a list of CMake library targets so that other CMake projects can
# link against them.
set(CIRCT_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/circt)
set(circt_cmake_builddir "${CMAKE_BINARY_DIR}/${CIRCT_INSTALL_PACKAGE_DIR}")

# Keep this in sync with mlir/cmake/CMakeLists.txt!
set(MLIR_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir)
set(mlir_cmake_builddir "${MLIR_BINARY_DIR}/${MLIR_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
get_property(CIRCT_EXPORTS GLOBAL PROPERTY CIRCT_EXPORTS)
export(TARGETS ${CIRCT_EXPORTS} FILE ${circt_cmake_builddir}/CIRCTTargets.cmake)

get_property(CIRCT_ALL_LIBS GLOBAL PROPERTY CIRCT_ALL_LIBS)
get_property(CIRCT_DIALECT_LIBS GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(CIRCT_CONVERSION_LIBS GLOBAL PROPERTY CIRCT_CONVERSION_LIBS)
get_property(CIRCT_TRANSLATION_LIBS GLOBAL PROPERTY CIRCT_TRANSLATION_LIBS)
get_property(CIRCT_ANALYSIS_LIBS GLOBAL PROPERTY CIRCT_ANALYSIS_LIBS)
get_property(CIRCT_VERIFICATION_LIBS GLOBAL PROPERTY CIRCT_VERIFICATION_LIBS)

# Generate CIRCTConfig.cmake for the build tree.
set(CIRCT_CONFIG_CMAKE_DIR "${circt_cmake_builddir}")
set(CIRCT_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(CIRCT_CONFIG_MLIR_CMAKE_DIR "${mlir_cmake_builddir}")
set(CIRCT_CONFIG_LIBRARY_DIRS "${CIRCT_LIBRARY_DIR}")
set(CIRCT_CONFIG_BINARY_DIR "${CIRCT_BINARY_DIR}")
set(CIRCT_CONFIG_TOOLS_DIR "${CIRCT_TOOLS_DIR}")
set(CIRCT_CONFIG_INCLUDE_EXPORTS "include(\"\${CIRCT_CMAKE_DIR}/CIRCTTargets.cmake\")")
set(CIRCT_CONFIG_INCLUDE_DIRS
  "${CIRCT_SOURCE_DIR}/include"
  "${CIRCT_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/CIRCTConfig.cmake.in
  ${circt_cmake_builddir}/CIRCTConfig.cmake
  @ONLY)
set(CIRCT_CONFIG_CMAKE_DIR)
set(CIRCT_CONFIG_LLVM_CMAKE_DIR)
set(CIRCT_CONFIG_MLIR_CMAKE_DIR)
set(CIRCT_CONFIG_LIBRARY_DIRS)
set(CIRCT_CONFIG_BINARY_DIR)
set(CIRCT_CONFIG_TOOLS_DIR)
set(CIRCT_CONFIG_INCLUDE_EXPORTS)
set(CIRCT_CONFIG_INCLUDE_DIRS)

# Generate CIRCTConfig.cmake for the install tree.
set(CIRCT_CONFIG_CODE "
# Compute the installation prefix from this CIRCTConfig.cmake file location.
get_filename_component(CIRCT_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${CIRCT_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(CIRCT_CONFIG_CODE "${CIRCT_CONFIG_CODE}
get_filename_component(CIRCT_INSTALL_PREFIX \"\${CIRCT_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(CIRCT_CONFIG_CMAKE_DIR "\${CIRCT_INSTALL_PREFIX}/${CIRCT_INSTALL_PACKAGE_DIR}")
set(CIRCT_CONFIG_LLVM_CMAKE_DIR "\${CIRCT_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
set(CIRCT_CONFIG_MLIR_CMAKE_DIR "\${CIRCT_INSTALL_PREFIX}/${MLIR_INSTALL_PACKAGE_DIR}")
set(CIRCT_CONFIG_LIBRARY_DIRS "\${CIRCT_INSTALL_PREFIX}/lib")
set(CIRCT_CONFIG_BINARY_DIR "\${CIRCT_INSTALL_PREFIX}")
set(CIRCT_CONFIG_TOOLS_DIR "\${CIRCT_INSTALL_PREFIX}/bin")
set(CIRCT_CONFIG_INCLUDE_EXPORTS "include(\"\${CIRCT_CMAKE_DIR}/CIRCTTargets.cmake\")")
set(CIRCT_CONFIG_INCLUDE_DIRS
  "\${CIRCT_INSTALL_PREFIX}/include"
  )
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/CIRCTConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CIRCTConfig.cmake
    @ONLY)
set(CIRCT_CONFIG_CODE)
set(CIRCT_CONFIG_CMAKE_DIR)
set(CIRCT_CONFIG_LLVM_CMAKE_DIR)
set(CIRCT_CONFIG_MLIR_CMAKE_DIR)
set(CIRCT_CONFIG_LIBRARY_DIRS)
set(CIRCT_CONFIG_BINARY_DIR)
set(CIRCT_CONFIG_TOOLS_DIR)
set(CIRCT_CONFIG_INCLUDE_EXPORTS)
set(CIRCT_CONFIG_INCLUDE_DIRS)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  # Not TOOLCHAIN ONLY, so install the CIRCT parts as well
  # Include the cmake files so other tools can use circt-tblgen, etc.
  install(EXPORT CIRCTTargets DESTINATION ${CIRCT_INSTALL_PACKAGE_DIR}
          COMPONENT circt-cmake-exports)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CIRCTConfig.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AddCIRCT.cmake
    DESTINATION ${CIRCT_INSTALL_PACKAGE_DIR}
    COMPONENT circt-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(circt-cmake-exports)
    add_llvm_install_targets(install-circt-cmake-exports
                             COMPONENT circt-cmake-exports)
  endif()
endif()
