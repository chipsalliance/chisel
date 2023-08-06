include_guard()

function(add_circt_dialect dialect dialect_namespace)
  add_mlir_dialect(${ARGV})
  add_dependencies(circt-headers MLIR${dialect}IncGen)
endfunction()

function(add_circt_interface interface)
  add_mlir_interface(${ARGV})
  add_dependencies(circt-headers MLIR${interface}IncGen)
endfunction()

# Additional parameters are forwarded to tablegen.
function(add_circt_doc tablegen_file output_path command)
  set(LLVM_TARGET_DEFINITIONS ${tablegen_file}.td)
  string(MAKE_C_IDENTIFIER ${output_path} output_id)
  tablegen(MLIR ${output_id}.md ${command} ${ARGN})
  set(GEN_DOC_FILE ${CIRCT_BINARY_DIR}/docs/${output_path}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md)
  add_custom_target(${output_id}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(circt-doc ${output_id}DocGen)
endfunction()

function(add_circt_dialect_doc dialect dialect_namespace)
  add_circt_doc(
    ${dialect} Dialects/${dialect}
    -gen-dialect-doc -dialect ${dialect_namespace})
endfunction()

function(add_circt_library name)
  add_mlir_library(${ARGV})
  add_circt_library_install(${name})
endfunction()

# Adds a CIRCT library target for installation.  This should normally only be
# called from add_circt_library().
function(add_circt_library_install name)
  install(TARGETS ${name} COMPONENT ${name} EXPORT CIRCTTargets)
  set_property(GLOBAL APPEND PROPERTY CIRCT_ALL_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY CIRCT_EXPORTS ${name})
endfunction()

function(add_circt_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY CIRCT_DIALECT_LIBS ${name})
  add_circt_library(${ARGV} DEPENDS circt-headers)
endfunction()

function(add_circt_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY CIRCT_CONVERSION_LIBS ${name})
  add_circt_library(${ARGV} DEPENDS circt-headers)
endfunction()

function(add_circt_translation_library name)
  set_property(GLOBAL APPEND PROPERTY CIRCT_TRANSLATION_LIBS ${name})
  add_circt_library(${ARGV} DEPENDS circt-headers)
endfunction()

function(add_circt_verification_library name)
  set_property(GLOBAL APPEND PROPERTY CIRCT_VERIFICATION_LIBS ${name})
  add_circt_library(${ARGV} DEPENDS circt-headers)
endfunction()
