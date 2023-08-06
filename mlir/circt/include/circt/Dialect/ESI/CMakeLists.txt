add_circt_dialect(ESI esi)

set(LLVM_TARGET_DEFINITIONS ESIPasses.td)
mlir_tablegen(ESIPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRESITransformsIncGen)
add_circt_doc(ESIPasses ESIPasses -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS ESI.td)
mlir_tablegen(ESIEnums.h.inc -gen-enum-decls)
mlir_tablegen(ESIEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRESIEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS ESIInterfaces.td)
mlir_tablegen(ESIInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ESIInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRESIInterfacesIncGen)
add_dependencies(circt-headers MLIRESIInterfacesIncGen)

add_circt_doc(ESIChannels Dialects/ESIChannels -gen-op-doc)
add_circt_doc(ESIChannels Dialects/ESIChannelTypes -gen-typedef-doc)
add_circt_doc(ESIServices Dialects/ESIServices -gen-op-doc)
add_circt_doc(ESIStdServices Dialects/ESIStdServices -gen-op-doc)
add_circt_doc(ESIStructure Dialects/ESIStructure -gen-op-doc)

add_circt_doc(ESIInterfaces Dialects/ESIInterfaces -gen-op-interface-docs)
add_circt_doc(ESITypes Dialects/ESITypes -gen-typedef-doc)


add_subdirectory(cosim)
