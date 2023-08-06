add_circt_dialect(Handshake handshake)
add_circt_dialect_doc(Handshake handshake)
add_circt_interface(HandshakeInterfaces)

set(LLVM_TARGET_DEFINITIONS Handshake.td)

mlir_tablegen(HandshakeEnums.h.inc -gen-enum-decls)
mlir_tablegen(HandshakeEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHandshakeEnumsIncGen)
add_dependencies(circt-headers MLIRHandshakeEnumsIncGen)

mlir_tablegen(HandshakeAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=handshake)
mlir_tablegen(HandshakeAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=handshake)
add_public_tablegen_target(MLIRHandshakeAttributesIncGen)
add_dependencies(circt-headers MLIRHandshakeAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS HandshakeCanonicalization.td)
mlir_tablegen(HandshakeCanonicalization.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRHandshakeCanonicalizationIncGen)
add_dependencies(circt-headers MLIRHandshakeCanonicalizationIncGen)

set(LLVM_TARGET_DEFINITIONS HandshakePasses.td)
mlir_tablegen(HandshakePasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTHandshakeTransformsIncGen)
add_circt_doc(HandshakePasses HandshakePasses -gen-pass-doc)
add_dependencies(circt-headers CIRCTHandshakeTransformsIncGen)
