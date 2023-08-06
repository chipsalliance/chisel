set(HANDSHAKE_SRCS
  HandshakeExecutableOps.cpp
  HandshakeOps.cpp
  HandshakeDialect.cpp
  )

add_circt_dialect_library(CIRCTHandshake
  ${HANDSHAKE_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR

  DEPENDS
  MLIRArithDialect
  MLIRHandshakeInterfacesIncGen
  MLIRHandshakeCanonicalizationIncGen
  MLIRHandshakeAttributesIncGen
  MLIRHandshakeEnumsIncGen
  )

add_subdirectory(Transforms)
