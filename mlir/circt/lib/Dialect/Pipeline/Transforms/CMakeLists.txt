add_circt_dialect_library(CIRCTPipelineTransforms
  ExplicitRegs.cpp
  ScheduleLinearPipeline.cpp

  DEPENDS
  CIRCTPipelineTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTScheduling
  CIRCTComb
  CIRCTSSP
  CIRCTHW
  CIRCTPipelineOps
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
