; SPDX-License-Identifier: Apache-2.0
circuit CanTop :
  module CanRegisterDoubleReset :
    input clock : Clock
    input reset : UInt<1>
    input io_resetSync : UInt<1>
    input io_dataIn : UInt<1>
    output io_dataOut : UInt<1>
    input io_writeEn : UInt<1>

    reg dataOut : UInt<1>, clock with :
      reset => (UInt<1>("h0"), dataOut) @[CanRegisterDoubleReset.scala 13:31]
    node _GEN_0 = mux(io_writeEn, io_dataIn, dataOut) @[CanRegisterDoubleReset.scala 17:26 18:15 13:31]
    node _GEN_1 = or(io_resetSync, _GEN_0) @[CanRegisterDoubleReset.scala 15:22 16:13]
    io_dataOut <= dataOut @[CanRegisterDoubleReset.scala 14:14]
    dataOut <= or(reset, _GEN_1) @[CanRegisterDoubleReset.scala 13:{31,31}]

  module CanRegister :
    input clock : Clock
    input reset : UInt<1>
    input io_dataIn : UInt<4>
    output io_dataOut : UInt<4>
    input io_writeEn : UInt<1>

    reg dataOut : UInt<4>, clock with :
      reset => (UInt<1>("h0"), dataOut) @[CanRegister.scala 13:31]
    node _GEN_0 = mux(io_writeEn, io_dataIn, dataOut) @[CanRegister.scala 15:20 16:13 13:31]
    io_dataOut <= dataOut @[CanRegister.scala 14:14]
    dataOut <= mux(reset, UInt<4>("h0"), _GEN_0) @[CanRegister.scala 13:{31,31}]

  module CanRegister_1 :
    input clock : Clock
    input reset : UInt<1>
    input io_dataIn : UInt<3>
    output io_dataOut : UInt<3>
    input io_writeEn : UInt<1>

    reg dataOut : UInt<3>, clock with :
      reset => (UInt<1>("h0"), dataOut) @[CanRegister.scala 13:31]
    node _GEN_0 = mux(io_writeEn, io_dataIn, dataOut) @[CanRegister.scala 15:20 16:13 13:31]
    io_dataOut <= dataOut @[CanRegister.scala 14:14]
    dataOut <= mux(reset, UInt<3>("h0"), _GEN_0) @[CanRegister.scala 13:{31,31}]

  module CanRegisterDoubleReset_1 :
    input clock : Clock
    input reset : UInt<1>
    input io_resetSync : UInt<1>
    input io_dataIn : UInt<1>
    output io_dataOut : UInt<1>
    input io_writeEn : UInt<1>

    reg dataOut : UInt<1>, clock with :
      reset => (UInt<1>("h0"), dataOut) @[CanRegisterDoubleReset.scala 13:31]
    node _GEN_0 = mux(io_writeEn, io_dataIn, dataOut) @[CanRegisterDoubleReset.scala 17:26 18:15 13:31]
    node _GEN_1 = mux(io_resetSync, UInt<1>("h0"), _GEN_0) @[CanRegisterDoubleReset.scala 15:22 16:13]
    io_dataOut <= dataOut @[CanRegisterDoubleReset.scala 14:14]
    dataOut <= mux(reset, UInt<1>("h0"), _GEN_1) @[CanRegisterDoubleReset.scala 13:{31,31}]

  module CanRegister_2 :
    input clock : Clock
    input reset : UInt<1>
    input io_dataIn : UInt<8>
    output io_dataOut : UInt<8>
    input io_writeEn : UInt<1>

    reg dataOut : UInt<8>, clock with :
      reset => (UInt<1>("h0"), dataOut) @[CanRegister.scala 13:31]
    node _GEN_0 = mux(io_writeEn, io_dataIn, dataOut) @[CanRegister.scala 15:20 16:13 13:31]
    io_dataOut <= dataOut @[CanRegister.scala 14:14]
    dataOut <= mux(reset, UInt<8>("h0"), _GEN_0) @[CanRegister.scala 13:{31,31}]

  module CanRegister_5 :
    input clock : Clock
    input reset : UInt<1>
    input io_dataIn : UInt<8>
    output io_dataOut : UInt<8>
    input io_writeEn : UInt<1>

    reg dataOut : UInt<8>, clock with :
      reset => (UInt<1>("h0"), dataOut) @[CanRegister.scala 13:31]
    node _GEN_0 = mux(io_writeEn, io_dataIn, dataOut) @[CanRegister.scala 15:20 16:13 13:31]
    io_dataOut <= dataOut @[CanRegister.scala 14:14]
    dataOut <= mux(reset, UInt<8>("h60"), _GEN_0) @[CanRegister.scala 13:{31,31}]

  module CanRegister_6 :
    input clock : Clock
    input reset : UInt<1>
    input io_dataIn : UInt<1>
    output io_dataOut : UInt<1>
    input io_writeEn : UInt<1>

    reg dataOut : UInt<1>, clock with :
      reset => (UInt<1>("h0"), dataOut) @[CanRegister.scala 13:31]
    node _GEN_0 = mux(io_writeEn, io_dataIn, dataOut) @[CanRegister.scala 15:20 16:13 13:31]
    io_dataOut <= dataOut @[CanRegister.scala 14:14]
    dataOut <= mux(reset, UInt<1>("h0"), _GEN_0) @[CanRegister.scala 13:{31,31}]

  module CanRegisters :
    input clock : Clock
    input reset : UInt<1>
    input io_cs : UInt<1>
    input io_writeEn : UInt<1>
    input io_addr : UInt<8>
    input io_dataIn : UInt<8>
    output io_dataOut : UInt<8>
    output io_irqN : UInt<1>
    input io_samplePoint : UInt<1>
    input io_transmitting : UInt<1>
    input io_setResetMode : UInt<1>
    input io_nodeBusOff : UInt<1>
    input io_errorStatus : UInt<1>
    input io_rxErrorCount : UInt<8>
    input io_txErrorCount : UInt<8>
    input io_transmitStatus : UInt<1>
    input io_receiveStatus : UInt<1>
    input io_txSuccessful : UInt<1>
    input io_needToTx : UInt<1>
    input io_overrun : UInt<1>
    input io_infoEmpty : UInt<1>
    input io_setBusErrorIrq : UInt<1>
    input io_setArbitrationLostIrq : UInt<1>
    input io_arbitrationLostCapture : UInt<5>
    input io_nodeErrorPassive : UInt<1>
    input io_nodeErrorActive : UInt<1>
    input io_rxMessageCounter : UInt<7>
    output io_resetMode : UInt<1>
    output io_listenOnlyMode : UInt<1>
    output io_acceptanceFilterMode : UInt<1>
    output io_selfTestMode : UInt<1>
    output io_clearDataOverrun : UInt<1>
    output io_releaseBuffer : UInt<1>
    output io_abortTx : UInt<1>
    output io_txRequest : UInt<1>
    output io_selfRxRequest : UInt<1>
    output io_singleShotTransmission : UInt<1>
    input io_txState : UInt<1>
    input io_txStateQ : UInt<1>
    input io_overloadFrame : UInt<1>
    output io_readArbitrationLostCaptureReg : UInt<1>
    output io_readErrorCodeCaptureReg : UInt<1>
    input io_errorCaptureCode : UInt<8>
    output io_baudRatePrescaler : UInt<6>
    output io_syncJumpWidth : UInt<2>
    output io_timeSegment1 : UInt<4>
    output io_timeSegment2 : UInt<3>
    output io_tripleSampling : UInt<1>
    output io_errorWarningLimit : UInt<8>
    output io_writeEnReceiveErrorCounter : UInt<1>
    output io_writeEnTransmitErrorCounter : UInt<1>
    output io_extendedMode : UInt<1>
    output io_clkout : UInt<1>
    output io_acceptanceCode_0 : UInt<8>
    output io_acceptanceCode_1 : UInt<8>
    output io_acceptanceCode_2 : UInt<8>
    output io_acceptanceCode_3 : UInt<8>
    output io_acceptanceMask_0 : UInt<8>
    output io_acceptanceMask_1 : UInt<8>
    output io_acceptanceMask_2 : UInt<8>
    output io_acceptanceMask_3 : UInt<8>
    output io_txData_0 : UInt<8>
    output io_txData_1 : UInt<8>
    output io_txData_2 : UInt<8>
    output io_txData_3 : UInt<8>
    output io_txData_4 : UInt<8>
    output io_txData_5 : UInt<8>
    output io_txData_6 : UInt<8>
    output io_txData_7 : UInt<8>
    output io_txData_8 : UInt<8>
    output io_txData_9 : UInt<8>
    output io_txData_10 : UInt<8>
    output io_txData_11 : UInt<8>
    output io_txData_12 : UInt<8>

    inst mode_m of CanRegisterDoubleReset @[CanRegisterDoubleReset.scala 24:19]
    inst modeBasic_m of CanRegister @[CanRegister.scala 30:19]
    inst modeExt_m of CanRegister_1 @[CanRegister.scala 30:19]
    inst command_0_m of CanRegisterDoubleReset_1 @[CanRegisterDoubleReset.scala 24:19]
    inst command_1_m of CanRegisterDoubleReset_1 @[CanRegisterDoubleReset.scala 24:19]
    inst command_2_m of CanRegisterDoubleReset_1 @[CanRegisterDoubleReset.scala 24:19]
    inst command_3_m of CanRegisterDoubleReset_1 @[CanRegisterDoubleReset.scala 24:19]
    inst command_4_m of CanRegisterDoubleReset_1 @[CanRegisterDoubleReset.scala 24:19]
    inst io_overloadRequest_m of CanRegisterDoubleReset_1 @[CanRegisterDoubleReset.scala 24:19]
    inst irqEnExt_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst busTiming0_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst busTiming1_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_errorWarningLimit_m of CanRegister_5 @[CanRegister.scala 30:19]
    inst io_extendedMode_m of CanRegister_6 @[CanRegister.scala 30:19]
    inst clockOff_m of CanRegister_6 @[CanRegister.scala 30:19]
    inst cd_m of CanRegister_1 @[CanRegister.scala 30:19]
    inst io_acceptanceCode_0_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_acceptanceMask_0_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_acceptanceCode_1_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_acceptanceMask_1_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_acceptanceCode_2_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_acceptanceMask_2_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_acceptanceCode_3_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_acceptanceMask_3_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_txData_0_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_txData_1_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_txData_2_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_txData_3_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_txData_4_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_txData_5_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_txData_6_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_txData_7_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_txData_8_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_txData_9_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_txData_10_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_txData_11_m of CanRegister_2 @[CanRegister.scala 24:19]
    inst io_txData_12_m of CanRegister_2 @[CanRegister.scala 24:19]
    reg irqN : UInt<1>, clock with :
      reset => (UInt<1>("h0"), irqN) @[CanRegisters.scala 100:27]
    reg selfRxRequest : UInt<1>, clock with :
      reset => (UInt<1>("h0"), selfRxRequest) @[CanRegisters.scala 102:36]
    reg singleShotTransmission : UInt<1>, clock with :
      reset => (UInt<1>("h0"), singleShotTransmission) @[CanRegisters.scala 104:45]
    reg txSuccessfulQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), txSuccessfulQ) @[CanRegisters.scala 106:36]
    reg overrunQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), overrunQ) @[CanRegisters.scala 107:31]
    reg overrunStatus : UInt<1>, clock with :
      reset => (UInt<1>("h0"), overrunStatus) @[CanRegisters.scala 108:36]
    reg transmissionComplete : UInt<1>, clock with :
      reset => (UInt<1>("h0"), transmissionComplete) @[CanRegisters.scala 109:43]
    reg transmitBufferStatus : UInt<1>, clock with :
      reset => (UInt<1>("h0"), transmitBufferStatus) @[CanRegisters.scala 110:43]
    reg transmitBufferStatusQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), transmitBufferStatusQ) @[CanRegisters.scala 111:44]
    reg receiveBufferStatus : UInt<1>, clock with :
      reset => (UInt<1>("h0"), receiveBufferStatus) @[CanRegisters.scala 112:42]
    reg errorStatusQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), errorStatusQ) @[CanRegisters.scala 113:35]
    reg nodeBusOffQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), nodeBusOffQ) @[CanRegisters.scala 114:34]
    reg nodeErrorPassiveQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), nodeErrorPassiveQ) @[CanRegisters.scala 115:40]
    node _writeEnMode_T = and(io_cs, io_writeEn) @[CanRegisters.scala 97:11]
    node _writeEnMode_T_1 = eq(io_addr, UInt<8>("h0")) @[CanRegisters.scala 97:35]
    node writeEnMode = and(_writeEnMode_T, _writeEnMode_T_1) @[CanRegisters.scala 97:24]
    skip
    node _writeEnCommand_T_1 = eq(io_addr, UInt<8>("h1")) @[CanRegisters.scala 97:35]
    skip
    skip
    node _writeEnBusTiming0_T_1 = eq(io_addr, UInt<8>("h6")) @[CanRegisters.scala 97:35]
    node _writeEnBusTiming0_T_2 = and(_writeEnMode_T, _writeEnBusTiming0_T_1) @[CanRegisters.scala 97:24]
    skip
    skip
    node _writeEnBusTiming1_T_1 = eq(io_addr, UInt<8>("h7")) @[CanRegisters.scala 97:35]
    node _writeEnBusTiming1_T_2 = and(_writeEnMode_T, _writeEnBusTiming1_T_1) @[CanRegisters.scala 97:24]
    skip
    skip
    node _writeEnClockDivLow_T_1 = eq(io_addr, UInt<8>("h1f")) @[CanRegisters.scala 97:35]
    node writeEnClockDivLow = and(_writeEnMode_T, _writeEnClockDivLow_T_1) @[CanRegisters.scala 97:24]
    skip
    node _read_T = not(io_writeEn) @[CanRegisters.scala 132:29]
    node read = and(io_cs, _read_T) @[CanRegisters.scala 132:26]
    node _readIrqReg_T = eq(io_addr, UInt<8>("h3")) @[CanRegisters.scala 133:42]
    node readIrqReg = and(read, _readIrqReg_T) @[CanRegisters.scala 133:31]
    node _io_readArbitrationLostCaptureReg_T = and(read, io_extendedMode) @[CanRegisters.scala 134:44]
    node _io_readArbitrationLostCaptureReg_T_1 = eq(io_addr, UInt<8>("hb")) @[CanRegisters.scala 134:73]
    skip
    skip
    node _io_readErrorCodeCaptureReg_T_1 = eq(io_addr, UInt<8>("hc")) @[CanRegisters.scala 135:67]
    skip
    skip
    node _writeEnAcceptanceCode_0_T_1 = not(io_extendedMode) @[CanRegisters.scala 93:28]
    node _writeEnAcceptanceCode_0_T_2 = eq(io_addr, UInt<8>("h4")) @[CanRegisters.scala 93:57]
    node _writeEnAcceptanceCode_0_T_3 = and(_writeEnAcceptanceCode_0_T_1, _writeEnAcceptanceCode_0_T_2) @[CanRegisters.scala 93:46]
    node _writeEnAcceptanceCode_0_T_4 = eq(io_addr, UInt<8>("h10")) @[CanRegisters.scala 93:107]
    node _writeEnAcceptanceCode_0_T_5 = and(io_extendedMode, _writeEnAcceptanceCode_0_T_4) @[CanRegisters.scala 93:96]
    node _writeEnAcceptanceCode_0_T_6 = or(_writeEnAcceptanceCode_0_T_3, _writeEnAcceptanceCode_0_T_5) @[CanRegisters.scala 93:78]
    node _writeEnAcceptanceCode_0_T_7 = and(_writeEnMode_T, _writeEnAcceptanceCode_0_T_6) @[CanRegisters.scala 93:24]
    skip
    skip
    skip
    node _writeEnAcceptanceMask_0_T_2 = eq(io_addr, UInt<8>("h5")) @[CanRegisters.scala 93:57]
    node _writeEnAcceptanceMask_0_T_3 = and(_writeEnAcceptanceCode_0_T_1, _writeEnAcceptanceMask_0_T_2) @[CanRegisters.scala 93:46]
    node _writeEnAcceptanceMask_0_T_4 = eq(io_addr, UInt<8>("h14")) @[CanRegisters.scala 93:107]
    node _writeEnAcceptanceMask_0_T_5 = and(io_extendedMode, _writeEnAcceptanceMask_0_T_4) @[CanRegisters.scala 93:96]
    node _writeEnAcceptanceMask_0_T_6 = or(_writeEnAcceptanceMask_0_T_3, _writeEnAcceptanceMask_0_T_5) @[CanRegisters.scala 93:78]
    node _writeEnAcceptanceMask_0_T_7 = and(_writeEnMode_T, _writeEnAcceptanceMask_0_T_6) @[CanRegisters.scala 93:24]
    skip
    skip
    node _writeEnAcceptanceCode_1_T_1 = eq(io_addr, UInt<8>("h11")) @[CanRegisters.scala 89:35]
    node _writeEnAcceptanceCode_1_T_2 = and(_writeEnMode_T, _writeEnAcceptanceCode_1_T_1) @[CanRegisters.scala 89:24]
    node _writeEnAcceptanceCode_1_T_3 = and(_writeEnAcceptanceCode_1_T_2, io_extendedMode) @[CanRegisters.scala 89:59]
    skip
    skip
    node _writeEnAcceptanceMask_1_T_1 = eq(io_addr, UInt<8>("h15")) @[CanRegisters.scala 89:35]
    node _writeEnAcceptanceMask_1_T_2 = and(_writeEnMode_T, _writeEnAcceptanceMask_1_T_1) @[CanRegisters.scala 89:24]
    node _writeEnAcceptanceMask_1_T_3 = and(_writeEnAcceptanceMask_1_T_2, io_extendedMode) @[CanRegisters.scala 89:59]
    skip
    skip
    node _writeEnAcceptanceCode_2_T_1 = eq(io_addr, UInt<8>("h12")) @[CanRegisters.scala 89:35]
    node _writeEnAcceptanceCode_2_T_2 = and(_writeEnMode_T, _writeEnAcceptanceCode_2_T_1) @[CanRegisters.scala 89:24]
    node _writeEnAcceptanceCode_2_T_3 = and(_writeEnAcceptanceCode_2_T_2, io_extendedMode) @[CanRegisters.scala 89:59]
    skip
    skip
    node _writeEnAcceptanceMask_2_T_1 = eq(io_addr, UInt<8>("h16")) @[CanRegisters.scala 89:35]
    node _writeEnAcceptanceMask_2_T_2 = and(_writeEnMode_T, _writeEnAcceptanceMask_2_T_1) @[CanRegisters.scala 89:24]
    node _writeEnAcceptanceMask_2_T_3 = and(_writeEnAcceptanceMask_2_T_2, io_extendedMode) @[CanRegisters.scala 89:59]
    skip
    skip
    node _writeEnAcceptanceCode_3_T_1 = eq(io_addr, UInt<8>("h13")) @[CanRegisters.scala 89:35]
    node _writeEnAcceptanceCode_3_T_2 = and(_writeEnMode_T, _writeEnAcceptanceCode_3_T_1) @[CanRegisters.scala 89:24]
    node _writeEnAcceptanceCode_3_T_3 = and(_writeEnAcceptanceCode_3_T_2, io_extendedMode) @[CanRegisters.scala 89:59]
    skip
    skip
    node _writeEnAcceptanceMask_3_T_1 = eq(io_addr, UInt<8>("h17")) @[CanRegisters.scala 89:35]
    node _writeEnAcceptanceMask_3_T_2 = and(_writeEnMode_T, _writeEnAcceptanceMask_3_T_1) @[CanRegisters.scala 89:24]
    node _writeEnAcceptanceMask_3_T_3 = and(_writeEnAcceptanceMask_3_T_2, io_extendedMode) @[CanRegisters.scala 89:59]
    skip
    skip
    skip
    node _writeEnTxData_0_T_2 = eq(io_addr, UInt<8>("ha")) @[CanRegisters.scala 93:57]
    node _writeEnTxData_0_T_3 = and(_writeEnAcceptanceCode_0_T_1, _writeEnTxData_0_T_2) @[CanRegisters.scala 93:46]
    skip
    skip
    node _writeEnTxData_0_T_6 = or(_writeEnTxData_0_T_3, _writeEnAcceptanceCode_0_T_5) @[CanRegisters.scala 93:78]
    node _writeEnTxData_0_T_7 = and(_writeEnMode_T, _writeEnTxData_0_T_6) @[CanRegisters.scala 93:24]
    node _writeEnTxData_0_T_8 = not(io_resetMode) @[CanRegisters.scala 150:61]
    node _writeEnTxData_0_T_9 = and(_writeEnTxData_0_T_7, _writeEnTxData_0_T_8) @[CanRegisters.scala 150:58]
    skip
    skip
    skip
    skip
    node _writeEnTxData_1_T_3 = and(_writeEnAcceptanceCode_0_T_1, _io_readArbitrationLostCaptureReg_T_1) @[CanRegisters.scala 93:46]
    skip
    node _writeEnTxData_1_T_5 = and(io_extendedMode, _writeEnAcceptanceCode_1_T_1) @[CanRegisters.scala 93:96]
    node _writeEnTxData_1_T_6 = or(_writeEnTxData_1_T_3, _writeEnTxData_1_T_5) @[CanRegisters.scala 93:78]
    node _writeEnTxData_1_T_7 = and(_writeEnMode_T, _writeEnTxData_1_T_6) @[CanRegisters.scala 93:24]
    skip
    node _writeEnTxData_1_T_9 = and(_writeEnTxData_1_T_7, _writeEnTxData_0_T_8) @[CanRegisters.scala 150:58]
    skip
    skip
    skip
    skip
    node _writeEnTxData_2_T_3 = and(_writeEnAcceptanceCode_0_T_1, _io_readErrorCodeCaptureReg_T_1) @[CanRegisters.scala 93:46]
    skip
    node _writeEnTxData_2_T_5 = and(io_extendedMode, _writeEnAcceptanceCode_2_T_1) @[CanRegisters.scala 93:96]
    node _writeEnTxData_2_T_6 = or(_writeEnTxData_2_T_3, _writeEnTxData_2_T_5) @[CanRegisters.scala 93:78]
    node _writeEnTxData_2_T_7 = and(_writeEnMode_T, _writeEnTxData_2_T_6) @[CanRegisters.scala 93:24]
    skip
    node _writeEnTxData_2_T_9 = and(_writeEnTxData_2_T_7, _writeEnTxData_0_T_8) @[CanRegisters.scala 150:58]
    skip
    skip
    skip
    node _writeEnTxData_3_T_2 = eq(io_addr, UInt<8>("hd")) @[CanRegisters.scala 93:57]
    node _writeEnTxData_3_T_3 = and(_writeEnAcceptanceCode_0_T_1, _writeEnTxData_3_T_2) @[CanRegisters.scala 93:46]
    skip
    node _writeEnTxData_3_T_5 = and(io_extendedMode, _writeEnAcceptanceCode_3_T_1) @[CanRegisters.scala 93:96]
    node _writeEnTxData_3_T_6 = or(_writeEnTxData_3_T_3, _writeEnTxData_3_T_5) @[CanRegisters.scala 93:78]
    node _writeEnTxData_3_T_7 = and(_writeEnMode_T, _writeEnTxData_3_T_6) @[CanRegisters.scala 93:24]
    skip
    node _writeEnTxData_3_T_9 = and(_writeEnTxData_3_T_7, _writeEnTxData_0_T_8) @[CanRegisters.scala 150:58]
    skip
    skip
    skip
    node _writeEnTxData_4_T_2 = eq(io_addr, UInt<8>("he")) @[CanRegisters.scala 93:57]
    node _writeEnTxData_4_T_3 = and(_writeEnAcceptanceCode_0_T_1, _writeEnTxData_4_T_2) @[CanRegisters.scala 93:46]
    skip
    skip
    node _writeEnTxData_4_T_6 = or(_writeEnTxData_4_T_3, _writeEnAcceptanceMask_0_T_5) @[CanRegisters.scala 93:78]
    node _writeEnTxData_4_T_7 = and(_writeEnMode_T, _writeEnTxData_4_T_6) @[CanRegisters.scala 93:24]
    skip
    node _writeEnTxData_4_T_9 = and(_writeEnTxData_4_T_7, _writeEnTxData_0_T_8) @[CanRegisters.scala 150:58]
    skip
    skip
    skip
    node _writeEnTxData_5_T_2 = eq(io_addr, UInt<8>("hf")) @[CanRegisters.scala 93:57]
    node _writeEnTxData_5_T_3 = and(_writeEnAcceptanceCode_0_T_1, _writeEnTxData_5_T_2) @[CanRegisters.scala 93:46]
    skip
    node _writeEnTxData_5_T_5 = and(io_extendedMode, _writeEnAcceptanceMask_1_T_1) @[CanRegisters.scala 93:96]
    node _writeEnTxData_5_T_6 = or(_writeEnTxData_5_T_3, _writeEnTxData_5_T_5) @[CanRegisters.scala 93:78]
    node _writeEnTxData_5_T_7 = and(_writeEnMode_T, _writeEnTxData_5_T_6) @[CanRegisters.scala 93:24]
    skip
    node _writeEnTxData_5_T_9 = and(_writeEnTxData_5_T_7, _writeEnTxData_0_T_8) @[CanRegisters.scala 150:58]
    skip
    skip
    skip
    skip
    node _writeEnTxData_6_T_3 = and(_writeEnAcceptanceCode_0_T_1, _writeEnAcceptanceCode_0_T_4) @[CanRegisters.scala 93:46]
    skip
    node _writeEnTxData_6_T_5 = and(io_extendedMode, _writeEnAcceptanceMask_2_T_1) @[CanRegisters.scala 93:96]
    node _writeEnTxData_6_T_6 = or(_writeEnTxData_6_T_3, _writeEnTxData_6_T_5) @[CanRegisters.scala 93:78]
    node _writeEnTxData_6_T_7 = and(_writeEnMode_T, _writeEnTxData_6_T_6) @[CanRegisters.scala 93:24]
    skip
    node _writeEnTxData_6_T_9 = and(_writeEnTxData_6_T_7, _writeEnTxData_0_T_8) @[CanRegisters.scala 150:58]
    skip
    skip
    skip
    skip
    node _writeEnTxData_7_T_3 = and(_writeEnAcceptanceCode_0_T_1, _writeEnAcceptanceCode_1_T_1) @[CanRegisters.scala 93:46]
    skip
    node _writeEnTxData_7_T_5 = and(io_extendedMode, _writeEnAcceptanceMask_3_T_1) @[CanRegisters.scala 93:96]
    node _writeEnTxData_7_T_6 = or(_writeEnTxData_7_T_3, _writeEnTxData_7_T_5) @[CanRegisters.scala 93:78]
    node _writeEnTxData_7_T_7 = and(_writeEnMode_T, _writeEnTxData_7_T_6) @[CanRegisters.scala 93:24]
    skip
    node _writeEnTxData_7_T_9 = and(_writeEnTxData_7_T_7, _writeEnTxData_0_T_8) @[CanRegisters.scala 150:58]
    skip
    skip
    skip
    skip
    node _writeEnTxData_8_T_3 = and(_writeEnAcceptanceCode_0_T_1, _writeEnAcceptanceCode_2_T_1) @[CanRegisters.scala 93:46]
    node _writeEnTxData_8_T_4 = eq(io_addr, UInt<8>("h18")) @[CanRegisters.scala 93:107]
    node _writeEnTxData_8_T_5 = and(io_extendedMode, _writeEnTxData_8_T_4) @[CanRegisters.scala 93:96]
    node _writeEnTxData_8_T_6 = or(_writeEnTxData_8_T_3, _writeEnTxData_8_T_5) @[CanRegisters.scala 93:78]
    node _writeEnTxData_8_T_7 = and(_writeEnMode_T, _writeEnTxData_8_T_6) @[CanRegisters.scala 93:24]
    skip
    node _writeEnTxData_8_T_9 = and(_writeEnTxData_8_T_7, _writeEnTxData_0_T_8) @[CanRegisters.scala 150:58]
    skip
    skip
    skip
    skip
    node _writeEnTxData_9_T_3 = and(_writeEnAcceptanceCode_0_T_1, _writeEnAcceptanceCode_3_T_1) @[CanRegisters.scala 93:46]
    node _writeEnTxData_9_T_4 = eq(io_addr, UInt<8>("h19")) @[CanRegisters.scala 93:107]
    node _writeEnTxData_9_T_5 = and(io_extendedMode, _writeEnTxData_9_T_4) @[CanRegisters.scala 93:96]
    node _writeEnTxData_9_T_6 = or(_writeEnTxData_9_T_3, _writeEnTxData_9_T_5) @[CanRegisters.scala 93:78]
    node _writeEnTxData_9_T_7 = and(_writeEnMode_T, _writeEnTxData_9_T_6) @[CanRegisters.scala 93:24]
    skip
    node _writeEnTxData_9_T_9 = and(_writeEnTxData_9_T_7, _writeEnTxData_0_T_8) @[CanRegisters.scala 150:58]
    skip
    skip
    node _writeEnTxData_10_T_1 = eq(io_addr, UInt<8>("h1a")) @[CanRegisters.scala 89:35]
    node _writeEnTxData_10_T_2 = and(_writeEnMode_T, _writeEnTxData_10_T_1) @[CanRegisters.scala 89:24]
    node _writeEnTxData_10_T_3 = and(_writeEnTxData_10_T_2, io_extendedMode) @[CanRegisters.scala 89:59]
    skip
    node _writeEnTxData_10_T_5 = and(_writeEnTxData_10_T_3, _writeEnTxData_0_T_8) @[CanRegisters.scala 152:48]
    skip
    skip
    node _writeEnTxData_11_T_1 = eq(io_addr, UInt<8>("h1b")) @[CanRegisters.scala 89:35]
    node _writeEnTxData_11_T_2 = and(_writeEnMode_T, _writeEnTxData_11_T_1) @[CanRegisters.scala 89:24]
    node _writeEnTxData_11_T_3 = and(_writeEnTxData_11_T_2, io_extendedMode) @[CanRegisters.scala 89:59]
    skip
    node _writeEnTxData_11_T_5 = and(_writeEnTxData_11_T_3, _writeEnTxData_0_T_8) @[CanRegisters.scala 152:48]
    skip
    skip
    node _writeEnTxData_12_T_1 = eq(io_addr, UInt<8>("h1c")) @[CanRegisters.scala 89:35]
    node _writeEnTxData_12_T_2 = and(_writeEnMode_T, _writeEnTxData_12_T_1) @[CanRegisters.scala 89:24]
    node _writeEnTxData_12_T_3 = and(_writeEnTxData_12_T_2, io_extendedMode) @[CanRegisters.scala 89:59]
    skip
    node _writeEnTxData_12_T_5 = and(_writeEnTxData_12_T_3, _writeEnTxData_0_T_8) @[CanRegisters.scala 152:48]
    skip
    skip
    skip
    node _writeEnInterruptEnable_T_2 = and(_writeEnMode_T, _writeEnAcceptanceCode_0_T_2) @[CanRegisters.scala 89:24]
    skip
    skip
    skip
    node _writeEnErrorWarningLimit_T_2 = and(_writeEnMode_T, _writeEnTxData_3_T_2) @[CanRegisters.scala 89:24]
    node _writeEnErrorWarningLimit_T_3 = and(_writeEnErrorWarningLimit_T_2, io_extendedMode) @[CanRegisters.scala 89:59]
    skip
    skip
    skip
    node _io_writeEnReceiveErrorCounter_T_2 = and(_writeEnMode_T, _writeEnTxData_4_T_2) @[CanRegisters.scala 89:24]
    node _io_writeEnReceiveErrorCounter_T_3 = and(_io_writeEnReceiveErrorCounter_T_2, io_extendedMode) @[CanRegisters.scala 89:59]
    skip
    skip
    skip
    node _io_writeEnTransmitErrorCounter_T_2 = and(_writeEnMode_T, _writeEnTxData_5_T_2) @[CanRegisters.scala 89:24]
    node _io_writeEnTransmitErrorCounter_T_3 = and(_io_writeEnTransmitErrorCounter_T_2, io_extendedMode) @[CanRegisters.scala 89:59]
    skip
    skip
    node mode = mode_m.io_dataOut @[CanRegisters.scala 160:93]
    skip
    skip
    skip
    node receiveIrqEnBasic = bits(modeBasic_m.io_dataOut, 0, 0) @[CanRegisters.scala 163:42]
    node transmitIrqEnBasic = bits(modeBasic_m.io_dataOut, 1, 1) @[CanRegisters.scala 164:43]
    node errorIrqEnBasic = bits(modeBasic_m.io_dataOut, 2, 2) @[CanRegisters.scala 165:40]
    node overrunIrqEnBasic = bits(modeBasic_m.io_dataOut, 3, 3) @[CanRegisters.scala 166:42]
    node _io_listenOnlyMode_T = bits(modeExt_m.io_dataOut, 0, 0) @[CanRegisters.scala 169:49]
    skip
    node _io_selfTestMode_T = bits(modeExt_m.io_dataOut, 1, 1) @[CanRegisters.scala 170:47]
    skip
    node _io_acceptanceFilterMode_T = bits(modeExt_m.io_dataOut, 2, 2) @[CanRegisters.scala 171:55]
    skip
    node command_0 = command_0_m.io_dataOut @[CanRegisters.scala 173:32 174:14]
    node _command_0_T = and(command_0, io_samplePoint) @[CanRegisters.scala 174:53]
    skip
    skip
    node _command_1_T = not(io_transmitting) @[CanRegisters.scala 175:89]
    node _command_1_T_1 = and(io_abortTx, _command_1_T) @[CanRegisters.scala 175:87]
    node _command_1_T_2 = or(io_txRequest, _command_1_T_1) @[CanRegisters.scala 175:73]
    node _command_1_T_3 = and(io_samplePoint, _command_1_T_2) @[CanRegisters.scala 175:57]
    skip
    skip
    node command_3 = command_3_m.io_dataOut @[CanRegisters.scala 173:32 177:14]
    node command_2 = command_2_m.io_dataOut @[CanRegisters.scala 173:32 176:14]
    node _command_2_T = or(command_3, command_2) @[CanRegisters.scala 176:53]
    skip
    skip
    skip
    skip
    skip
    node command_4 = command_4_m.io_dataOut @[CanRegisters.scala 173:32 178:14]
    node _command_4_T = and(command_4, io_samplePoint) @[CanRegisters.scala 178:53]
    skip
    skip
    node _T = not(command_0) @[CanRegisters.scala 180:21]
    node _T_1 = and(command_4, _T) @[CanRegisters.scala 180:19]
    node _T_2 = not(io_txState) @[CanRegisters.scala 182:15]
    node _T_3 = and(_T_2, io_txStateQ) @[CanRegisters.scala 182:28]
    node _GEN_0 = mux(_T_3, UInt<1>("h0"), selfRxRequest) @[CanRegisters.scala 182:43 183:19 102:36]
    node _GEN_1 = or(_T_1, _GEN_0) @[CanRegisters.scala 180:34 181:19]
    skip
    node _io_abortTx_T = not(io_txRequest) @[CanRegisters.scala 189:30]
    node command_1 = command_1_m.io_dataOut @[CanRegisters.scala 173:32 175:14]
    skip
    node _T_4 = and(io_txRequest, command_1) @[CanRegisters.scala 191:21]
    node _T_5 = and(_T_4, io_samplePoint) @[CanRegisters.scala 191:34]
    skip
    skip
    node _GEN_2 = mux(_T_3, UInt<1>("h0"), singleShotTransmission) @[CanRegisters.scala 194:42 195:28 104:45]
    node _GEN_3 = or(_T_5, _GEN_2) @[CanRegisters.scala 191:52 192:28]
    reg overloadFrameQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), overloadFrameQ) @[CanRegisters.scala 198:38]
    node _io_overloadRequest_T = not(overloadFrameQ) @[CanRegisters.scala 200:69]
    skip
    skip
    node status_lo_lo = cat(overrunStatus, receiveBufferStatus) @[Cat.scala 30:58]
    node status_lo_hi = cat(transmissionComplete, transmitBufferStatus) @[Cat.scala 30:58]
    node status_lo = cat(status_lo_hi, status_lo_lo) @[Cat.scala 30:58]
    node status_hi_lo = cat(io_transmitStatus, io_receiveStatus) @[Cat.scala 30:58]
    node status_hi_hi = cat(io_nodeBusOff, io_errorStatus) @[Cat.scala 30:58]
    node status_hi = cat(status_hi_hi, status_hi_lo) @[Cat.scala 30:58]
    node status = cat(status_hi, status_lo) @[Cat.scala 30:58]
    node _T_8 = not(io_needToTx) @[CanRegisters.scala 214:29]
    node _T_9 = or(io_resetMode, _T_8) @[CanRegisters.scala 214:27]
    node _GEN_4 = or(_T_9, transmitBufferStatus) @[CanRegisters.scala 214:43 215:26 110:43]
    node _GEN_5 = mux(io_txRequest, UInt<1>("h0"), _GEN_4) @[CanRegisters.scala 212:22 213:26]
    node _T_10 = not(txSuccessfulQ) @[CanRegisters.scala 218:27]
    node _T_11 = and(io_txSuccessful, _T_10) @[CanRegisters.scala 218:24]
    node _T_12 = or(_T_11, io_abortTx) @[CanRegisters.scala 218:43]
    node _GEN_6 = mux(io_txRequest, UInt<1>("h0"), transmissionComplete) @[CanRegisters.scala 221:27 222:26 109:43]
    node _GEN_7 = or(_T_12, _GEN_6) @[CanRegisters.scala 218:57 219:26]
    node _T_13 = not(overrunQ) @[CanRegisters.scala 225:21]
    node _T_14 = and(io_overrun, _T_13) @[CanRegisters.scala 225:19]
    node _T_15 = or(io_resetMode, io_clearDataOverrun) @[CanRegisters.scala 228:26]
    node _GEN_8 = mux(_T_15, UInt<1>("h0"), overrunStatus) @[CanRegisters.scala 228:49 229:19 108:36]
    node _GEN_9 = or(_T_14, _GEN_8) @[CanRegisters.scala 225:32 226:19]
    node _T_16 = or(io_resetMode, io_releaseBuffer) @[CanRegisters.scala 232:21]
    node _T_17 = not(io_infoEmpty) @[CanRegisters.scala 235:13]
    node _GEN_10 = or(_T_17, receiveBufferStatus) @[CanRegisters.scala 235:28 236:25 112:42]
    node _GEN_11 = mux(_T_16, UInt<1>("h0"), _GEN_10) @[CanRegisters.scala 232:41 233:25]
    node irqEnExt = irqEnExt_m.io_dataOut @[CanRegisters.scala 239:29 248:12]
    node busErrorIrqEn = bits(irqEnExt, 7, 7) @[CanRegisters.scala 240:38]
    node arbitrationLostIrqEn = bits(irqEnExt, 6, 6) @[CanRegisters.scala 241:45]
    node errorPassiveIrqEn = bits(irqEnExt, 5, 5) @[CanRegisters.scala 242:42]
    node dataOverrunIrqEnExt = bits(irqEnExt, 3, 3) @[CanRegisters.scala 243:44]
    node errorWarningIrqEnExt = bits(irqEnExt, 2, 2) @[CanRegisters.scala 244:45]
    node transmitIrqEnExt = bits(irqEnExt, 1, 1) @[CanRegisters.scala 245:41]
    node receiveIrqEnExt = bits(irqEnExt, 0, 0) @[CanRegisters.scala 246:40]
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    node clockOff = clockOff_m.io_dataOut @[CanRegisters.scala 264:79]
    skip
    node _clkoutDiv_T = eq(cd_m.io_dataOut, UInt<3>("h7")) @[CanRegisters.scala 266:33]
    node clkoutDiv = mux(_clkoutDiv_T, UInt<3>("h0"), cd_m.io_dataOut) @[CanRegisters.scala 266:29]
    node clockDivider_lo = cat(clockOff, cd_m.io_dataOut) @[Cat.scala 30:58]
    node clockDivider_hi = cat(io_extendedMode, UInt<3>("h0")) @[Cat.scala 30:58]
    node clockDivider = cat(clockDivider_hi, clockDivider_lo) @[Cat.scala 30:58]
    reg clkoutCnt : UInt<3>, clock with :
      reset => (UInt<1>("h0"), clkoutCnt) @[CanRegisters.scala 268:33]
    reg clkoutTmp : UInt<1>, clock with :
      reset => (UInt<1>("h0"), clkoutTmp) @[CanRegisters.scala 269:33]
    node _T_18 = eq(clkoutCnt, clkoutDiv) @[CanRegisters.scala 271:18]
    node _clkoutTmp_T = not(clkoutTmp) @[CanRegisters.scala 273:18]
    node _clkoutCnt_T = add(clkoutCnt, UInt<3>("h1")) @[CanRegisters.scala 276:28]
    node _clkoutCnt_T_1 = tail(_clkoutCnt_T, 1) @[CanRegisters.scala 276:28]
    node _GEN_12 = mux(_T_18, UInt<3>("h0"), _clkoutCnt_T_1) @[CanRegisters.scala 271:33 272:15 276:15]
    node _GEN_13 = mux(_T_18, _clkoutTmp_T, clkoutTmp) @[CanRegisters.scala 271:33 273:15 269:33]
    node _io_clkout_T = andr(cd_m.io_dataOut) @[CanRegisters.scala 279:50]
    node _io_clkout_T_1 = asUInt(clock) @[CanRegisters.scala 279:66]
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    node _io_clkout_T_7 = or(clockOff, _io_clkout_T_1) @[CanRegisters.scala 282:28]
    node _io_clkout_T_8 = or(clockOff, clkoutTmp) @[CanRegisters.scala 285:27]
    skip
    node _io_dataOut_T = bits(io_addr, 4, 0) @[CanRegisters.scala 298:22]
    node io_dataOut_hi = cat(UInt<4>("h0"), modeExt_m.io_dataOut) @[Cat.scala 30:58]
    node _io_dataOut_T_1 = cat(io_dataOut_hi, mode) @[Cat.scala 30:58]
    node _io_dataOut_T_2 = cat(UInt<3>("h0"), io_arbitrationLostCapture) @[Cat.scala 30:58]
    node _io_dataOut_T_3 = cat(UInt<1>("h0"), io_rxMessageCounter) @[Cat.scala 30:58]
    node _io_dataOut_T_4 = eq(UInt<5>("h1f"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_5 = mux(_io_dataOut_T_4, clockDivider, UInt<8>("h0")) @[Mux.scala 80:57]
    node _io_dataOut_T_6 = eq(UInt<5>("h1d"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_7 = mux(_io_dataOut_T_6, _io_dataOut_T_3, _io_dataOut_T_5) @[Mux.scala 80:57]
    node _io_dataOut_T_8 = eq(UInt<5>("h17"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_9 = mux(_io_dataOut_T_8, io_acceptanceMask_3, _io_dataOut_T_7) @[Mux.scala 80:57]
    node _io_dataOut_T_10 = eq(UInt<5>("h16"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_11 = mux(_io_dataOut_T_10, io_acceptanceMask_2, _io_dataOut_T_9) @[Mux.scala 80:57]
    node _io_dataOut_T_12 = eq(UInt<5>("h15"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_13 = mux(_io_dataOut_T_12, io_acceptanceMask_1, _io_dataOut_T_11) @[Mux.scala 80:57]
    node _io_dataOut_T_14 = eq(UInt<5>("h14"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_15 = mux(_io_dataOut_T_14, io_acceptanceMask_0, _io_dataOut_T_13) @[Mux.scala 80:57]
    node _io_dataOut_T_16 = eq(UInt<5>("h13"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_17 = mux(_io_dataOut_T_16, io_acceptanceCode_3, _io_dataOut_T_15) @[Mux.scala 80:57]
    node _io_dataOut_T_18 = eq(UInt<5>("h12"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_19 = mux(_io_dataOut_T_18, io_acceptanceCode_2, _io_dataOut_T_17) @[Mux.scala 80:57]
    node _io_dataOut_T_20 = eq(UInt<5>("h11"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_21 = mux(_io_dataOut_T_20, io_acceptanceCode_1, _io_dataOut_T_19) @[Mux.scala 80:57]
    node _io_dataOut_T_22 = eq(UInt<5>("h10"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_23 = mux(_io_dataOut_T_22, io_acceptanceCode_0, _io_dataOut_T_21) @[Mux.scala 80:57]
    node _io_dataOut_T_24 = eq(UInt<5>("hf"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_25 = mux(_io_dataOut_T_24, io_txErrorCount, _io_dataOut_T_23) @[Mux.scala 80:57]
    node _io_dataOut_T_26 = eq(UInt<5>("he"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_27 = mux(_io_dataOut_T_26, io_rxErrorCount, _io_dataOut_T_25) @[Mux.scala 80:57]
    node _io_dataOut_T_28 = eq(UInt<5>("hd"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_29 = mux(_io_dataOut_T_28, io_errorWarningLimit, _io_dataOut_T_27) @[Mux.scala 80:57]
    node _io_dataOut_T_30 = eq(UInt<5>("hc"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_31 = mux(_io_dataOut_T_30, io_errorCaptureCode, _io_dataOut_T_29) @[Mux.scala 80:57]
    node _io_dataOut_T_32 = eq(UInt<5>("hb"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_33 = mux(_io_dataOut_T_32, _io_dataOut_T_2, _io_dataOut_T_31) @[Mux.scala 80:57]
    node _io_dataOut_T_34 = eq(UInt<5>("h7"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_35 = mux(_io_dataOut_T_34, busTiming1_m.io_dataOut, _io_dataOut_T_33) @[Mux.scala 80:57]
    node _io_dataOut_T_36 = eq(UInt<5>("h6"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_37 = mux(_io_dataOut_T_36, busTiming0_m.io_dataOut, _io_dataOut_T_35) @[Mux.scala 80:57]
    node _io_dataOut_T_38 = eq(UInt<5>("h4"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_39 = mux(_io_dataOut_T_38, irqEnExt, _io_dataOut_T_37) @[Mux.scala 80:57]
    node _io_dataOut_T_40 = eq(UInt<5>("h3"), _io_dataOut_T) @[Mux.scala 80:60]
    reg busErrorIrq : UInt<1>, clock with :
      reset => (UInt<1>("h0"), busErrorIrq) @[CanRegisters.scala 352:35]
    reg arbitrationLostIrq : UInt<1>, clock with :
      reset => (UInt<1>("h0"), arbitrationLostIrq) @[CanRegisters.scala 353:42]
    node irqReg_hi_hi = cat(busErrorIrq, arbitrationLostIrq) @[Cat.scala 30:58]
    reg errorPassiveIrq : UInt<1>, clock with :
      reset => (UInt<1>("h0"), errorPassiveIrq) @[CanRegisters.scala 354:39]
    node irqReg_hi_lo = cat(errorPassiveIrq, UInt<1>("h0")) @[Cat.scala 30:58]
    node irqReg_hi = cat(irqReg_hi_hi, irqReg_hi_lo) @[Cat.scala 30:58]
    reg dataOverrunIrq : UInt<1>, clock with :
      reset => (UInt<1>("h0"), dataOverrunIrq) @[CanRegisters.scala 348:38]
    reg errorIrq : UInt<1>, clock with :
      reset => (UInt<1>("h0"), errorIrq) @[CanRegisters.scala 351:32]
    node irqReg_lo_hi = cat(dataOverrunIrq, errorIrq) @[Cat.scala 30:58]
    reg transmitIrq : UInt<1>, clock with :
      reset => (UInt<1>("h0"), transmitIrq) @[CanRegisters.scala 349:35]
    reg receiveIrq : UInt<1>, clock with :
      reset => (UInt<1>("h0"), receiveIrq) @[CanRegisters.scala 350:34]
    node irqReg_lo_lo = cat(transmitIrq, receiveIrq) @[Cat.scala 30:58]
    node irqReg_lo = cat(irqReg_lo_hi, irqReg_lo_lo) @[Cat.scala 30:58]
    node irqReg = cat(irqReg_hi, irqReg_lo) @[Cat.scala 30:58]
    skip
    node _io_dataOut_T_41 = mux(_io_dataOut_T_40, irqReg, _io_dataOut_T_39) @[Mux.scala 80:57]
    node _io_dataOut_T_42 = eq(UInt<5>("h2"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_43 = mux(_io_dataOut_T_42, status, _io_dataOut_T_41) @[Mux.scala 80:57]
    node _io_dataOut_T_44 = eq(UInt<5>("h0"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_45 = mux(_io_dataOut_T_44, _io_dataOut_T_1, _io_dataOut_T_43) @[Mux.scala 80:57]
    skip
    node io_dataOut_hi_1 = cat(UInt<3>("h1"), modeBasic_m.io_dataOut) @[Cat.scala 30:58]
    node _io_dataOut_T_47 = cat(io_dataOut_hi_1, mode) @[Cat.scala 30:58]
    node io_dataOut_lo = bits(irqReg, 3, 0) @[CanRegisters.scala 325:35]
    node _io_dataOut_T_48 = cat(UInt<4>("he"), io_dataOut_lo) @[Cat.scala 30:58]
    node _io_dataOut_T_49 = mux(io_resetMode, io_acceptanceCode_0, UInt<8>("hff")) @[CanRegisters.scala 326:17]
    node _io_dataOut_T_50 = mux(io_resetMode, io_acceptanceMask_0, UInt<8>("hff")) @[CanRegisters.scala 327:17]
    node _io_dataOut_T_51 = mux(io_resetMode, busTiming0_m.io_dataOut, UInt<8>("hff")) @[CanRegisters.scala 328:17]
    node _io_dataOut_T_52 = mux(io_resetMode, busTiming1_m.io_dataOut, UInt<8>("hff")) @[CanRegisters.scala 329:17]
    node _io_dataOut_T_53 = mux(io_resetMode, UInt<8>("hff"), io_txData_0) @[CanRegisters.scala 330:18]
    node _io_dataOut_T_54 = mux(io_resetMode, UInt<8>("hff"), io_txData_1) @[CanRegisters.scala 331:18]
    node _io_dataOut_T_55 = mux(io_resetMode, UInt<8>("hff"), io_txData_2) @[CanRegisters.scala 332:18]
    node _io_dataOut_T_56 = mux(io_resetMode, UInt<8>("hff"), io_txData_3) @[CanRegisters.scala 333:18]
    node _io_dataOut_T_57 = mux(io_resetMode, UInt<8>("hff"), io_txData_4) @[CanRegisters.scala 334:18]
    node _io_dataOut_T_58 = mux(io_resetMode, UInt<8>("hff"), io_txData_5) @[CanRegisters.scala 335:18]
    node _io_dataOut_T_59 = mux(io_resetMode, UInt<8>("hff"), io_txData_6) @[CanRegisters.scala 336:18]
    node _io_dataOut_T_60 = mux(io_resetMode, UInt<8>("hff"), io_txData_7) @[CanRegisters.scala 337:18]
    node _io_dataOut_T_61 = mux(io_resetMode, UInt<8>("hff"), io_txData_8) @[CanRegisters.scala 338:18]
    node _io_dataOut_T_62 = mux(io_resetMode, UInt<8>("hff"), io_txData_9) @[CanRegisters.scala 339:18]
    skip
    skip
    skip
    node _io_dataOut_T_66 = mux(_io_dataOut_T_16, _io_dataOut_T_62, _io_dataOut_T_5) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_68 = mux(_io_dataOut_T_18, _io_dataOut_T_61, _io_dataOut_T_66) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_70 = mux(_io_dataOut_T_20, _io_dataOut_T_60, _io_dataOut_T_68) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_72 = mux(_io_dataOut_T_22, _io_dataOut_T_59, _io_dataOut_T_70) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_74 = mux(_io_dataOut_T_24, _io_dataOut_T_58, _io_dataOut_T_72) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_76 = mux(_io_dataOut_T_26, _io_dataOut_T_57, _io_dataOut_T_74) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_78 = mux(_io_dataOut_T_28, _io_dataOut_T_56, _io_dataOut_T_76) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_80 = mux(_io_dataOut_T_30, _io_dataOut_T_55, _io_dataOut_T_78) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_82 = mux(_io_dataOut_T_32, _io_dataOut_T_54, _io_dataOut_T_80) @[Mux.scala 80:57]
    node _io_dataOut_T_83 = eq(UInt<5>("ha"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_84 = mux(_io_dataOut_T_83, _io_dataOut_T_53, _io_dataOut_T_82) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_86 = mux(_io_dataOut_T_34, _io_dataOut_T_52, _io_dataOut_T_84) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_88 = mux(_io_dataOut_T_36, _io_dataOut_T_51, _io_dataOut_T_86) @[Mux.scala 80:57]
    node _io_dataOut_T_89 = eq(UInt<5>("h5"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_90 = mux(_io_dataOut_T_89, _io_dataOut_T_50, _io_dataOut_T_88) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_92 = mux(_io_dataOut_T_38, _io_dataOut_T_49, _io_dataOut_T_90) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_94 = mux(_io_dataOut_T_40, _io_dataOut_T_48, _io_dataOut_T_92) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_96 = mux(_io_dataOut_T_42, status, _io_dataOut_T_94) @[Mux.scala 80:57]
    node _io_dataOut_T_97 = eq(UInt<5>("h1"), _io_dataOut_T) @[Mux.scala 80:60]
    node _io_dataOut_T_98 = mux(_io_dataOut_T_97, UInt<8>("hff"), _io_dataOut_T_96) @[Mux.scala 80:57]
    skip
    node _io_dataOut_T_100 = mux(_io_dataOut_T_44, _io_dataOut_T_47, _io_dataOut_T_98) @[Mux.scala 80:57]
    skip
    node dataOverrunIrqEn = mux(io_extendedMode, dataOverrunIrqEnExt, overrunIrqEnBasic) @[CanRegisters.scala 343:26]
    node errorWarningIrqEn = mux(io_extendedMode, errorWarningIrqEnExt, errorIrqEnBasic) @[CanRegisters.scala 344:27]
    node transmitIrqEn = mux(io_extendedMode, transmitIrqEnExt, transmitIrqEnBasic) @[CanRegisters.scala 345:23]
    node receiveIrqEn = mux(io_extendedMode, receiveIrqEnExt, receiveIrqEnBasic) @[CanRegisters.scala 346:22]
    skip
    skip
    skip
    node _T_23 = and(_T_14, dataOverrunIrqEn) @[CanRegisters.scala 356:31]
    node _T_24 = or(io_resetMode, readIrqReg) @[CanRegisters.scala 358:27]
    node _GEN_15 = mux(_T_24, UInt<1>("h0"), dataOverrunIrq) @[CanRegisters.scala 358:41 359:20 348:38]
    node _GEN_16 = or(_T_23, _GEN_15) @[CanRegisters.scala 356:51 357:20]
    skip
    node _T_26 = not(transmitBufferStatusQ) @[CanRegisters.scala 364:37]
    node _T_27 = and(transmitBufferStatus, _T_26) @[CanRegisters.scala 364:35]
    skip
    node _T_28 = and(_T_27, transmitIrqEn) @[CanRegisters.scala 364:60]
    node _GEN_17 = or(_T_28, transmitIrq) @[CanRegisters.scala 364:77 365:17 349:35]
    node _GEN_18 = mux(_T_24, UInt<1>("h0"), _GEN_17) @[CanRegisters.scala 362:35 363:17]
    skip
    node _T_30 = not(receiveIrq) @[CanRegisters.scala 368:27]
    node _T_31 = and(_T_17, _T_30) @[CanRegisters.scala 368:24]
    skip
    node _T_32 = and(_T_31, receiveIrqEn) @[CanRegisters.scala 368:40]
    skip
    node _GEN_19 = mux(_T_16, UInt<1>("h0"), receiveIrq) @[CanRegisters.scala 370:48 371:16 350:34]
    node _GEN_20 = or(_T_32, _GEN_19) @[CanRegisters.scala 368:56 369:16]
    node _T_34 = xor(io_errorStatus, errorStatusQ) @[CanRegisters.scala 374:25]
    node _T_35 = xor(io_nodeBusOff, nodeBusOffQ) @[CanRegisters.scala 374:58]
    node _T_36 = or(_T_34, _T_35) @[CanRegisters.scala 374:41]
    skip
    node _T_37 = and(_T_36, errorWarningIrqEn) @[CanRegisters.scala 374:74]
    node _GEN_21 = mux(readIrqReg, UInt<1>("h0"), errorIrq) @[CanRegisters.scala 376:26 377:14 351:32]
    node _GEN_22 = or(_T_37, _GEN_21) @[CanRegisters.scala 374:95 375:14]
    node _T_38 = and(io_setBusErrorIrq, busErrorIrqEn) @[CanRegisters.scala 380:26]
    skip
    node _GEN_23 = mux(_T_24, UInt<1>("h0"), busErrorIrq) @[CanRegisters.scala 382:41 383:17 352:35]
    node _GEN_24 = or(_T_38, _GEN_23) @[CanRegisters.scala 380:43 381:17]
    node _T_40 = and(io_setArbitrationLostIrq, arbitrationLostIrqEn) @[CanRegisters.scala 386:33]
    skip
    node _GEN_25 = mux(_T_24, UInt<1>("h0"), arbitrationLostIrq) @[CanRegisters.scala 388:41 389:24 353:42]
    node _GEN_26 = or(_T_40, _GEN_25) @[CanRegisters.scala 386:57 387:24]
    node _T_42 = not(nodeErrorPassiveQ) @[CanRegisters.scala 392:32]
    node _T_43 = and(io_nodeErrorPassive, _T_42) @[CanRegisters.scala 392:29]
    node _T_44 = not(io_nodeErrorPassive) @[CanRegisters.scala 392:55]
    node _T_45 = and(_T_44, nodeErrorPassiveQ) @[CanRegisters.scala 392:77]
    node _T_46 = and(_T_45, io_nodeErrorActive) @[CanRegisters.scala 392:97]
    node _T_47 = or(_T_43, _T_46) @[CanRegisters.scala 392:52]
    node _T_48 = and(_T_47, errorPassiveIrqEn) @[CanRegisters.scala 392:119]
    skip
    node _GEN_27 = mux(_T_24, UInt<1>("h0"), errorPassiveIrq) @[CanRegisters.scala 394:41 395:21 354:39]
    node _GEN_28 = or(_T_48, _GEN_27) @[CanRegisters.scala 392:140 393:21]
    node irq = orr(irqReg) @[CanRegisters.scala 399:20]
    node _T_50 = or(readIrqReg, io_releaseBuffer) @[CanRegisters.scala 401:19]
    skip
    node _GEN_29 = mux(irq, UInt<1>("h0"), irqN) @[CanRegisters.scala 403:19 404:10 100:27]
    node _GEN_30 = or(_T_50, _GEN_29) @[CanRegisters.scala 401:39 402:10]
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    io_dataOut <= mux(io_extendedMode, _io_dataOut_T_45, _io_dataOut_T_100) @[CanRegisters.scala 297:20]
    io_irqN <= irqN @[CanRegisters.scala 101:11]
    io_resetMode <= mode_m.io_dataOut @[CanRegisters.scala 160:93]
    io_listenOnlyMode <= and(io_extendedMode, _io_listenOnlyMode_T) @[CanRegisters.scala 169:40]
    io_acceptanceFilterMode <= and(io_extendedMode, _io_acceptanceFilterMode_T) @[CanRegisters.scala 171:46]
    io_selfTestMode <= and(io_extendedMode, _io_selfTestMode_T) @[CanRegisters.scala 170:38]
    io_clearDataOverrun <= command_3_m.io_dataOut @[CanRegisters.scala 173:32 177:14]
    io_releaseBuffer <= command_2_m.io_dataOut @[CanRegisters.scala 173:32 176:14]
    io_abortTx <= and(command_1, _io_abortTx_T) @[CanRegisters.scala 189:28]
    io_txRequest <= or(command_0, command_4) @[CanRegisters.scala 188:30]
    io_selfRxRequest <= selfRxRequest @[CanRegisters.scala 103:20]
    io_singleShotTransmission <= singleShotTransmission @[CanRegisters.scala 105:29]
    skip
    io_readArbitrationLostCaptureReg <= and(_io_readArbitrationLostCaptureReg_T, _io_readArbitrationLostCaptureReg_T_1) @[CanRegisters.scala 134:62]
    io_readErrorCodeCaptureReg <= and(_io_readArbitrationLostCaptureReg_T, _io_readErrorCodeCaptureReg_T_1) @[CanRegisters.scala 135:56]
    io_baudRatePrescaler <= bits(busTiming0_m.io_dataOut, 5, 0) @[CanRegisters.scala 252:37]
    io_syncJumpWidth <= bits(busTiming0_m.io_dataOut, 7, 6) @[CanRegisters.scala 253:33]
    io_timeSegment1 <= bits(busTiming1_m.io_dataOut, 3, 0) @[CanRegisters.scala 257:32]
    io_timeSegment2 <= bits(busTiming1_m.io_dataOut, 6, 4) @[CanRegisters.scala 258:32]
    io_tripleSampling <= bits(busTiming1_m.io_dataOut, 7, 7) @[CanRegisters.scala 259:34]
    io_errorWarningLimit <= io_errorWarningLimit_m.io_dataOut @[CanRegisters.scala 261:24]
    io_writeEnReceiveErrorCounter <= and(_io_writeEnReceiveErrorCounter_T_3, io_resetMode) @[CanRegisters.scala 157:55]
    io_writeEnTransmitErrorCounter <= and(_io_writeEnTransmitErrorCounter_T_3, io_resetMode) @[CanRegisters.scala 158:56]
    io_extendedMode <= io_extendedMode_m.io_dataOut @[CanRegisters.scala 263:76]
    io_clkout <= mux(_io_clkout_T, _io_clkout_T_7, _io_clkout_T_8) @[CanRegisters.scala 281:31 282:16 285:15]
    io_acceptanceCode_0 <= io_acceptanceCode_0_m.io_dataOut @[CanRegisters.scala 289:26]
    io_acceptanceCode_1 <= io_acceptanceCode_1_m.io_dataOut @[CanRegisters.scala 289:26]
    io_acceptanceCode_2 <= io_acceptanceCode_2_m.io_dataOut @[CanRegisters.scala 289:26]
    io_acceptanceCode_3 <= io_acceptanceCode_3_m.io_dataOut @[CanRegisters.scala 289:26]
    io_acceptanceMask_0 <= io_acceptanceMask_0_m.io_dataOut @[CanRegisters.scala 290:26]
    io_acceptanceMask_1 <= io_acceptanceMask_1_m.io_dataOut @[CanRegisters.scala 290:26]
    io_acceptanceMask_2 <= io_acceptanceMask_2_m.io_dataOut @[CanRegisters.scala 290:26]
    io_acceptanceMask_3 <= io_acceptanceMask_3_m.io_dataOut @[CanRegisters.scala 290:26]
    io_txData_0 <= io_txData_0_m.io_dataOut @[CanRegisters.scala 294:18]
    io_txData_1 <= io_txData_1_m.io_dataOut @[CanRegisters.scala 294:18]
    io_txData_2 <= io_txData_2_m.io_dataOut @[CanRegisters.scala 294:18]
    io_txData_3 <= io_txData_3_m.io_dataOut @[CanRegisters.scala 294:18]
    io_txData_4 <= io_txData_4_m.io_dataOut @[CanRegisters.scala 294:18]
    io_txData_5 <= io_txData_5_m.io_dataOut @[CanRegisters.scala 294:18]
    io_txData_6 <= io_txData_6_m.io_dataOut @[CanRegisters.scala 294:18]
    io_txData_7 <= io_txData_7_m.io_dataOut @[CanRegisters.scala 294:18]
    io_txData_8 <= io_txData_8_m.io_dataOut @[CanRegisters.scala 294:18]
    io_txData_9 <= io_txData_9_m.io_dataOut @[CanRegisters.scala 294:18]
    io_txData_10 <= io_txData_10_m.io_dataOut @[CanRegisters.scala 294:18]
    io_txData_11 <= io_txData_11_m.io_dataOut @[CanRegisters.scala 294:18]
    io_txData_12 <= io_txData_12_m.io_dataOut @[CanRegisters.scala 294:18]
    irqN <= or(reset, _GEN_30) @[CanRegisters.scala 100:{27,27}]
    selfRxRequest <= mux(reset, UInt<1>("h0"), _GEN_1) @[CanRegisters.scala 102:{36,36}]
    singleShotTransmission <= mux(reset, UInt<1>("h0"), _GEN_3) @[CanRegisters.scala 104:{45,45}]
    txSuccessfulQ <= io_txSuccessful @[CanRegisters.scala 106:36]
    overrunQ <= io_overrun @[CanRegisters.scala 107:31]
    overrunStatus <= mux(reset, UInt<1>("h0"), _GEN_9) @[CanRegisters.scala 108:{36,36}]
    transmissionComplete <= or(reset, _GEN_7) @[CanRegisters.scala 109:{43,43}]
    transmitBufferStatus <= or(reset, _GEN_5) @[CanRegisters.scala 110:{43,43}]
    transmitBufferStatusQ <= transmitBufferStatus @[CanRegisters.scala 111:44]
    receiveBufferStatus <= mux(reset, UInt<1>("h0"), _GEN_11) @[CanRegisters.scala 112:{42,42}]
    errorStatusQ <= io_errorStatus @[CanRegisters.scala 113:35]
    nodeBusOffQ <= io_nodeBusOff @[CanRegisters.scala 114:34]
    nodeErrorPassiveQ <= io_nodeErrorPassive @[CanRegisters.scala 115:40]
    mode_m.clock <= clock
    mode_m.reset <= reset
    mode_m.io_resetSync <= io_setResetMode @[CanRegisterDoubleReset.scala 27:20]
    mode_m.io_dataIn <= bits(io_dataIn, 0, 0) @[CanRegisters.scala 160:70]
    mode_m.io_writeEn <= and(_writeEnMode_T, _writeEnMode_T_1) @[CanRegisters.scala 97:24]
    modeBasic_m.clock <= clock
    modeBasic_m.reset <= reset
    modeBasic_m.io_dataIn <= bits(io_dataIn, 4, 1) @[CanRegisters.scala 161:48]
    modeBasic_m.io_writeEn <= and(_writeEnMode_T, _writeEnMode_T_1) @[CanRegisters.scala 97:24]
    modeExt_m.clock <= clock
    modeExt_m.reset <= reset
    modeExt_m.io_dataIn <= bits(io_dataIn, 3, 1) @[CanRegisters.scala 162:46]
    modeExt_m.io_writeEn <= and(writeEnMode, io_resetMode) @[CanRegisters.scala 162:64]
    command_0_m.clock <= clock
    command_0_m.reset <= reset
    command_0_m.io_resetSync <= or(_command_0_T, io_resetMode) @[CanRegisters.scala 174:70]
    command_0_m.io_dataIn <= bits(io_dataIn, 0, 0) @[CanRegisters.scala 174:94]
    command_0_m.io_writeEn <= and(_writeEnMode_T, _writeEnCommand_T_1) @[CanRegisters.scala 97:24]
    command_1_m.clock <= clock
    command_1_m.reset <= reset
    command_1_m.io_resetSync <= or(_command_1_T_3, io_resetMode) @[CanRegisters.scala 175:108]
    command_1_m.io_dataIn <= bits(io_dataIn, 1, 1) @[CanRegisters.scala 175:132]
    command_1_m.io_writeEn <= and(_writeEnMode_T, _writeEnCommand_T_1) @[CanRegisters.scala 97:24]
    command_2_m.clock <= clock
    command_2_m.reset <= reset
    command_2_m.io_resetSync <= or(_command_2_T, io_resetMode) @[CanRegisters.scala 176:66]
    command_2_m.io_dataIn <= bits(io_dataIn, 2, 2) @[CanRegisters.scala 176:90]
    command_2_m.io_writeEn <= and(_writeEnMode_T, _writeEnCommand_T_1) @[CanRegisters.scala 97:24]
    command_3_m.clock <= clock
    command_3_m.reset <= reset
    command_3_m.io_resetSync <= or(_command_2_T, io_resetMode) @[CanRegisters.scala 177:66]
    command_3_m.io_dataIn <= bits(io_dataIn, 3, 3) @[CanRegisters.scala 177:90]
    command_3_m.io_writeEn <= and(_writeEnMode_T, _writeEnCommand_T_1) @[CanRegisters.scala 97:24]
    command_4_m.clock <= clock
    command_4_m.reset <= reset
    command_4_m.io_resetSync <= or(_command_4_T, io_resetMode) @[CanRegisters.scala 178:70]
    command_4_m.io_dataIn <= bits(io_dataIn, 4, 4) @[CanRegisters.scala 178:94]
    command_4_m.io_writeEn <= and(_writeEnMode_T, _writeEnCommand_T_1) @[CanRegisters.scala 97:24]
    overloadFrameQ <= mux(reset, UInt<1>("h0"), io_overloadFrame) @[CanRegisters.scala 198:{38,38,38}]
    io_overloadRequest_m.clock <= clock
    io_overloadRequest_m.reset <= reset
    io_overloadRequest_m.io_resetSync <= and(io_overloadFrame, _io_overloadRequest_T) @[CanRegisters.scala 200:67]
    io_overloadRequest_m.io_dataIn <= bits(io_dataIn, 5, 5) @[CanRegisters.scala 200:94]
    io_overloadRequest_m.io_writeEn <= and(_writeEnMode_T, _writeEnCommand_T_1) @[CanRegisters.scala 97:24]
    irqEnExt_m.clock <= clock
    irqEnExt_m.reset <= reset
    irqEnExt_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    irqEnExt_m.io_writeEn <= and(_writeEnInterruptEnable_T_2, io_extendedMode) @[CanRegisters.scala 89:59]
    busTiming0_m.clock <= clock
    busTiming0_m.reset <= reset
    busTiming0_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    busTiming0_m.io_writeEn <= and(_writeEnBusTiming0_T_2, io_resetMode) @[CanRegisters.scala 127:48]
    busTiming1_m.clock <= clock
    busTiming1_m.reset <= reset
    busTiming1_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    busTiming1_m.io_writeEn <= and(_writeEnBusTiming1_T_2, io_resetMode) @[CanRegisters.scala 128:48]
    io_errorWarningLimit_m.clock <= clock
    io_errorWarningLimit_m.reset <= reset
    io_errorWarningLimit_m.io_dataIn <= io_dataIn @[CanRegister.scala 31:17]
    io_errorWarningLimit_m.io_writeEn <= and(_writeEnErrorWarningLimit_T_3, io_resetMode) @[CanRegisters.scala 156:59]
    io_extendedMode_m.clock <= clock
    io_extendedMode_m.reset <= reset
    io_extendedMode_m.io_dataIn <= bits(io_dataIn, 7, 7) @[CanRegisters.scala 263:45]
    io_extendedMode_m.io_writeEn <= and(writeEnClockDivLow, io_resetMode) @[CanRegisters.scala 130:54]
    clockOff_m.clock <= clock
    clockOff_m.reset <= reset
    clockOff_m.io_dataIn <= bits(io_dataIn, 3, 3) @[CanRegisters.scala 264:48]
    clockOff_m.io_writeEn <= and(writeEnClockDivLow, io_resetMode) @[CanRegisters.scala 130:54]
    cd_m.clock <= clock
    cd_m.reset <= reset
    cd_m.io_dataIn <= bits(io_dataIn, 2, 0) @[CanRegisters.scala 265:42]
    cd_m.io_writeEn <= and(_writeEnMode_T, _writeEnClockDivLow_T_1) @[CanRegisters.scala 97:24]
    clkoutCnt <= mux(reset, UInt<3>("h0"), _GEN_12) @[CanRegisters.scala 268:{33,33}]
    clkoutTmp <= mux(reset, UInt<1>("h0"), _GEN_13) @[CanRegisters.scala 269:{33,33}]
    io_acceptanceCode_0_m.clock <= clock
    io_acceptanceCode_0_m.reset <= reset
    io_acceptanceCode_0_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_acceptanceCode_0_m.io_writeEn <= and(_writeEnAcceptanceCode_0_T_7, io_resetMode) @[CanRegisters.scala 139:57]
    io_acceptanceMask_0_m.clock <= clock
    io_acceptanceMask_0_m.reset <= reset
    io_acceptanceMask_0_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_acceptanceMask_0_m.io_writeEn <= and(_writeEnAcceptanceMask_0_T_7, io_resetMode) @[CanRegisters.scala 140:57]
    io_acceptanceCode_1_m.clock <= clock
    io_acceptanceCode_1_m.reset <= reset
    io_acceptanceCode_1_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_acceptanceCode_1_m.io_writeEn <= and(_writeEnAcceptanceCode_1_T_3, io_resetMode) @[CanRegisters.scala 143:56]
    io_acceptanceMask_1_m.clock <= clock
    io_acceptanceMask_1_m.reset <= reset
    io_acceptanceMask_1_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_acceptanceMask_1_m.io_writeEn <= and(_writeEnAcceptanceMask_1_T_3, io_resetMode) @[CanRegisters.scala 144:56]
    io_acceptanceCode_2_m.clock <= clock
    io_acceptanceCode_2_m.reset <= reset
    io_acceptanceCode_2_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_acceptanceCode_2_m.io_writeEn <= and(_writeEnAcceptanceCode_2_T_3, io_resetMode) @[CanRegisters.scala 143:56]
    io_acceptanceMask_2_m.clock <= clock
    io_acceptanceMask_2_m.reset <= reset
    io_acceptanceMask_2_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_acceptanceMask_2_m.io_writeEn <= and(_writeEnAcceptanceMask_2_T_3, io_resetMode) @[CanRegisters.scala 144:56]
    io_acceptanceCode_3_m.clock <= clock
    io_acceptanceCode_3_m.reset <= reset
    io_acceptanceCode_3_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_acceptanceCode_3_m.io_writeEn <= and(_writeEnAcceptanceCode_3_T_3, io_resetMode) @[CanRegisters.scala 143:56]
    io_acceptanceMask_3_m.clock <= clock
    io_acceptanceMask_3_m.reset <= reset
    io_acceptanceMask_3_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_acceptanceMask_3_m.io_writeEn <= and(_writeEnAcceptanceMask_3_T_3, io_resetMode) @[CanRegisters.scala 144:56]
    io_txData_0_m.clock <= clock
    io_txData_0_m.reset <= reset
    io_txData_0_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_txData_0_m.io_writeEn <= and(_writeEnTxData_0_T_9, transmitBufferStatus) @[CanRegisters.scala 150:76]
    io_txData_1_m.clock <= clock
    io_txData_1_m.reset <= reset
    io_txData_1_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_txData_1_m.io_writeEn <= and(_writeEnTxData_1_T_9, transmitBufferStatus) @[CanRegisters.scala 150:76]
    io_txData_2_m.clock <= clock
    io_txData_2_m.reset <= reset
    io_txData_2_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_txData_2_m.io_writeEn <= and(_writeEnTxData_2_T_9, transmitBufferStatus) @[CanRegisters.scala 150:76]
    io_txData_3_m.clock <= clock
    io_txData_3_m.reset <= reset
    io_txData_3_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_txData_3_m.io_writeEn <= and(_writeEnTxData_3_T_9, transmitBufferStatus) @[CanRegisters.scala 150:76]
    io_txData_4_m.clock <= clock
    io_txData_4_m.reset <= reset
    io_txData_4_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_txData_4_m.io_writeEn <= and(_writeEnTxData_4_T_9, transmitBufferStatus) @[CanRegisters.scala 150:76]
    io_txData_5_m.clock <= clock
    io_txData_5_m.reset <= reset
    io_txData_5_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_txData_5_m.io_writeEn <= and(_writeEnTxData_5_T_9, transmitBufferStatus) @[CanRegisters.scala 150:76]
    io_txData_6_m.clock <= clock
    io_txData_6_m.reset <= reset
    io_txData_6_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_txData_6_m.io_writeEn <= and(_writeEnTxData_6_T_9, transmitBufferStatus) @[CanRegisters.scala 150:76]
    io_txData_7_m.clock <= clock
    io_txData_7_m.reset <= reset
    io_txData_7_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_txData_7_m.io_writeEn <= and(_writeEnTxData_7_T_9, transmitBufferStatus) @[CanRegisters.scala 150:76]
    io_txData_8_m.clock <= clock
    io_txData_8_m.reset <= reset
    io_txData_8_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_txData_8_m.io_writeEn <= and(_writeEnTxData_8_T_9, transmitBufferStatus) @[CanRegisters.scala 150:76]
    io_txData_9_m.clock <= clock
    io_txData_9_m.reset <= reset
    io_txData_9_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_txData_9_m.io_writeEn <= and(_writeEnTxData_9_T_9, transmitBufferStatus) @[CanRegisters.scala 150:76]
    io_txData_10_m.clock <= clock
    io_txData_10_m.reset <= reset
    io_txData_10_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_txData_10_m.io_writeEn <= and(_writeEnTxData_10_T_5, transmitBufferStatus) @[CanRegisters.scala 152:66]
    io_txData_11_m.clock <= clock
    io_txData_11_m.reset <= reset
    io_txData_11_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_txData_11_m.io_writeEn <= and(_writeEnTxData_11_T_5, transmitBufferStatus) @[CanRegisters.scala 152:66]
    io_txData_12_m.clock <= clock
    io_txData_12_m.reset <= reset
    io_txData_12_m.io_dataIn <= io_dataIn @[CanRegister.scala 25:17]
    io_txData_12_m.io_writeEn <= and(_writeEnTxData_12_T_5, transmitBufferStatus) @[CanRegisters.scala 152:66]
    dataOverrunIrq <= mux(reset, UInt<1>("h0"), _GEN_16) @[CanRegisters.scala 348:{38,38}]
    transmitIrq <= mux(reset, UInt<1>("h0"), _GEN_18) @[CanRegisters.scala 349:{35,35}]
    receiveIrq <= mux(reset, UInt<1>("h0"), _GEN_20) @[CanRegisters.scala 350:{34,34}]
    errorIrq <= mux(reset, UInt<1>("h0"), _GEN_22) @[CanRegisters.scala 351:{32,32}]
    busErrorIrq <= mux(reset, UInt<1>("h0"), _GEN_24) @[CanRegisters.scala 352:{35,35}]
    arbitrationLostIrq <= mux(reset, UInt<1>("h0"), _GEN_26) @[CanRegisters.scala 353:{42,42}]
    errorPassiveIrq <= mux(reset, UInt<1>("h0"), _GEN_28) @[CanRegisters.scala 354:{39,39}]

  module CanBtl :
    input clock : Clock
    input reset : UInt<1>
    input io_rx : UInt<1>
    input io_tx : UInt<1>
    input io_baudRatePrescaler : UInt<6>
    input io_syncJumpWidth : UInt<2>
    input io_timeSegment1 : UInt<4>
    input io_timeSegment2 : UInt<3>
    input io_tripleSampling : UInt<1>
    input io_rxIdle : UInt<1>
    input io_rxInter : UInt<1>
    input io_transmitting : UInt<1>
    input io_transmitter : UInt<1>
    input io_goRxInter : UInt<1>
    input io_txNext : UInt<1>
    input io_goOverloadFrame : UInt<1>
    input io_goErrorFrame : UInt<1>
    input io_goTx : UInt<1>
    input io_sendAck : UInt<1>
    input io_nodeErrorPassive : UInt<1>
    output io_samplePoint : UInt<1>
    output io_sampledBit : UInt<1>
    output io_sampledBitQ : UInt<1>
    output io_txPoint : UInt<1>
    output io_hardSync : UInt<1>

    reg samplePoint : UInt<1>, clock with :
      reset => (UInt<1>("h0"), samplePoint) @[CanBtl.scala 95:35]
    reg txPoint : UInt<1>, clock with :
      reset => (UInt<1>("h0"), txPoint) @[CanBtl.scala 97:31]
    reg sampledBit : UInt<1>, clock with :
      reset => (UInt<1>("h0"), sampledBit) @[CanBtl.scala 99:34]
    reg sampledBitQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), sampledBitQ) @[CanBtl.scala 101:35]
    reg clockCount : UInt<7>, clock with :
      reset => (UInt<1>("h0"), clockCount) @[CanBtl.scala 104:34]
    reg clockEn : UInt<1>, clock with :
      reset => (UInt<1>("h0"), clockEn) @[CanBtl.scala 106:31]
    reg clockEnQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), clockEnQ) @[CanBtl.scala 108:32]
    reg syncBlocked : UInt<1>, clock with :
      reset => (UInt<1>("h0"), syncBlocked) @[CanBtl.scala 110:35]
    reg hardSyncBlocked : UInt<1>, clock with :
      reset => (UInt<1>("h0"), hardSyncBlocked) @[CanBtl.scala 112:39]
    reg quantaCounter : UInt<5>, clock with :
      reset => (UInt<1>("h0"), quantaCounter) @[CanBtl.scala 114:37]
    reg delay : UInt<4>, clock with :
      reset => (UInt<1>("h0"), delay) @[CanBtl.scala 116:29]
    reg sync : UInt<1>, clock with :
      reset => (UInt<1>("h0"), sync) @[CanBtl.scala 118:28]
    reg seg1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), seg1) @[CanBtl.scala 120:28]
    reg seg2 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), seg2) @[CanBtl.scala 122:28]
    reg resyncLatched : UInt<1>, clock with :
      reset => (UInt<1>("h0"), resyncLatched) @[CanBtl.scala 124:37]
    reg sample : UInt<2>, clock with :
      reset => (UInt<1>("h0"), sample) @[CanBtl.scala 126:30]
    reg txNextSp : UInt<1>, clock with :
      reset => (UInt<1>("h0"), txNextSp) @[CanBtl.scala 128:31]
    node _goSeg1_T = or(sync, io_hardSync) @[CanBtl.scala 138:40]
    node _resync_T = not(io_rxIdle) @[CanBtl.scala 148:13]
    node _resync_T_1 = not(io_rxInter) @[CanBtl.scala 148:26]
    node _resync_T_2 = and(_resync_T, _resync_T_1) @[CanBtl.scala 148:24]
    node _resync_T_3 = not(io_rx) @[CanBtl.scala 148:40]
    node _resync_T_4 = and(_resync_T_2, _resync_T_3) @[CanBtl.scala 148:38]
    node _resync_T_5 = and(_resync_T_4, io_sampledBit) @[CanBtl.scala 148:47]
    node _resync_T_6 = not(syncBlocked) @[CanBtl.scala 148:65]
    node resync = and(_resync_T_5, _resync_T_6) @[CanBtl.scala 148:63]
    skip
    node _goSeg1_T_1 = and(resync, seg2) @[CanBtl.scala 138:64]
    node _syncWindow_T = bits(quantaCounter, 2, 0) @[CanBtl.scala 151:50]
    node _syncWindow_T_1 = sub(io_timeSegment2, _syncWindow_T) @[CanBtl.scala 151:35]
    node _syncWindow_T_2 = tail(_syncWindow_T_1, 1) @[CanBtl.scala 151:35]
    node _GEN_32 = pad(io_syncJumpWidth, 4) @[CanBtl.scala 151:77]
    node _syncWindow_T_3 = add(_GEN_32, UInt<4>("h1")) @[CanBtl.scala 151:77]
    node _syncWindow_T_4 = tail(_syncWindow_T_3, 1) @[CanBtl.scala 151:77]
    node _GEN_33 = pad(_syncWindow_T_2, 4) @[CanBtl.scala 151:57]
    node syncWindow = lt(_GEN_33, _syncWindow_T_4) @[CanBtl.scala 151:57]
    skip
    node _goSeg1_T_2 = and(_goSeg1_T_1, syncWindow) @[CanBtl.scala 138:71]
    node _goSeg1_T_3 = or(_goSeg1_T, _goSeg1_T_2) @[CanBtl.scala 138:54]
    node _goSeg1_T_4 = and(resyncLatched, syncWindow) @[CanBtl.scala 138:102]
    node _goSeg1_T_5 = or(_goSeg1_T_3, _goSeg1_T_4) @[CanBtl.scala 138:85]
    node goSeg1 = and(clockEnQ, _goSeg1_T_5) @[CanBtl.scala 138:32]
    node _goSeg2_T = not(io_hardSync) @[CanBtl.scala 141:43]
    node _goSeg2_T_1 = and(seg1, _goSeg2_T) @[CanBtl.scala 141:40]
    node _goSeg2_T_2 = add(io_timeSegment1, delay) @[CanBtl.scala 141:95]
    node _goSeg2_T_3 = eq(quantaCounter, _goSeg2_T_2) @[CanBtl.scala 141:74]
    node _goSeg2_T_4 = and(_goSeg2_T_1, _goSeg2_T_3) @[CanBtl.scala 141:57]
    node goSeg2 = and(clockEnQ, _goSeg2_T_4) @[CanBtl.scala 141:32]
    node _prescalerLimit_T = add(io_baudRatePrescaler, UInt<6>("h1")) @[CanBtl.scala 143:43]
    node _prescalerLimit_T_1 = tail(_prescalerLimit_T, 1) @[CanBtl.scala 143:43]
    node _prescalerLimit_T_2 = shl(_prescalerLimit_T_1, 1) @[CanBtl.scala 143:50]
    node _io_hardSync_T = or(io_rxIdle, io_rxInter) @[CanBtl.scala 146:29]
    skip
    node _io_hardSync_T_2 = and(_io_hardSync_T, _resync_T_3) @[CanBtl.scala 146:43]
    node _io_hardSync_T_3 = and(_io_hardSync_T_2, io_sampledBit) @[CanBtl.scala 146:54]
    node _io_hardSync_T_4 = not(hardSyncBlocked) @[CanBtl.scala 146:73]
    skip
    node _goSync_T = and(clockEnQ, seg2) @[CanBtl.scala 149:22]
    skip
    node _goSync_T_2 = eq(_syncWindow_T, io_timeSegment2) @[CanBtl.scala 149:51]
    node _goSync_T_3 = and(_goSync_T, _goSync_T_2) @[CanBtl.scala 149:29]
    skip
    node _goSync_T_5 = and(_goSync_T_3, _goSeg2_T) @[CanBtl.scala 149:72]
    node _goSync_T_6 = not(resync) @[CanBtl.scala 149:92]
    node goSync = and(_goSync_T_5, _goSync_T_6) @[CanBtl.scala 149:89]
    node _txPoint_T = not(txPoint) @[CanBtl.scala 153:14]
    node _txPoint_T_1 = and(_txPoint_T, seg2) @[CanBtl.scala 153:23]
    skip
    skip
    node _txPoint_T_4 = and(clockEn, _goSync_T_2) @[CanBtl.scala 153:42]
    node _txPoint_T_5 = or(clockEn, clockEnQ) @[CanBtl.scala 153:96]
    node _txPoint_T_6 = or(resync, io_hardSync) @[CanBtl.scala 153:118]
    node _txPoint_T_7 = and(_txPoint_T_5, _txPoint_T_6) @[CanBtl.scala 153:108]
    node _txPoint_T_8 = or(_txPoint_T_4, _txPoint_T_7) @[CanBtl.scala 153:85]
    node _txPoint_T_9 = and(_txPoint_T_1, _txPoint_T_8) @[CanBtl.scala 153:30]
    node prescalerLimit = pad(_prescalerLimit_T_2, 8) @[CanBtl.scala 132:35 143:18]
    node _T = sub(prescalerLimit, UInt<8>("h1")) @[CanBtl.scala 155:38]
    node _T_1 = tail(_T, 1) @[CanBtl.scala 155:38]
    node _GEN_34 = pad(clockCount, 8) @[CanBtl.scala 155:19]
    node _T_2 = geq(_GEN_34, _T_1) @[CanBtl.scala 155:19]
    node _clockCount_T = add(clockCount, UInt<7>("h1")) @[CanBtl.scala 158:30]
    node _clockCount_T_1 = tail(_clockCount_T, 1) @[CanBtl.scala 158:30]
    node _GEN_0 = mux(_T_2, UInt<7>("h0"), _clockCount_T_1) @[CanBtl.scala 155:46 156:16 158:16]
    skip
    skip
    skip
    node _T_5 = eq(_GEN_34, _T_1) @[CanBtl.scala 161:19]
    skip
    skip
    node _T_7 = not(syncWindow) @[CanBtl.scala 167:24]
    node _T_8 = and(_goSeg1_T_1, _T_7) @[CanBtl.scala 167:22]
    node _GEN_2 = mux(goSeg1, UInt<1>("h0"), resyncLatched) @[CanBtl.scala 169:22 170:19 124:37]
    node _GEN_3 = or(_T_8, _GEN_2) @[CanBtl.scala 167:37 168:19]
    node sample_hi = bits(sample, 0, 0) @[CanBtl.scala 175:25]
    node _sample_T = cat(sample_hi, io_rx) @[Cat.scala 30:58]
    skip
    node _GEN_4 = mux(clockEnQ, goSync, sync) @[CanBtl.scala 173:18 174:10 118:28]
    node _GEN_5 = mux(clockEnQ, _sample_T, sample) @[CanBtl.scala 173:18 175:12 126:30]
    node _GEN_6 = mux(goSeg2, UInt<1>("h0"), seg1) @[CanBtl.scala 181:22 183:10 120:28]
    node _GEN_7 = or(goSeg1, _GEN_6) @[CanBtl.scala 178:16 180:10]
    node _T_9 = or(goSync, goSeg1) @[CanBtl.scala 188:21]
    node _GEN_8 = mux(_T_9, UInt<1>("h0"), seg2) @[CanBtl.scala 188:31 189:10 122:28]
    node _GEN_9 = or(goSeg2, _GEN_8) @[CanBtl.scala 186:16 187:10]
    skip
    node _T_11 = or(_T_9, goSeg2) @[CanBtl.scala 192:24]
    node _quantaCounter_T = add(quantaCounter, UInt<5>("h1")) @[CanBtl.scala 197:36]
    node _GEN_10 = mux(clockEnQ, _quantaCounter_T, pad(quantaCounter, 6)) @[CanBtl.scala 195:24 197:19 114:37]
    node _GEN_11 = mux(_T_11, UInt<6>("h0"), _GEN_10) @[CanBtl.scala 192:34 194:19]
    node _T_12 = and(resync, seg1) @[CanBtl.scala 200:15]
    node _T_13 = not(io_transmitting) @[CanBtl.scala 200:25]
    skip
    node _T_15 = and(io_tx, _resync_T_3) @[CanBtl.scala 200:81]
    node _T_16 = or(txNextSp, _T_15) @[CanBtl.scala 200:72]
    node _T_17 = and(io_transmitting, _T_16) @[CanBtl.scala 200:60]
    node _T_18 = or(_T_13, _T_17) @[CanBtl.scala 200:42]
    node _T_19 = and(_T_12, _T_18) @[CanBtl.scala 200:22]
    node _GEN_36 = pad(io_syncJumpWidth, 5) @[CanBtl.scala 201:32]
    node _delay_T = gt(quantaCounter, _GEN_36) @[CanBtl.scala 201:32]
    skip
    skip
    skip
    skip
    node _delay_T_4 = mux(_delay_T, pad(_syncWindow_T_4, 6), _quantaCounter_T) @[CanBtl.scala 201:17]
    skip
    node _GEN_12 = mux(_T_9, UInt<4>("h0"), delay) @[CanBtl.scala 203:30 204:11 116:29]
    node _GEN_13 = mux(_T_19, _delay_T_4, pad(_GEN_12, 6)) @[CanBtl.scala 200:94 201:11]
    skip
    node _T_22 = and(clockEnQ, _goSeg2_T) @[CanBtl.scala 211:22]
    skip
    skip
    node _T_25 = and(seg1, _goSeg2_T_3) @[CanBtl.scala 212:15]
    node _sampledBit_T = andr(sample) @[CanBtl.scala 214:34]
    skip
    node _sampledBit_T_2 = and(sample_hi, io_rx) @[CanBtl.scala 214:50]
    node _sampledBit_T_3 = or(_sampledBit_T, _sampledBit_T_2) @[CanBtl.scala 214:37]
    node _sampledBit_T_4 = bits(sample, 1, 1) @[CanBtl.scala 214:68]
    node _sampledBit_T_5 = and(_sampledBit_T_4, io_rx) @[CanBtl.scala 214:72]
    node _sampledBit_T_6 = or(_sampledBit_T_3, _sampledBit_T_5) @[CanBtl.scala 214:59]
    skip
    node _GEN_14 = mux(io_tripleSampling, _sampledBit_T_6, io_rx) @[CanBtl.scala 213:31 214:20 216:20]
    node _GEN_15 = mux(_T_25, _GEN_14, sampledBit) @[CanBtl.scala 212:65 99:34]
    node _GEN_16 = or(_T_25, samplePoint) @[CanBtl.scala 212:65 218:19 95:35]
    node _GEN_17 = mux(_T_25, sampledBit, sampledBitQ) @[CanBtl.scala 212:65 219:19 101:35]
    node _GEN_18 = mux(_T_22, _GEN_15, sampledBit) @[CanBtl.scala 211:38 99:34]
    node _GEN_19 = and(_T_22, _GEN_16) @[CanBtl.scala 211:38 222:17]
    node _GEN_20 = mux(_T_22, _GEN_17, sampledBitQ) @[CanBtl.scala 101:35 211:38]
    node _GEN_21 = mux(io_goErrorFrame, sampledBit, _GEN_20) @[CanBtl.scala 207:25 208:17]
    node _GEN_22 = mux(io_goErrorFrame, UInt<1>("h0"), _GEN_19) @[CanBtl.scala 207:25 209:17]
    node _GEN_23 = mux(io_goErrorFrame, sampledBit, _GEN_18) @[CanBtl.scala 207:25 99:34]
    node _T_26 = not(io_nodeErrorPassive) @[CanBtl.scala 225:48]
    node _T_27 = and(io_goErrorFrame, _T_26) @[CanBtl.scala 225:46]
    node _T_28 = or(io_goOverloadFrame, _T_27) @[CanBtl.scala 225:27]
    node _T_29 = or(_T_28, io_goTx) @[CanBtl.scala 225:70]
    node _T_30 = or(_T_29, io_sendAck) @[CanBtl.scala 225:80]
    node _T_31 = and(io_goErrorFrame, io_nodeErrorPassive) @[CanBtl.scala 228:29]
    node _GEN_24 = mux(samplePoint, io_txNext, txNextSp) @[CanBtl.scala 231:26 232:14 128:31]
    node _GEN_25 = or(_T_31, _GEN_24) @[CanBtl.scala 228:52 229:14]
    node _GEN_26 = mux(_T_30, UInt<1>("h0"), _GEN_25) @[CanBtl.scala 225:94 226:14]
    node _GEN_27 = mux(goSeg2, UInt<1>("h0"), syncBlocked) @[CanBtl.scala 238:24 239:19 110:35]
    node _GEN_28 = or(resync, _GEN_27) @[CanBtl.scala 236:18 237:19]
    node _GEN_29 = mux(clockEnQ, _GEN_28, syncBlocked) @[CanBtl.scala 235:18 110:35]
    node _T_32 = and(io_hardSync, clockEnQ) @[CanBtl.scala 243:20]
    node _T_33 = and(io_transmitting, io_transmitter) @[CanBtl.scala 243:50]
    node _T_34 = or(_T_33, io_goTx) @[CanBtl.scala 243:67]
    node _T_35 = and(_T_34, io_txPoint) @[CanBtl.scala 243:78]
    node _T_36 = not(io_txNext) @[CanBtl.scala 243:94]
    node _T_37 = and(_T_35, _T_36) @[CanBtl.scala 243:91]
    node _T_38 = or(_T_32, _T_37) @[CanBtl.scala 243:31]
    skip
    node _T_40 = and(_io_hardSync_T, samplePoint) @[CanBtl.scala 246:53]
    node _T_41 = and(_T_40, io_sampledBit) @[CanBtl.scala 246:67]
    node _T_42 = or(io_goRxInter, _T_41) @[CanBtl.scala 246:26]
    node _GEN_30 = mux(_T_42, UInt<1>("h0"), hardSyncBlocked) @[CanBtl.scala 246:84 247:21 112:39]
    node _GEN_31 = or(_T_38, _GEN_30) @[CanBtl.scala 243:107 244:21]
    io_samplePoint <= samplePoint @[CanBtl.scala 96:18]
    io_sampledBit <= sampledBit @[CanBtl.scala 100:17]
    io_sampledBitQ <= sampledBitQ @[CanBtl.scala 102:18]
    io_txPoint <= txPoint @[CanBtl.scala 98:14]
    io_hardSync <= and(_io_hardSync_T_3, _io_hardSync_T_4) @[CanBtl.scala 146:70]
    samplePoint <= mux(reset, UInt<1>("h0"), _GEN_22) @[CanBtl.scala 95:{35,35}]
    txPoint <= mux(reset, UInt<1>("h0"), _txPoint_T_9) @[CanBtl.scala 153:11 97:{31,31}]
    sampledBit <= or(reset, _GEN_23) @[CanBtl.scala 99:{34,34}]
    sampledBitQ <= or(reset, _GEN_21) @[CanBtl.scala 101:{35,35}]
    clockCount <= mux(reset, UInt<7>("h0"), _GEN_0) @[CanBtl.scala 104:{34,34}]
    clockEn <= mux(reset, UInt<1>("h0"), _T_5) @[CanBtl.scala 106:{31,31}]
    clockEnQ <= mux(reset, UInt<1>("h0"), clockEn) @[CanBtl.scala 108:{32,32,32}]
    syncBlocked <= or(reset, _GEN_29) @[CanBtl.scala 110:{35,35}]
    hardSyncBlocked <= mux(reset, UInt<1>("h0"), _GEN_31) @[CanBtl.scala 112:{39,39}]
    node _GEN_38 = mux(reset, UInt<6>("h0"), _GEN_11) @[CanBtl.scala 114:{37,37}]
    quantaCounter <= bits(_GEN_38, 4, 0) @[CanBtl.scala 114:{37,37}]
    node _GEN_39 = mux(reset, UInt<6>("h0"), _GEN_13) @[CanBtl.scala 116:{29,29}]
    delay <= bits(_GEN_39, 3, 0) @[CanBtl.scala 116:{29,29}]
    sync <= mux(reset, UInt<1>("h0"), _GEN_4) @[CanBtl.scala 118:{28,28}]
    seg1 <= or(reset, _GEN_7) @[CanBtl.scala 120:{28,28}]
    seg2 <= mux(reset, UInt<1>("h0"), _GEN_9) @[CanBtl.scala 122:{28,28}]
    resyncLatched <= mux(reset, UInt<1>("h0"), _GEN_3) @[CanBtl.scala 124:{37,37}]
    sample <= mux(reset, UInt<2>("h3"), _GEN_5) @[CanBtl.scala 126:{30,30}]
    txNextSp <= mux(reset, UInt<1>("h0"), _GEN_26) @[CanBtl.scala 128:{31,31}]

  module CanCrc :
    input clock : Clock
    input reset : UInt<1>
    input io_data : UInt<1>
    input io_enable : UInt<1>
    output io_crc : UInt<15>

    reg crc : UInt<15>, clock with :
      reset => (UInt<1>("h0"), crc) @[CanCrc.scala 13:27]
    node _crcNext_T = bits(io_crc, 14, 14) @[CanCrc.scala 14:40]
    node crcNext = xor(io_data, _crcNext_T) @[CanCrc.scala 14:32]
    node crcTmp_hi = bits(io_crc, 13, 0) @[CanCrc.scala 15:33]
    node crcTmp = cat(crcTmp_hi, UInt<1>("h0")) @[Cat.scala 30:58]
    node _crc_T = xor(crcTmp, UInt<15>("h4599")) @[CanCrc.scala 20:21]
    node _GEN_0 = mux(crcNext, _crc_T, crcTmp) @[CanCrc.scala 19:19 20:11 22:11]
    node _GEN_1 = mux(io_enable, _GEN_0, crc) @[CanCrc.scala 18:19 13:27]
    io_crc <= crc @[CanCrc.scala 16:10]
    crc <= mux(reset, UInt<15>("h0"), _GEN_1) @[CanCrc.scala 13:{27,27}]

  module CanAcf :
    input clock : Clock
    input reset : UInt<1>
    input io_id : UInt<29>
    input io_resetMode : UInt<1>
    input io_acceptanceFilterMode : UInt<1>
    input io_extendedMode : UInt<1>
    input io_acceptanceCode_0 : UInt<8>
    input io_acceptanceCode_1 : UInt<8>
    input io_acceptanceCode_2 : UInt<8>
    input io_acceptanceCode_3 : UInt<8>
    input io_acceptanceMask_0 : UInt<8>
    input io_acceptanceMask_1 : UInt<8>
    input io_acceptanceMask_2 : UInt<8>
    input io_acceptanceMask_3 : UInt<8>
    input io_goRxCrcLim : UInt<1>
    input io_goRxInter : UInt<1>
    input io_goErrorFrame : UInt<1>
    input io_data0 : UInt<8>
    input io_data1 : UInt<8>
    input io_rtr1 : UInt<1>
    input io_rtr2 : UInt<1>
    input io_ide : UInt<1>
    input io_noByte0 : UInt<1>
    input io_noByte1 : UInt<1>
    output io_idOk : UInt<1>

    reg idOk : UInt<1>, clock with :
      reset => (UInt<1>("h0"), idOk) @[CanAcf.scala 27:28]
    node _idMatch_T = bits(io_id, 10, 3) @[CanAcf.scala 33:43]
    node _idMatch_T_1 = xor(_idMatch_T, io_acceptanceCode_0) @[CanAcf.scala 31:11]
    node _idMatch_T_2 = not(_idMatch_T_1) @[CanAcf.scala 31:7]
    node _idMatch_T_3 = or(_idMatch_T_2, io_acceptanceMask_0) @[CanAcf.scala 31:26]
    node idMatch = andr(_idMatch_T_3) @[CanAcf.scala 31:35]
    node matchSfStd_hi = bits(io_id, 2, 0) @[CanAcf.scala 36:26]
    node _matchSfStd_T = cat(matchSfStd_hi, io_rtr1) @[Cat.scala 30:58]
    node _matchSfStd_T_1 = bits(io_acceptanceCode_1, 7, 4) @[CanAcf.scala 36:61]
    node _matchSfStd_T_2 = bits(io_acceptanceMask_1, 7, 4) @[CanAcf.scala 36:88]
    node _matchSfStd_T_3 = xor(_matchSfStd_T, _matchSfStd_T_1) @[CanAcf.scala 31:11]
    node _matchSfStd_T_4 = not(_matchSfStd_T_3) @[CanAcf.scala 31:7]
    node _matchSfStd_T_5 = or(_matchSfStd_T_4, _matchSfStd_T_2) @[CanAcf.scala 31:26]
    node _matchSfStd_T_6 = andr(_matchSfStd_T_5) @[CanAcf.scala 31:35]
    node _matchSfStd_T_7 = and(idMatch, _matchSfStd_T_6) @[CanAcf.scala 35:35]
    node _matchSfStd_T_8 = xor(io_data0, io_acceptanceCode_2) @[CanAcf.scala 31:11]
    node _matchSfStd_T_9 = not(_matchSfStd_T_8) @[CanAcf.scala 31:7]
    node _matchSfStd_T_10 = or(_matchSfStd_T_9, io_acceptanceMask_2) @[CanAcf.scala 31:26]
    node _matchSfStd_T_11 = andr(_matchSfStd_T_10) @[CanAcf.scala 31:35]
    node _matchSfStd_T_12 = or(_matchSfStd_T_11, io_noByte0) @[CanAcf.scala 37:72]
    node _matchSfStd_T_13 = and(_matchSfStd_T_7, _matchSfStd_T_12) @[CanAcf.scala 36:95]
    node _matchSfStd_T_14 = xor(io_data1, io_acceptanceCode_3) @[CanAcf.scala 31:11]
    node _matchSfStd_T_15 = not(_matchSfStd_T_14) @[CanAcf.scala 31:7]
    node _matchSfStd_T_16 = or(_matchSfStd_T_15, io_acceptanceMask_3) @[CanAcf.scala 31:26]
    node _matchSfStd_T_17 = andr(_matchSfStd_T_16) @[CanAcf.scala 31:35]
    node _matchSfStd_T_18 = or(_matchSfStd_T_17, io_noByte1) @[CanAcf.scala 38:72]
    node matchSfStd = and(_matchSfStd_T_13, _matchSfStd_T_18) @[CanAcf.scala 37:86]
    node _matchSfExt_T = bits(io_id, 28, 21) @[CanAcf.scala 40:44]
    node _matchSfExt_T_1 = xor(_matchSfExt_T, io_acceptanceCode_0) @[CanAcf.scala 31:11]
    node _matchSfExt_T_2 = not(_matchSfExt_T_1) @[CanAcf.scala 31:7]
    node _matchSfExt_T_3 = or(_matchSfExt_T_2, io_acceptanceMask_0) @[CanAcf.scala 31:26]
    node _matchSfExt_T_4 = andr(_matchSfExt_T_3) @[CanAcf.scala 31:35]
    node _matchSfExt_T_5 = bits(io_id, 20, 13) @[CanAcf.scala 41:22]
    node _matchSfExt_T_6 = xor(_matchSfExt_T_5, io_acceptanceCode_1) @[CanAcf.scala 31:11]
    node _matchSfExt_T_7 = not(_matchSfExt_T_6) @[CanAcf.scala 31:7]
    node _matchSfExt_T_8 = or(_matchSfExt_T_7, io_acceptanceMask_1) @[CanAcf.scala 31:26]
    node _matchSfExt_T_9 = andr(_matchSfExt_T_8) @[CanAcf.scala 31:35]
    node _matchSfExt_T_10 = and(_matchSfExt_T_4, _matchSfExt_T_9) @[CanAcf.scala 40:97]
    node _matchSfExt_T_11 = bits(io_id, 12, 5) @[CanAcf.scala 42:22]
    node _matchSfExt_T_12 = xor(_matchSfExt_T_11, io_acceptanceCode_2) @[CanAcf.scala 31:11]
    node _matchSfExt_T_13 = not(_matchSfExt_T_12) @[CanAcf.scala 31:7]
    node _matchSfExt_T_14 = or(_matchSfExt_T_13, io_acceptanceMask_2) @[CanAcf.scala 31:26]
    node _matchSfExt_T_15 = andr(_matchSfExt_T_14) @[CanAcf.scala 31:35]
    node _matchSfExt_T_16 = and(_matchSfExt_T_10, _matchSfExt_T_15) @[CanAcf.scala 41:75]
    node matchSfExt_hi = bits(io_id, 4, 0) @[CanAcf.scala 43:26]
    node _matchSfExt_T_17 = cat(matchSfExt_hi, io_rtr2) @[Cat.scala 30:58]
    node _matchSfExt_T_18 = bits(io_acceptanceCode_3, 7, 2) @[CanAcf.scala 43:61]
    node _matchSfExt_T_19 = bits(io_acceptanceMask_3, 7, 2) @[CanAcf.scala 43:87]
    node _matchSfExt_T_20 = xor(_matchSfExt_T_17, _matchSfExt_T_18) @[CanAcf.scala 31:11]
    node _matchSfExt_T_21 = not(_matchSfExt_T_20) @[CanAcf.scala 31:7]
    node _matchSfExt_T_22 = or(_matchSfExt_T_21, _matchSfExt_T_19) @[CanAcf.scala 31:26]
    node _matchSfExt_T_23 = andr(_matchSfExt_T_22) @[CanAcf.scala 31:35]
    node matchSfExt = and(_matchSfExt_T_16, _matchSfExt_T_23) @[CanAcf.scala 42:74]
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    node _matchDfStd_T_8 = bits(io_data0, 7, 4) @[CanAcf.scala 48:26]
    node _matchDfStd_T_9 = bits(io_acceptanceCode_1, 3, 0) @[CanAcf.scala 48:53]
    node _matchDfStd_T_10 = bits(io_acceptanceMask_1, 3, 0) @[CanAcf.scala 48:80]
    node _matchDfStd_T_11 = xor(_matchDfStd_T_8, _matchDfStd_T_9) @[CanAcf.scala 31:11]
    node _matchDfStd_T_12 = not(_matchDfStd_T_11) @[CanAcf.scala 31:7]
    node _matchDfStd_T_13 = or(_matchDfStd_T_12, _matchDfStd_T_10) @[CanAcf.scala 31:26]
    node _matchDfStd_T_14 = andr(_matchDfStd_T_13) @[CanAcf.scala 31:35]
    node _matchDfStd_T_15 = or(_matchDfStd_T_14, io_noByte0) @[CanAcf.scala 48:87]
    node _matchDfStd_T_16 = and(_matchSfStd_T_7, _matchDfStd_T_15) @[CanAcf.scala 47:94]
    node _matchDfStd_T_17 = bits(io_data0, 3, 0) @[CanAcf.scala 49:26]
    node _matchDfStd_T_18 = bits(io_acceptanceCode_3, 3, 0) @[CanAcf.scala 49:53]
    node _matchDfStd_T_19 = bits(io_acceptanceMask_3, 3, 0) @[CanAcf.scala 49:80]
    node _matchDfStd_T_20 = xor(_matchDfStd_T_17, _matchDfStd_T_18) @[CanAcf.scala 31:11]
    node _matchDfStd_T_21 = not(_matchDfStd_T_20) @[CanAcf.scala 31:7]
    node _matchDfStd_T_22 = or(_matchDfStd_T_21, _matchDfStd_T_19) @[CanAcf.scala 31:26]
    node _matchDfStd_T_23 = andr(_matchDfStd_T_22) @[CanAcf.scala 31:35]
    node _matchDfStd_T_24 = or(_matchDfStd_T_23, io_noByte0) @[CanAcf.scala 49:87]
    node _matchDfStd_T_25 = and(_matchDfStd_T_16, _matchDfStd_T_24) @[CanAcf.scala 48:101]
    skip
    node _matchDfStd_T_27 = xor(_idMatch_T, io_acceptanceCode_2) @[CanAcf.scala 31:11]
    node _matchDfStd_T_28 = not(_matchDfStd_T_27) @[CanAcf.scala 31:7]
    node _matchDfStd_T_29 = or(_matchDfStd_T_28, io_acceptanceMask_2) @[CanAcf.scala 31:26]
    node _matchDfStd_T_30 = andr(_matchDfStd_T_29) @[CanAcf.scala 31:35]
    skip
    skip
    node _matchDfStd_T_32 = bits(io_acceptanceCode_3, 7, 4) @[CanAcf.scala 51:61]
    node _matchDfStd_T_33 = bits(io_acceptanceMask_3, 7, 4) @[CanAcf.scala 51:87]
    node _matchDfStd_T_34 = xor(_matchSfStd_T, _matchDfStd_T_32) @[CanAcf.scala 31:11]
    node _matchDfStd_T_35 = not(_matchDfStd_T_34) @[CanAcf.scala 31:7]
    node _matchDfStd_T_36 = or(_matchDfStd_T_35, _matchDfStd_T_33) @[CanAcf.scala 31:26]
    node _matchDfStd_T_37 = andr(_matchDfStd_T_36) @[CanAcf.scala 31:35]
    node _matchDfStd_T_38 = and(_matchDfStd_T_30, _matchDfStd_T_37) @[CanAcf.scala 50:75]
    node matchDfStd = or(_matchDfStd_T_25, _matchDfStd_T_38) @[CanAcf.scala 49:102]
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    node _matchDfExt_T_12 = xor(_matchSfExt_T, io_acceptanceCode_2) @[CanAcf.scala 31:11]
    node _matchDfExt_T_13 = not(_matchDfExt_T_12) @[CanAcf.scala 31:7]
    node _matchDfExt_T_14 = or(_matchDfExt_T_13, io_acceptanceMask_2) @[CanAcf.scala 31:26]
    node _matchDfExt_T_15 = andr(_matchDfExt_T_14) @[CanAcf.scala 31:35]
    skip
    node _matchDfExt_T_17 = xor(_matchSfExt_T_5, io_acceptanceCode_3) @[CanAcf.scala 31:11]
    node _matchDfExt_T_18 = not(_matchDfExt_T_17) @[CanAcf.scala 31:7]
    node _matchDfExt_T_19 = or(_matchDfExt_T_18, io_acceptanceMask_3) @[CanAcf.scala 31:26]
    node _matchDfExt_T_20 = andr(_matchDfExt_T_19) @[CanAcf.scala 31:35]
    node _matchDfExt_T_21 = and(_matchDfExt_T_15, _matchDfExt_T_20) @[CanAcf.scala 55:92]
    node matchDfExt = or(_matchSfExt_T_10, _matchDfExt_T_21) @[CanAcf.scala 54:93]
    node _idOk_T = mux(io_ide, matchSfExt, matchSfStd) @[CanAcf.scala 59:64]
    node _idOk_T_1 = mux(io_ide, matchDfExt, matchDfStd) @[CanAcf.scala 59:98]
    node _idOk_T_2 = mux(io_acceptanceFilterMode, _idOk_T, _idOk_T_1) @[CanAcf.scala 59:36]
    node _idOk_T_3 = mux(io_extendedMode, _idOk_T_2, idMatch) @[CanAcf.scala 59:16]
    node _T = or(io_resetMode, io_goRxInter) @[CanAcf.scala 60:27]
    node _T_1 = or(_T, io_goErrorFrame) @[CanAcf.scala 60:42]
    node _GEN_0 = mux(_T_1, UInt<1>("h0"), idOk) @[CanAcf.scala 60:61 61:10 27:28]
    node _GEN_1 = mux(io_goRxCrcLim, _idOk_T_3, _GEN_0) @[CanAcf.scala 58:23 59:10]
    io_idOk <= idOk @[CanAcf.scala 28:11]
    idOk <= mux(reset, UInt<1>("h0"), _GEN_1) @[CanAcf.scala 27:{28,28}]

  module CanFifo :
    input clock : Clock
    input reset : UInt<1>
    input io_wr : UInt<1>
    input io_dataIn : UInt<8>
    input io_addr : UInt<6>
    output io_dataOut : UInt<8>
    input io_resetMode : UInt<1>
    input io_releaseBuffer : UInt<1>
    input io_extendedMode : UInt<1>
    output io_overrun : UInt<1>
    output io_infoEmpty : UInt<1>
    output io_infoCnt : UInt<7>

    mem fifo : @[CanFifo.scala 37:45]
      data-type => UInt<8>
      depth => 64
      read-latency => 1
      write-latency => 1
      reader => io_dataOut_MPORT
      writer => MPORT
      read-under-write => undefined
    mem lengthFifo : @[CanFifo.scala 38:51]
      data-type => UInt<4>
      depth => 64
      read-latency => 1
      write-latency => 1
      reader => lengthInfo_MPORT
      writer => MPORT_1
      read-under-write => undefined
    mem overrunInfo : @[CanFifo.scala 39:52]
      data-type => UInt<1>
      depth => 64
      read-latency => 1
      write-latency => 1
      reader => io_overrun_MPORT
      writer => MPORT_2
      read-under-write => undefined
    reg infoCnt : UInt<7>, clock with :
      reset => (UInt<1>("h0"), infoCnt) @[CanFifo.scala 23:31]
    reg rdPointer : UInt<6>, clock with :
      reset => (UInt<1>("h0"), rdPointer) @[CanFifo.scala 41:33]
    reg wrPointer : UInt<6>, clock with :
      reset => (UInt<1>("h0"), wrPointer) @[CanFifo.scala 42:33]
    reg wrInfoPointer : UInt<6>, clock with :
      reset => (UInt<1>("h0"), wrInfoPointer) @[CanFifo.scala 44:37]
    reg rdInfoPointer : UInt<6>, clock with :
      reset => (UInt<1>("h0"), rdInfoPointer) @[CanFifo.scala 45:37]
    node _wrQ_T = mux(io_resetMode, UInt<1>("h0"), io_wr) @[CanFifo.scala 46:31]
    reg wrQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), wrQ) @[CanFifo.scala 46:27]
    reg lenCnt : UInt<4>, clock with :
      reset => (UInt<1>("h0"), lenCnt) @[CanFifo.scala 47:30]
    reg fifoCnt : UInt<7>, clock with :
      reset => (UInt<1>("h0"), fifoCnt) @[CanFifo.scala 48:31]
    reg latchOverrun : UInt<1>, clock with :
      reset => (UInt<1>("h0"), latchOverrun) @[CanFifo.scala 50:36]
    reg initializeMemories : UInt<1>, clock with :
      reset => (UInt<1>("h0"), initializeMemories) @[CanFifo.scala 51:42]
    node _writeLengthInfo_T = not(io_wr) @[CanFifo.scala 53:32]
    node writeLengthInfo = and(_writeLengthInfo_T, wrQ) @[CanFifo.scala 53:39]
    node fifoEmpty = eq(fifoCnt, UInt<7>("h0")) @[CanFifo.scala 54:34]
    node fifoFull = eq(fifoCnt, UInt<7>("h40")) @[CanFifo.scala 55:33]
    node infoFull = eq(infoCnt, UInt<7>("h40")) @[CanFifo.scala 56:33]
    skip
    node _readAddress_T = mux(io_extendedMode, UInt<6>("h10"), UInt<6>("h14")) @[CanFifo.scala 59:45]
    node _readAddress_T_1 = sub(io_addr, _readAddress_T) @[CanFifo.scala 59:40]
    node _readAddress_T_2 = tail(_readAddress_T_1, 1) @[CanFifo.scala 59:40]
    node _readAddress_T_3 = add(rdPointer, _readAddress_T_2) @[CanFifo.scala 59:29]
    skip
    node _T = or(io_resetMode, writeLengthInfo) @[CanFifo.scala 61:21]
    node _T_1 = not(fifoFull) @[CanFifo.scala 63:23]
    node _T_2 = and(io_wr, _T_1) @[CanFifo.scala 63:20]
    node _lenCnt_T = add(lenCnt, UInt<4>("h1")) @[CanFifo.scala 64:22]
    node _lenCnt_T_1 = tail(_lenCnt_T, 1) @[CanFifo.scala 64:22]
    node _GEN_0 = mux(_T_2, _lenCnt_T_1, lenCnt) @[CanFifo.scala 63:35 64:12 47:30]
    node _GEN_1 = mux(_T, UInt<4>("h0"), _GEN_0) @[CanFifo.scala 61:40 62:12]
    node _T_3 = not(infoFull) @[CanFifo.scala 67:27]
    node _T_4 = and(writeLengthInfo, _T_3) @[CanFifo.scala 67:24]
    node _T_5 = or(_T_4, initializeMemories) @[CanFifo.scala 67:38]
    node _wrInfoPointer_T = add(wrInfoPointer, UInt<6>("h1")) @[CanFifo.scala 68:36]
    node _wrInfoPointer_T_1 = tail(_wrInfoPointer_T, 1) @[CanFifo.scala 68:36]
    node _GEN_2 = mux(io_resetMode, rdInfoPointer, wrInfoPointer) @[CanFifo.scala 69:28 70:19 44:37]
    node _GEN_3 = mux(_T_5, _wrInfoPointer_T_1, _GEN_2) @[CanFifo.scala 67:60 68:19]
    skip
    node _T_7 = and(io_releaseBuffer, _T_3) @[CanFifo.scala 73:25]
    node _rdInfoPointer_T = add(rdInfoPointer, UInt<6>("h1")) @[CanFifo.scala 74:36]
    node _GEN_4 = mux(_T_7, _rdInfoPointer_T, pad(rdInfoPointer, 7)) @[CanFifo.scala 73:38 74:19 45:37]
    skip
    node _T_8 = not(fifoEmpty) @[CanFifo.scala 77:27]
    node _T_9 = and(io_releaseBuffer, _T_8) @[CanFifo.scala 77:25]
    node lengthInfo = lengthFifo.lengthInfo_MPORT.data @[CanFifo.scala 52:31 58:14]
    node _GEN_35 = pad(lengthInfo, 6) @[CanFifo.scala 78:28]
    node _rdPointer_T = add(rdPointer, _GEN_35) @[CanFifo.scala 78:28]
    node _rdPointer_T_1 = tail(_rdPointer_T, 1) @[CanFifo.scala 78:28]
    node _GEN_6 = mux(_T_9, _rdPointer_T_1, rdPointer) @[CanFifo.scala 77:39 78:15 41:33]
    skip
    skip
    node _wrPointer_T = add(wrPointer, UInt<6>("h1")) @[CanFifo.scala 84:28]
    node _wrPointer_T_1 = tail(_wrPointer_T, 1) @[CanFifo.scala 84:28]
    node _GEN_7 = mux(_T_2, _wrPointer_T_1, wrPointer) @[CanFifo.scala 83:33 84:15 42:33]
    node _GEN_8 = mux(io_resetMode, rdPointer, _GEN_7) @[CanFifo.scala 81:22 82:15]
    skip
    node _T_13 = and(io_wr, fifoFull) @[CanFifo.scala 89:20]
    node _GEN_9 = or(_T_13, latchOverrun) @[CanFifo.scala 89:32 90:18 50:36]
    node _GEN_10 = mux(_T, UInt<1>("h0"), _GEN_9) @[CanFifo.scala 87:40 88:18]
    node _T_14 = not(io_releaseBuffer) @[CanFifo.scala 95:22]
    node _T_15 = and(io_wr, _T_14) @[CanFifo.scala 95:20]
    skip
    node _T_17 = and(_T_15, _T_1) @[CanFifo.scala 95:40]
    node _fifoCnt_T = add(fifoCnt, UInt<7>("h1")) @[CanFifo.scala 96:24]
    skip
    node _T_19 = and(_writeLengthInfo_T, io_releaseBuffer) @[CanFifo.scala 97:21]
    skip
    node _T_21 = and(_T_19, _T_8) @[CanFifo.scala 97:40]
    node _fifoCnt_T_1 = cat(UInt<3>("h0"), lengthInfo) @[Cat.scala 30:58]
    node _fifoCnt_T_2 = sub(fifoCnt, _fifoCnt_T_1) @[CanFifo.scala 98:24]
    node _fifoCnt_T_3 = tail(_fifoCnt_T_2, 1) @[CanFifo.scala 98:24]
    node _T_22 = and(io_wr, io_releaseBuffer) @[CanFifo.scala 99:20]
    skip
    node _T_24 = and(_T_22, _T_1) @[CanFifo.scala 99:39]
    skip
    node _T_26 = and(_T_24, _T_8) @[CanFifo.scala 99:51]
    skip
    skip
    skip
    node _fifoCnt_T_7 = add(_fifoCnt_T_3, UInt<7>("h1")) @[CanFifo.scala 100:53]
    node _GEN_11 = mux(_T_26, _fifoCnt_T_7, pad(fifoCnt, 8)) @[CanFifo.scala 100:15 48:31 99:65]
    node _GEN_12 = mux(_T_21, pad(_fifoCnt_T_3, 8), _GEN_11) @[CanFifo.scala 97:54 98:13]
    node _GEN_13 = mux(_T_17, _fifoCnt_T, _GEN_12) @[CanFifo.scala 95:53 96:13]
    node _GEN_14 = mux(io_resetMode, UInt<8>("h0"), _GEN_13) @[CanFifo.scala 93:22 94:13]
    node _T_27 = xor(writeLengthInfo, io_releaseBuffer) @[CanFifo.scala 105:30]
    node _T_28 = not(io_infoEmpty) @[CanFifo.scala 106:29]
    node _T_29 = and(io_releaseBuffer, _T_28) @[CanFifo.scala 106:27]
    node _infoCnt_T = sub(infoCnt, UInt<7>("h1")) @[CanFifo.scala 107:27]
    skip
    skip
    skip
    node _infoCnt_T_2 = add(infoCnt, UInt<7>("h1")) @[CanFifo.scala 109:27]
    node _GEN_15 = mux(_T_4, _infoCnt_T_2, pad(infoCnt, 8)) @[CanFifo.scala 108:45 109:15 23:31]
    node _GEN_16 = mux(_T_29, _infoCnt_T, _GEN_15) @[CanFifo.scala 106:44 107:15]
    node _GEN_17 = mux(_T_27, _GEN_16, pad(infoCnt, 8)) @[CanFifo.scala 105:50 23:31]
    node _GEN_18 = mux(io_resetMode, UInt<8>("h0"), _GEN_17) @[CanFifo.scala 103:22 104:13]
    node _T_32 = andr(wrInfoPointer) @[CanFifo.scala 113:26]
    node _GEN_19 = mux(_T_32, UInt<1>("h0"), initializeMemories) @[CanFifo.scala 113:30 114:24 51:42]
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    node _T_43 = or(latchOverrun, _T_13) @[CanFifo.scala 128:51]
    node _T_44 = not(initializeMemories) @[CanFifo.scala 128:76]
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    io_dataOut <= fifo.io_dataOut_MPORT.data @[CanFifo.scala 121:14]
    io_overrun <= overrunInfo.io_overrun_MPORT.data @[CanFifo.scala 130:14]
    io_infoEmpty <= eq(infoCnt, UInt<7>("h0")) @[CanFifo.scala 57:27]
    io_infoCnt <= infoCnt @[CanFifo.scala 24:14]
    node _GEN_36 = mux(reset, UInt<8>("h0"), _GEN_18) @[CanFifo.scala 23:{31,31}]
    infoCnt <= bits(_GEN_36, 6, 0) @[CanFifo.scala 23:{31,31}]
    fifo.io_dataOut_MPORT.addr <= tail(_readAddress_T_3, 1) @[CanFifo.scala 59:29]
    fifo.io_dataOut_MPORT.en <= UInt<1>("h1") @[CanFifo.scala 59:15]
    fifo.io_dataOut_MPORT.clk <= clock @[CanFifo.scala 121:21]
    fifo.MPORT.addr <= wrPointer @[CanFifo.scala 117:27 118:9]
    fifo.MPORT.en <= and(io_wr, _T_1) @[CanFifo.scala 117:14]
    fifo.MPORT.clk <= clock @[CanFifo.scala 117:27 118:9]
    fifo.MPORT.data <= io_dataIn @[CanFifo.scala 117:27 118:21]
    fifo.MPORT.mask <= UInt<1>("h1") @[CanFifo.scala 117:27 118:21]
    lengthFifo.lengthInfo_MPORT.addr <= rdInfoPointer @[CanFifo.scala 58:32]
    lengthFifo.lengthInfo_MPORT.en <= UInt<1>("h0") @[CanFifo.scala 38:51]
    lengthFifo.lengthInfo_MPORT.clk <= clock @[CanFifo.scala 58:32]
    lengthFifo.MPORT_1.addr <= wrInfoPointer @[CanFifo.scala 123:60 124:15]
    lengthFifo.MPORT_1.en <= or(_T_4, initializeMemories) @[CanFifo.scala 123:38]
    lengthFifo.MPORT_1.clk <= clock @[CanFifo.scala 123:60 124:15]
    lengthFifo.MPORT_1.data <= mux(initializeMemories, UInt<4>("h0"), lenCnt) @[CanFifo.scala 124:37]
    lengthFifo.MPORT_1.mask <= UInt<1>("h1") @[CanFifo.scala 123:60 124:31]
    overrunInfo.io_overrun_MPORT.addr <= rdInfoPointer @[CanFifo.scala 130:28]
    overrunInfo.io_overrun_MPORT.en <= and(io_releaseBuffer, _T_3) @[CanFifo.scala 73:25]
    overrunInfo.io_overrun_MPORT.clk <= clock @[CanFifo.scala 130:28]
    overrunInfo.MPORT_2.addr <= wrInfoPointer @[CanFifo.scala 127:60]
    overrunInfo.MPORT_2.en <= or(_T_4, initializeMemories) @[CanFifo.scala 127:38]
    overrunInfo.MPORT_2.clk <= clock @[CanFifo.scala 127:60]
    overrunInfo.MPORT_2.data <= and(_T_43, _T_44) @[CanFifo.scala 128:73]
    overrunInfo.MPORT_2.mask <= UInt<1>("h1") @[CanFifo.scala 127:60]
    rdPointer <= mux(reset, UInt<6>("h0"), _GEN_6) @[CanFifo.scala 41:{33,33}]
    wrPointer <= mux(reset, UInt<6>("h0"), _GEN_8) @[CanFifo.scala 42:{33,33}]
    wrInfoPointer <= mux(reset, UInt<6>("h0"), _GEN_3) @[CanFifo.scala 44:{37,37}]
    node _GEN_37 = mux(reset, UInt<7>("h0"), _GEN_4) @[CanFifo.scala 45:{37,37}]
    rdInfoPointer <= bits(_GEN_37, 5, 0) @[CanFifo.scala 45:{37,37}]
    wrQ <= mux(reset, UInt<1>("h0"), _wrQ_T) @[CanFifo.scala 46:{27,27,27}]
    lenCnt <= mux(reset, UInt<4>("h0"), _GEN_1) @[CanFifo.scala 47:{30,30}]
    node _GEN_38 = mux(reset, UInt<8>("h0"), _GEN_14) @[CanFifo.scala 48:{31,31}]
    fifoCnt <= bits(_GEN_38, 6, 0) @[CanFifo.scala 48:{31,31}]
    latchOverrun <= mux(reset, UInt<1>("h0"), _GEN_10) @[CanFifo.scala 50:{36,36}]
    initializeMemories <= or(reset, _GEN_19) @[CanFifo.scala 51:{42,42}]

  module CanBsp :
    input clock : Clock
    input reset : UInt<1>
    input io_samplePoint : UInt<1>
    input io_sampledBit : UInt<1>
    input io_sampledBitQ : UInt<1>
    input io_txPoint : UInt<1>
    input io_hardSync : UInt<1>
    input io_addr : UInt<8>
    input io_dataIn : UInt<8>
    output io_dataOut : UInt<8>
    input io_resetMode : UInt<1>
    input io_listenOnlyMode : UInt<1>
    input io_acceptanceFilterMode : UInt<1>
    input io_extendedMode : UInt<1>
    input io_selfTestMode : UInt<1>
    input io_releaseBuffer : UInt<1>
    input io_txRequest : UInt<1>
    input io_abortTx : UInt<1>
    input io_selfRxRequest : UInt<1>
    input io_singleShotTransmission : UInt<1>
    output io_txState : UInt<1>
    output io_txStateQ : UInt<1>
    output io_overloadFrame : UInt<1>
    input io_readArbitrationLostCaptureReg : UInt<1>
    input io_readErrorCodeCaptureReg : UInt<1>
    output io_errorCaptureCode : UInt<8>
    input io_errorWarningLimit : UInt<8>
    input io_writeEnReceiveErrorCounter : UInt<1>
    input io_writeEnTransmitErrorCounter : UInt<1>
    output io_rxIdle : UInt<1>
    output io_transmitting : UInt<1>
    output io_transmitter : UInt<1>
    output io_goRxInter : UInt<1>
    output io_notFirstBitOfInter : UInt<1>
    output io_rxInter : UInt<1>
    output io_setResetMode : UInt<1>
    output io_nodeBusOff : UInt<1>
    output io_errorStatus : UInt<1>
    output io_rxErrorCount : UInt<9>
    output io_txErrorCount : UInt<9>
    output io_transmitStatus : UInt<1>
    output io_receiveStatus : UInt<1>
    output io_txSuccessful : UInt<1>
    output io_needToTx : UInt<1>
    output io_overrun : UInt<1>
    output io_infoEmpty : UInt<1>
    output io_setBusErrorIrq : UInt<1>
    output io_setArbitrationLostIrq : UInt<1>
    output io_arbitrationLostCapture : UInt<5>
    output io_nodeErrorPassive : UInt<1>
    output io_nodeErrorActive : UInt<1>
    output io_rxMessageCounter : UInt<7>
    input io_acceptanceCode_0 : UInt<8>
    input io_acceptanceCode_1 : UInt<8>
    input io_acceptanceCode_2 : UInt<8>
    input io_acceptanceCode_3 : UInt<8>
    input io_acceptanceMask_0 : UInt<8>
    input io_acceptanceMask_1 : UInt<8>
    input io_acceptanceMask_2 : UInt<8>
    input io_acceptanceMask_3 : UInt<8>
    input io_txData_0 : UInt<8>
    input io_txData_1 : UInt<8>
    input io_txData_2 : UInt<8>
    input io_txData_3 : UInt<8>
    input io_txData_4 : UInt<8>
    input io_txData_5 : UInt<8>
    input io_txData_6 : UInt<8>
    input io_txData_7 : UInt<8>
    input io_txData_8 : UInt<8>
    input io_txData_9 : UInt<8>
    input io_txData_10 : UInt<8>
    input io_txData_11 : UInt<8>
    input io_txData_12 : UInt<8>
    output io_tx : UInt<1>
    output io_txNext : UInt<1>
    output io_busOffOn : UInt<1>
    output io_goOverloadFrame : UInt<1>
    output io_goErrorFrame : UInt<1>
    output io_goTx : UInt<1>
    output io_sendAck : UInt<1>

    mem tmpFifo : @[CanBsp.scala 136:32]
      data-type => UInt<8>
      depth => 8
      read-latency => 0
      write-latency => 1
      reader => canAcf_io_data0_MPORT
      reader => canAcf_io_data1_MPORT
      reader => dataForFifo_MPORT
      writer => MPORT
      read-under-write => undefined
    inst canCrcRx of CanCrc @[CanBsp.scala 522:72]
    inst canAcf of CanAcf @[CanBsp.scala 527:31]
    inst canFifo of CanFifo @[CanBsp.scala 578:33]
    reg txState : UInt<1>, clock with :
      reset => (UInt<1>("h0"), txState) @[CanBsp.scala 75:31]
    node _txStateQ_T = mux(io_resetMode, UInt<1>("h0"), txState) @[CanBsp.scala 77:36]
    reg txStateQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), txStateQ) @[CanBsp.scala 77:32]
    reg overloadFrame : UInt<1>, clock with :
      reset => (UInt<1>("h0"), overloadFrame) @[CanBsp.scala 79:37]
    reg errorCaptureCode : UInt<8>, clock with :
      reset => (UInt<1>("h0"), errorCaptureCode) @[CanBsp.scala 81:40]
    reg rxIdle : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxIdle) @[CanBsp.scala 83:30]
    reg transmitting : UInt<1>, clock with :
      reset => (UInt<1>("h0"), transmitting) @[CanBsp.scala 85:36]
    reg transmitter : UInt<1>, clock with :
      reset => (UInt<1>("h0"), transmitter) @[CanBsp.scala 87:35]
    reg rxInter : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxInter) @[CanBsp.scala 89:31]
    reg nodeBusOff : UInt<1>, clock with :
      reset => (UInt<1>("h0"), nodeBusOff) @[CanBsp.scala 91:34]
    reg needToTx : UInt<1>, clock with :
      reset => (UInt<1>("h0"), needToTx) @[CanBsp.scala 93:32]
    reg rxErrorCount : UInt<9>, clock with :
      reset => (UInt<1>("h0"), rxErrorCount) @[CanBsp.scala 95:36]
    reg txErrorCount : UInt<9>, clock with :
      reset => (UInt<1>("h0"), txErrorCount) @[CanBsp.scala 97:36]
    reg arbitrationLostCapture : UInt<5>, clock with :
      reset => (UInt<1>("h0"), arbitrationLostCapture) @[CanBsp.scala 99:46]
    reg nodeErrorPassive : UInt<1>, clock with :
      reset => (UInt<1>("h0"), nodeErrorPassive) @[CanBsp.scala 101:40]
    reg tx : UInt<1>, clock with :
      reset => (UInt<1>("h0"), tx) @[CanBsp.scala 103:26]
    reg resetModeQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), resetModeQ) @[CanBsp.scala 105:34]
    reg bitCnt : UInt<6>, clock with :
      reset => (UInt<1>("h0"), bitCnt) @[CanBsp.scala 106:30]
    reg dataLen : UInt<4>, clock with :
      reset => (UInt<1>("h0"), dataLen) @[CanBsp.scala 107:31]
    reg id : UInt<29>, clock with :
      reset => (UInt<1>("h0"), id) @[CanBsp.scala 108:26]
    reg bitStuffCnt : UInt<3>, clock with :
      reset => (UInt<1>("h0"), bitStuffCnt) @[CanBsp.scala 109:35]
    reg bitStuffCntTx : UInt<3>, clock with :
      reset => (UInt<1>("h0"), bitStuffCntTx) @[CanBsp.scala 110:37]
    node _txPointQ_T = mux(io_resetMode, UInt<1>("h0"), io_txPoint) @[CanBsp.scala 111:36]
    reg txPointQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), txPointQ) @[CanBsp.scala 111:32]
    reg rxId1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxId1) @[CanBsp.scala 112:29]
    reg rxRtr1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxRtr1) @[CanBsp.scala 113:30]
    reg rxIde : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxIde) @[CanBsp.scala 114:29]
    reg rxId2 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxId2) @[CanBsp.scala 115:29]
    reg rxRtr2 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxRtr2) @[CanBsp.scala 116:30]
    reg rxR0 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxR0) @[CanBsp.scala 117:28]
    reg rxR1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxR1) @[CanBsp.scala 118:27]
    reg rxDlc : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxDlc) @[CanBsp.scala 119:29]
    reg rxData : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxData) @[CanBsp.scala 120:30]
    reg rxCrc : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxCrc) @[CanBsp.scala 121:29]
    reg rxCrcLim : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxCrcLim) @[CanBsp.scala 122:32]
    reg rxAck : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxAck) @[CanBsp.scala 123:29]
    reg rxAckLim : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxAckLim) @[CanBsp.scala 124:32]
    reg rxEof : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxEof) @[CanBsp.scala 125:29]
    reg goEarlyTxLatched : UInt<1>, clock with :
      reset => (UInt<1>("h0"), goEarlyTxLatched) @[CanBsp.scala 126:40]
    reg rtr1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rtr1) @[CanBsp.scala 127:28]
    reg ide : UInt<1>, clock with :
      reset => (UInt<1>("h0"), ide) @[CanBsp.scala 128:27]
    reg rtr2 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rtr2) @[CanBsp.scala 129:28]
    reg crcIn : UInt<15>, clock with :
      reset => (UInt<1>("h0"), crcIn) @[CanBsp.scala 130:29]
    reg tmpData : UInt<8>, clock with :
      reset => (UInt<1>("h0"), tmpData) @[CanBsp.scala 131:31]
    reg writeDataToTmpFifo : UInt<1>, clock with :
      reset => (UInt<1>("h0"), writeDataToTmpFifo) @[CanBsp.scala 137:42]
    reg byteCnt : UInt<3>, clock with :
      reset => (UInt<1>("h0"), byteCnt) @[CanBsp.scala 138:31]
    reg bitStuffCntEn : UInt<1>, clock with :
      reset => (UInt<1>("h0"), bitStuffCntEn) @[CanBsp.scala 139:37]
    reg crcEnable : UInt<1>, clock with :
      reset => (UInt<1>("h0"), crcEnable) @[CanBsp.scala 140:33]
    reg eofCnt : UInt<3>, clock with :
      reset => (UInt<1>("h0"), eofCnt) @[CanBsp.scala 141:30]
    reg passiveCnt : UInt<3>, clock with :
      reset => (UInt<1>("h0"), passiveCnt) @[CanBsp.scala 142:34]
    reg errorFrame : UInt<1>, clock with :
      reset => (UInt<1>("h0"), errorFrame) @[CanBsp.scala 143:34]
    reg enableErrorCnt2 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), enableErrorCnt2) @[CanBsp.scala 144:39]
    reg errorCnt1 : UInt<3>, clock with :
      reset => (UInt<1>("h0"), errorCnt1) @[CanBsp.scala 145:33]
    reg errorCnt2 : UInt<3>, clock with :
      reset => (UInt<1>("h0"), errorCnt2) @[CanBsp.scala 146:33]
    reg delayedDominantCnt : UInt<3>, clock with :
      reset => (UInt<1>("h0"), delayedDominantCnt) @[CanBsp.scala 147:42]
    reg enableOverloadCnt2 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), enableOverloadCnt2) @[CanBsp.scala 148:42]
    skip
    skip
    reg overloadCnt1 : UInt<3>, clock with :
      reset => (UInt<1>("h0"), overloadCnt1) @[CanBsp.scala 151:36]
    reg overloadCnt2 : UInt<3>, clock with :
      reset => (UInt<1>("h0"), overloadCnt2) @[CanBsp.scala 152:36]
    reg crcErr : UInt<1>, clock with :
      reset => (UInt<1>("h0"), crcErr) @[CanBsp.scala 153:30]
    reg arbitrationLost : UInt<1>, clock with :
      reset => (UInt<1>("h0"), arbitrationLost) @[CanBsp.scala 154:39]
    reg arbitrationLostQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), arbitrationLostQ) @[CanBsp.scala 155:40]
    reg arbitrationFieldD : UInt<1>, clock with :
      reset => (UInt<1>("h0"), arbitrationFieldD) @[CanBsp.scala 156:41]
    reg arbitrationCnt : UInt<5>, clock with :
      reset => (UInt<1>("h0"), arbitrationCnt) @[CanBsp.scala 157:38]
    reg arbitrationBlocked : UInt<1>, clock with :
      reset => (UInt<1>("h0"), arbitrationBlocked) @[CanBsp.scala 158:42]
    reg txQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), txQ) @[CanBsp.scala 159:27]
    reg dataCnt : UInt<4>, clock with :
      reset => (UInt<1>("h0"), dataCnt) @[CanBsp.scala 160:31]
    reg headerCnt : UInt<3>, clock with :
      reset => (UInt<1>("h0"), headerCnt) @[CanBsp.scala 161:33]
    reg wrFifo : UInt<1>, clock with :
      reset => (UInt<1>("h0"), wrFifo) @[CanBsp.scala 162:30]
    reg txPointer : UInt<6>, clock with :
      reset => (UInt<1>("h0"), txPointer) @[CanBsp.scala 164:33]
    reg finishMsg : UInt<1>, clock with :
      reset => (UInt<1>("h0"), finishMsg) @[CanBsp.scala 166:33]
    reg busFreeCnt : UInt<4>, clock with :
      reset => (UInt<1>("h0"), busFreeCnt) @[CanBsp.scala 167:34]
    reg busFreeCntEn : UInt<1>, clock with :
      reset => (UInt<1>("h0"), busFreeCntEn) @[CanBsp.scala 168:36]
    reg waitingForBusFree : UInt<1>, clock with :
      reset => (UInt<1>("h0"), waitingForBusFree) @[CanBsp.scala 169:41]
    node _busFree_T = and(io_samplePoint, io_sampledBit) @[CanBsp.scala 170:47]
    node _busFree_T_1 = eq(busFreeCnt, UInt<4>("ha")) @[CanBsp.scala 170:77]
    node _busFree_T_2 = and(_busFree_T, _busFree_T_1) @[CanBsp.scala 170:63]
    node _busFree_T_3 = and(_busFree_T_2, waitingForBusFree) @[CanBsp.scala 170:87]
    reg busFree : UInt<1>, clock with :
      reset => (UInt<1>("h0"), busFree) @[CanBsp.scala 170:31]
    reg nodeBusOffQ : UInt<1>, clock with :
      reset => (UInt<1>("h0"), nodeBusOffQ) @[CanBsp.scala 171:35]
    reg ackErrLatched : UInt<1>, clock with :
      reset => (UInt<1>("h0"), ackErrLatched) @[CanBsp.scala 172:37]
    reg bitErrLatched : UInt<1>, clock with :
      reset => (UInt<1>("h0"), bitErrLatched) @[CanBsp.scala 173:37]
    reg stuffErrLatched : UInt<1>, clock with :
      reset => (UInt<1>("h0"), stuffErrLatched) @[CanBsp.scala 174:39]
    reg formErrLatched : UInt<1>, clock with :
      reset => (UInt<1>("h0"), formErrLatched) @[CanBsp.scala 175:38]
    reg rule3Exc1_0 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rule3Exc1_0) @[CanBsp.scala 176:38]
    reg rule3Exc1_1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rule3Exc1_1) @[CanBsp.scala 176:38]
    reg suspend : UInt<1>, clock with :
      reset => (UInt<1>("h0"), suspend) @[CanBsp.scala 177:31]
    reg suspendCntEn : UInt<1>, clock with :
      reset => (UInt<1>("h0"), suspendCntEn) @[CanBsp.scala 178:36]
    reg suspendCnt : UInt<3>, clock with :
      reset => (UInt<1>("h0"), suspendCnt) @[CanBsp.scala 179:34]
    reg errorFlagOverLatched : UInt<1>, clock with :
      reset => (UInt<1>("h0"), errorFlagOverLatched) @[CanBsp.scala 180:44]
    reg errorCaptureCodeBlocked : UInt<1>, clock with :
      reset => (UInt<1>("h0"), errorCaptureCodeBlocked) @[CanBsp.scala 182:47]
    reg firstCompareBit : UInt<1>, clock with :
      reset => (UInt<1>("h0"), firstCompareBit) @[CanBsp.scala 183:39]
    node errorCaptureCodeDirection = not(io_transmitting) @[CanBsp.scala 186:42]
    node bitDeStuff = eq(bitStuffCnt, UInt<3>("h5")) @[CanBsp.scala 188:39]
    node bitDeStuffTx = eq(bitStuffCntTx, UInt<3>("h5")) @[CanBsp.scala 189:43]
    node _lastBitOfInter_T = bits(bitCnt, 1, 0) @[CanBsp.scala 192:48]
    node _lastBitOfInter_T_1 = eq(_lastBitOfInter_T, UInt<2>("h2")) @[CanBsp.scala 192:54]
    node lastBitOfInter = and(rxInter, _lastBitOfInter_T_1) @[CanBsp.scala 192:39]
    skip
    node _goRxIdle_T_1 = and(_busFree_T, lastBitOfInter) @[CanBsp.scala 194:56]
    node _goRxIdle_T_2 = not(io_nodeBusOff) @[CanBsp.scala 194:85]
    node _goRxIdle_T_3 = and(busFree, _goRxIdle_T_2) @[CanBsp.scala 194:83]
    node goRxIdle = or(_goRxIdle_T_1, _goRxIdle_T_3) @[CanBsp.scala 194:73]
    node _goRxId1_T = not(io_sampledBit) @[CanBsp.scala 195:41]
    node _goRxId1_T_1 = and(io_samplePoint, _goRxId1_T) @[CanBsp.scala 195:39]
    node _goRxId1_T_2 = or(io_rxIdle, lastBitOfInter) @[CanBsp.scala 195:69]
    node goRxId1 = and(_goRxId1_T_1, _goRxId1_T_2) @[CanBsp.scala 195:56]
    node _goRxRtr1_T = not(bitDeStuff) @[CanBsp.scala 196:26]
    node _goRxRtr1_T_1 = and(_goRxRtr1_T, io_samplePoint) @[CanBsp.scala 196:39]
    node _goRxRtr1_T_2 = and(_goRxRtr1_T_1, rxId1) @[CanBsp.scala 196:56]
    node _goRxRtr1_T_3 = bits(bitCnt, 3, 0) @[CanBsp.scala 196:73]
    node _goRxRtr1_T_4 = eq(_goRxRtr1_T_3, UInt<4>("ha")) @[CanBsp.scala 196:79]
    node goRxRtr1 = and(_goRxRtr1_T_2, _goRxRtr1_T_4) @[CanBsp.scala 196:64]
    skip
    skip
    node goRxIde = and(_goRxRtr1_T_1, rxRtr1) @[CanBsp.scala 197:55]
    skip
    skip
    node _goRxId2_T_2 = and(_goRxRtr1_T_1, rxIde) @[CanBsp.scala 198:55]
    node goRxId2 = and(_goRxId2_T_2, io_sampledBit) @[CanBsp.scala 198:63]
    skip
    skip
    node _goRxRtr2_T_2 = and(_goRxRtr1_T_1, rxId2) @[CanBsp.scala 199:56]
    node _goRxRtr2_T_3 = bits(bitCnt, 4, 0) @[CanBsp.scala 199:73]
    node _goRxRtr2_T_4 = eq(_goRxRtr2_T_3, UInt<5>("h11")) @[CanBsp.scala 199:79]
    node goRxRtr2 = and(_goRxRtr2_T_2, _goRxRtr2_T_4) @[CanBsp.scala 199:64]
    skip
    skip
    node goRxR1 = and(_goRxRtr1_T_1, rxRtr2) @[CanBsp.scala 200:54]
    skip
    skip
    skip
    node _goRxR0_T_3 = and(rxIde, _goRxId1_T) @[CanBsp.scala 201:63]
    node _goRxR0_T_4 = or(_goRxR0_T_3, rxR1) @[CanBsp.scala 201:82]
    node goRxR0 = and(_goRxRtr1_T_1, _goRxR0_T_4) @[CanBsp.scala 201:54]
    skip
    skip
    node goRxDlc = and(_goRxRtr1_T_1, rxR0) @[CanBsp.scala 202:55]
    skip
    skip
    node _goRxCrcLim_T_2 = and(_goRxRtr1_T_1, rxCrc) @[CanBsp.scala 205:58]
    skip
    node _goRxCrcLim_T_4 = eq(_goRxRtr1_T_3, UInt<4>("he")) @[CanBsp.scala 205:81]
    node goRxCrcLim = and(_goRxCrcLim_T_2, _goRxCrcLim_T_4) @[CanBsp.scala 205:66]
    skip
    skip
    node goRxAck = and(_goRxRtr1_T_1, rxCrcLim) @[CanBsp.scala 206:55]
    node goRxAckLim = and(io_samplePoint, rxAck) @[CanBsp.scala 207:42]
    node goRxEof = and(io_samplePoint, rxAckLim) @[CanBsp.scala 208:39]
    node _remoteRq_T = not(ide) @[CanBsp.scala 209:27]
    node _remoteRq_T_1 = and(_remoteRq_T, rtr1) @[CanBsp.scala 209:33]
    node _remoteRq_T_2 = and(ide, rtr2) @[CanBsp.scala 209:48]
    node remoteRq = or(_remoteRq_T_1, _remoteRq_T_2) @[CanBsp.scala 209:41]
    node goCrcEnable = or(io_hardSync, io_goTx) @[CanBsp.scala 211:40]
    node _bitDeStuffSet_T = not(io_goErrorFrame) @[CanBsp.scala 214:41]
    node bitDeStuffSet = and(goRxId1, _bitDeStuffSet_T) @[CanBsp.scala 214:38]
    node _bitDeStuffReset_T = or(goRxAck, io_goErrorFrame) @[CanBsp.scala 215:40]
    node bitDeStuffReset = or(_bitDeStuffReset_T, io_goOverloadFrame) @[CanBsp.scala 215:58]
    node _goEarlyTx_T = not(io_listenOnlyMode) @[CanBsp.scala 217:27]
    node _goEarlyTx_T_1 = and(_goEarlyTx_T, io_needToTx) @[CanBsp.scala 217:47]
    node _goEarlyTx_T_2 = not(io_txState) @[CanBsp.scala 217:64]
    node _goEarlyTx_T_3 = and(_goEarlyTx_T_1, _goEarlyTx_T_2) @[CanBsp.scala 217:61]
    node _goEarlyTx_T_4 = not(suspend) @[CanBsp.scala 217:80]
    node _goEarlyTx_T_5 = eq(suspendCnt, UInt<3>("h7")) @[CanBsp.scala 217:103]
    node _goEarlyTx_T_6 = or(_goEarlyTx_T_4, _goEarlyTx_T_5) @[CanBsp.scala 217:89]
    node _goEarlyTx_T_7 = and(_goEarlyTx_T_3, _goEarlyTx_T_6) @[CanBsp.scala 217:77]
    node _goEarlyTx_T_8 = and(_goEarlyTx_T_7, io_samplePoint) @[CanBsp.scala 217:113]
    skip
    node _goEarlyTx_T_10 = and(_goEarlyTx_T_8, _goRxId1_T) @[CanBsp.scala 217:130]
    skip
    node goEarlyTx = and(_goEarlyTx_T_10, _goRxId1_T_2) @[CanBsp.scala 217:149]
    node calculatedCrc = canCrcRx.io_crc @[CanBsp.scala 219:34 525:17]
    node _rCalculatedCrc_T = bits(calculatedCrc, 7, 0) @[Bitwise.scala 109:18]
    skip
    skip
    node _rCalculatedCrc_T_3 = shr(_rCalculatedCrc_T, 4) @[Bitwise.scala 103:21]
    node _GEN_191 = pad(_rCalculatedCrc_T_3, 8) @[Bitwise.scala 103:31]
    node _rCalculatedCrc_T_4 = and(_GEN_191, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rCalculatedCrc_T_5 = bits(_rCalculatedCrc_T, 3, 0) @[Bitwise.scala 103:46]
    node _rCalculatedCrc_T_6 = shl(_rCalculatedCrc_T_5, 4) @[Bitwise.scala 103:65]
    skip
    node _rCalculatedCrc_T_8 = and(_rCalculatedCrc_T_6, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rCalculatedCrc_T_9 = or(_rCalculatedCrc_T_4, _rCalculatedCrc_T_8) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rCalculatedCrc_T_13 = shr(_rCalculatedCrc_T_9, 2) @[Bitwise.scala 103:21]
    node _GEN_192 = pad(_rCalculatedCrc_T_13, 8) @[Bitwise.scala 103:31]
    node _rCalculatedCrc_T_14 = and(_GEN_192, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rCalculatedCrc_T_15 = bits(_rCalculatedCrc_T_9, 5, 0) @[Bitwise.scala 103:46]
    node _rCalculatedCrc_T_16 = shl(_rCalculatedCrc_T_15, 2) @[Bitwise.scala 103:65]
    skip
    node _rCalculatedCrc_T_18 = and(_rCalculatedCrc_T_16, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rCalculatedCrc_T_19 = or(_rCalculatedCrc_T_14, _rCalculatedCrc_T_18) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rCalculatedCrc_T_23 = shr(_rCalculatedCrc_T_19, 1) @[Bitwise.scala 103:21]
    node _GEN_193 = pad(_rCalculatedCrc_T_23, 8) @[Bitwise.scala 103:31]
    node _rCalculatedCrc_T_24 = and(_GEN_193, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rCalculatedCrc_T_25 = bits(_rCalculatedCrc_T_19, 6, 0) @[Bitwise.scala 103:46]
    node _rCalculatedCrc_T_26 = shl(_rCalculatedCrc_T_25, 1) @[Bitwise.scala 103:65]
    skip
    node _rCalculatedCrc_T_28 = and(_rCalculatedCrc_T_26, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rCalculatedCrc_hi = or(_rCalculatedCrc_T_24, _rCalculatedCrc_T_28) @[Bitwise.scala 103:39]
    node _rCalculatedCrc_T_29 = bits(calculatedCrc, 14, 8) @[Bitwise.scala 109:44]
    node _rCalculatedCrc_T_30 = bits(_rCalculatedCrc_T_29, 3, 0) @[Bitwise.scala 109:18]
    node _rCalculatedCrc_T_31 = bits(_rCalculatedCrc_T_30, 1, 0) @[Bitwise.scala 109:18]
    node rCalculatedCrc_hi_1 = bits(_rCalculatedCrc_T_31, 0, 0) @[Bitwise.scala 109:18]
    node rCalculatedCrc_lo = bits(_rCalculatedCrc_T_31, 1, 1) @[Bitwise.scala 109:44]
    node rCalculatedCrc_hi_2 = cat(rCalculatedCrc_hi_1, rCalculatedCrc_lo) @[Cat.scala 30:58]
    node _rCalculatedCrc_T_32 = bits(_rCalculatedCrc_T_30, 3, 2) @[Bitwise.scala 109:44]
    node rCalculatedCrc_hi_3 = bits(_rCalculatedCrc_T_32, 0, 0) @[Bitwise.scala 109:18]
    node rCalculatedCrc_lo_1 = bits(_rCalculatedCrc_T_32, 1, 1) @[Bitwise.scala 109:44]
    node rCalculatedCrc_lo_2 = cat(rCalculatedCrc_hi_3, rCalculatedCrc_lo_1) @[Cat.scala 30:58]
    node rCalculatedCrc_hi_4 = cat(rCalculatedCrc_hi_2, rCalculatedCrc_lo_2) @[Cat.scala 30:58]
    node _rCalculatedCrc_T_33 = bits(_rCalculatedCrc_T_29, 6, 4) @[Bitwise.scala 109:44]
    node _rCalculatedCrc_T_34 = bits(_rCalculatedCrc_T_33, 1, 0) @[Bitwise.scala 109:18]
    node rCalculatedCrc_hi_5 = bits(_rCalculatedCrc_T_34, 0, 0) @[Bitwise.scala 109:18]
    node rCalculatedCrc_lo_3 = bits(_rCalculatedCrc_T_34, 1, 1) @[Bitwise.scala 109:44]
    node rCalculatedCrc_hi_6 = cat(rCalculatedCrc_hi_5, rCalculatedCrc_lo_3) @[Cat.scala 30:58]
    node rCalculatedCrc_lo_4 = bits(_rCalculatedCrc_T_33, 2, 2) @[Bitwise.scala 109:44]
    node rCalculatedCrc_lo_5 = cat(rCalculatedCrc_hi_6, rCalculatedCrc_lo_4) @[Cat.scala 30:58]
    node rCalculatedCrc_lo_6 = cat(rCalculatedCrc_hi_4, rCalculatedCrc_lo_5) @[Cat.scala 30:58]
    node rCalculatedCrc = cat(rCalculatedCrc_hi, rCalculatedCrc_lo_6) @[Cat.scala 30:58]
    node _limitedDataLen_T = lt(dataLen, UInt<4>("h8")) @[CanBsp.scala 221:43]
    node limitedDataLen = mux(_limitedDataLen_T, dataLen, UInt<4>("h8")) @[CanBsp.scala 221:34]
    skip
    skip
    node _formErr_T_1 = and(_goRxRtr1_T, rxCrcLim) @[CanBsp.scala 222:57]
    skip
    node _formErr_T_3 = and(_formErr_T_1, _goRxId1_T) @[CanBsp.scala 222:68]
    skip
    node _formErr_T_5 = and(rxAckLim, _goRxId1_T) @[CanBsp.scala 223:15]
    node _formErr_T_6 = or(_formErr_T_3, _formErr_T_5) @[CanBsp.scala 222:88]
    node _formErr_T_7 = lt(eofCnt, UInt<3>("h6")) @[CanBsp.scala 223:46]
    node _formErr_T_8 = and(_formErr_T_7, rxEof) @[CanBsp.scala 223:52]
    skip
    node _formErr_T_10 = and(_formErr_T_8, _goRxId1_T) @[CanBsp.scala 223:60]
    node _formErr_T_11 = not(io_transmitter) @[CanBsp.scala 223:82]
    node _formErr_T_12 = and(_formErr_T_10, _formErr_T_11) @[CanBsp.scala 223:79]
    node _formErr_T_13 = or(_formErr_T_6, _formErr_T_12) @[CanBsp.scala 223:35]
    skip
    node _formErr_T_15 = and(rxEof, _goRxId1_T) @[CanBsp.scala 223:109]
    node _formErr_T_16 = and(_formErr_T_15, io_transmitter) @[CanBsp.scala 223:128]
    node _formErr_T_17 = or(_formErr_T_13, _formErr_T_16) @[CanBsp.scala 223:100]
    node formErr = and(io_samplePoint, _formErr_T_17) @[CanBsp.scala 222:39]
    skip
    node _bitErrCompGoRxCrc_T_1 = bits(limitedDataLen, 2, 0) @[CanBsp.scala 225:66]
    node _bitErrCompGoRxCrc_T_2 = shl(_bitErrCompGoRxCrc_T_1, 3) @[CanBsp.scala 225:71]
    node _bitErrCompGoRxCrc_T_3 = sub(_bitErrCompGoRxCrc_T_2, UInt<6>("h1")) @[CanBsp.scala 225:85]
    node _bitErrCompGoRxCrc_T_4 = tail(_bitErrCompGoRxCrc_T_3, 1) @[CanBsp.scala 225:85]
    node bitErrCompGoRxCrc = eq(bitCnt, _bitErrCompGoRxCrc_T_4) @[CanBsp.scala 225:46]
    skip
    skip
    node _goRxData_T_2 = and(_goRxRtr1_T_1, rxDlc) @[CanBsp.scala 226:46]
    skip
    node _goRxData_T_4 = eq(_lastBitOfInter_T, UInt<2>("h3")) @[CanBsp.scala 226:69]
    node _goRxData_T_5 = and(_goRxData_T_2, _goRxData_T_4) @[CanBsp.scala 226:54]
    node _goRxData_T_6 = bits(dataLen, 2, 0) @[CanBsp.scala 226:104]
    node _goRxData_T_7 = orr(_goRxData_T_6) @[CanBsp.scala 226:113]
    node _goRxData_T_8 = or(io_sampledBit, _goRxData_T_7) @[CanBsp.scala 226:95]
    node _goRxData_T_9 = and(_goRxData_T_5, _goRxData_T_8) @[CanBsp.scala 226:78]
    node _goRxData_T_10 = not(remoteRq) @[CanBsp.scala 226:120]
    node goRxData = and(_goRxData_T_9, _goRxData_T_10) @[CanBsp.scala 226:117]
    skip
    skip
    skip
    skip
    node _goRxCrc_T_4 = and(rxDlc, _goRxData_T_4) @[CanBsp.scala 227:54]
    skip
    skip
    skip
    node _goRxCrc_T_8 = not(_goRxData_T_7) @[CanBsp.scala 227:101]
    node _goRxCrc_T_9 = and(_goRxId1_T, _goRxCrc_T_8) @[CanBsp.scala 227:98]
    node _goRxCrc_T_10 = or(_goRxCrc_T_9, remoteRq) @[CanBsp.scala 227:122]
    node _goRxCrc_T_11 = and(_goRxCrc_T_4, _goRxCrc_T_10) @[CanBsp.scala 227:78]
    node _goRxCrc_T_12 = and(rxData, bitErrCompGoRxCrc) @[CanBsp.scala 227:143]
    node _goRxCrc_T_13 = or(_goRxCrc_T_11, _goRxCrc_T_12) @[CanBsp.scala 227:134]
    node goRxCrc = and(_goRxRtr1_T_1, _goRxCrc_T_13) @[CanBsp.scala 227:45]
    node _errorFrameEnded_T = andr(errorCnt2) @[CanBsp.scala 229:47]
    node errorFrameEnded = and(_errorFrameEnded_T, io_txPoint) @[CanBsp.scala 229:51]
    node _overloadFrameEnded_T = eq(overloadCnt2, UInt<3>("h7")) @[CanBsp.scala 230:49]
    node overloadFrameEnded = and(_overloadFrameEnded_T, io_txPoint) @[CanBsp.scala 230:58]
    node _ackErr_T = and(rxAck, io_samplePoint) @[CanBsp.scala 232:29]
    node _ackErr_T_1 = and(_ackErr_T, io_sampledBit) @[CanBsp.scala 232:46]
    node _ackErr_T_2 = and(_ackErr_T_1, io_txState) @[CanBsp.scala 232:62]
    node _ackErr_T_3 = not(io_selfTestMode) @[CanBsp.scala 232:78]
    node ackErr = and(_ackErr_T_2, _ackErr_T_3) @[CanBsp.scala 232:75]
    node _stuffErr_T = and(io_samplePoint, bitStuffCntEn) @[CanBsp.scala 233:40]
    node _stuffErr_T_1 = and(_stuffErr_T, bitDeStuff) @[CanBsp.scala 233:56]
    node _stuffErr_T_2 = eq(io_sampledBit, io_sampledBitQ) @[CanBsp.scala 233:86]
    node stuffErr = and(_stuffErr_T_1, _stuffErr_T_2) @[CanBsp.scala 233:69]
    node _io_goRxInter_T = and(io_samplePoint, rxEof) @[CanBsp.scala 235:36]
    node _io_goRxInter_T_1 = eq(eofCnt, UInt<3>("h6")) @[CanBsp.scala 235:56]
    node _io_goRxInter_T_2 = and(_io_goRxInter_T, _io_goRxInter_T_1) @[CanBsp.scala 235:46]
    node _io_goRxInter_T_3 = or(_io_goRxInter_T_2, errorFrameEnded) @[CanBsp.scala 235:66]
    skip
    skip
    skip
    node _noByte0_T = lt(dataLen, UInt<4>("h1")) @[CanBsp.scala 238:40]
    skip
    node _noByte1_T = lt(dataLen, UInt<4>("h2")) @[CanBsp.scala 239:40]
    skip
    node _headerLen_T = mux(ide, UInt<3>("h5"), UInt<3>("h3")) @[CanBsp.scala 241:50]
    node headerLen = mux(io_extendedMode, _headerLen_T, UInt<3>("h2")) @[CanBsp.scala 241:29]
    node storingHeader = lt(headerCnt, headerLen) @[CanBsp.scala 242:40]
    skip
    node _limitedDataLenSubOne_T_1 = sub(dataLen, UInt<4>("h1")) @[CanBsp.scala 243:85]
    node _limitedDataLenSubOne_T_2 = tail(_limitedDataLenSubOne_T_1, 1) @[CanBsp.scala 243:85]
    node _limitedDataLenSubOne_T_3 = mux(_limitedDataLen_T, _limitedDataLenSubOne_T_2, UInt<4>("h7")) @[CanBsp.scala 243:61]
    node limitedDataLenSubOne = mux(remoteRq, UInt<4>("hf"), _limitedDataLenSubOne_T_3) @[CanBsp.scala 243:40]
    node _GEN_194 = pad(headerLen, 4) @[CanBsp.scala 244:63]
    node _resetWrFifo_T = add(limitedDataLenSubOne, _GEN_194) @[CanBsp.scala 244:63]
    node _GEN_195 = pad(dataCnt, 5) @[CanBsp.scala 244:37]
    node _resetWrFifo_T_1 = eq(_GEN_195, _resetWrFifo_T) @[CanBsp.scala 244:37]
    node resetWrFifo = or(_resetWrFifo_T_1, io_resetMode) @[CanBsp.scala 244:78]
    node _err_T = or(formErr, stuffErr) @[CanBsp.scala 245:28]
    node _bitErr_T = or(io_txState, errorFrame) @[CanBsp.scala 270:25]
    node _bitErr_T_1 = or(_bitErr_T, io_overloadFrame) @[CanBsp.scala 270:38]
    node _bitErr_T_2 = or(_bitErr_T_1, rxAck) @[CanBsp.scala 270:57]
    node _bitErr_T_3 = and(_bitErr_T_2, io_samplePoint) @[CanBsp.scala 270:66]
    node _bitErr_T_4 = neq(io_tx, io_sampledBit) @[CanBsp.scala 270:92]
    node _bitErr_T_5 = and(_bitErr_T_3, _bitErr_T_4) @[CanBsp.scala 270:83]
    node _arbitrationField_T = or(rxId1, rxRtr1) @[CanBsp.scala 247:39]
    node _arbitrationField_T_1 = or(_arbitrationField_T, rxIde) @[CanBsp.scala 247:48]
    node _arbitrationField_T_2 = or(_arbitrationField_T_1, rxId2) @[CanBsp.scala 247:56]
    node arbitrationField = or(_arbitrationField_T_2, rxRtr2) @[CanBsp.scala 247:64]
    node _bitErrExc1_T = and(io_txState, arbitrationField) @[CanBsp.scala 262:38]
    node bitErrExc1 = and(_bitErrExc1_T, io_tx) @[CanBsp.scala 262:57]
    node _bitErr_T_6 = not(bitErrExc1) @[CanBsp.scala 270:114]
    node _bitErr_T_7 = and(_bitErr_T_5, _bitErr_T_6) @[CanBsp.scala 270:111]
    node bitErrExc2 = and(rxAck, io_tx) @[CanBsp.scala 263:33]
    node _bitErr_T_8 = not(bitErrExc2) @[CanBsp.scala 270:130]
    node _bitErr_T_9 = and(_bitErr_T_7, _bitErr_T_8) @[CanBsp.scala 270:127]
    node _bitErrExc3_T = and(errorFrame, io_nodeErrorPassive) @[CanBsp.scala 264:38]
    node _bitErrExc3_T_1 = lt(errorCnt1, UInt<3>("h7")) @[CanBsp.scala 264:73]
    node bitErrExc3 = and(_bitErrExc3_T, _bitErrExc3_T_1) @[CanBsp.scala 264:60]
    node _bitErr_T_10 = not(bitErrExc3) @[CanBsp.scala 270:146]
    node _bitErr_T_11 = and(_bitErr_T_9, _bitErr_T_10) @[CanBsp.scala 270:143]
    node _bitErrExc4_T = andr(errorCnt1) @[CanBsp.scala 265:56]
    node _bitErrExc4_T_1 = and(errorFrame, _bitErrExc4_T) @[CanBsp.scala 265:39]
    node _bitErrExc4_T_2 = not(enableErrorCnt2) @[CanBsp.scala 265:63]
    node _bitErrExc4_T_3 = and(_bitErrExc4_T_1, _bitErrExc4_T_2) @[CanBsp.scala 265:60]
    node _bitErrExc4_T_4 = eq(overloadCnt1, UInt<3>("h7")) @[CanBsp.scala 265:118]
    node _bitErrExc4_T_5 = and(io_overloadFrame, _bitErrExc4_T_4) @[CanBsp.scala 265:102]
    node _bitErrExc4_T_6 = not(enableOverloadCnt2) @[CanBsp.scala 265:130]
    node _bitErrExc4_T_7 = and(_bitErrExc4_T_5, _bitErrExc4_T_6) @[CanBsp.scala 265:127]
    node bitErrExc4 = or(_bitErrExc4_T_3, _bitErrExc4_T_7) @[CanBsp.scala 265:82]
    node _bitErr_T_12 = not(bitErrExc4) @[CanBsp.scala 270:162]
    node _bitErr_T_13 = and(_bitErr_T_11, _bitErr_T_12) @[CanBsp.scala 270:159]
    skip
    node _bitErrExc5_T_1 = and(errorFrame, _errorFrameEnded_T) @[CanBsp.scala 266:39]
    skip
    node _bitErrExc5_T_3 = and(io_overloadFrame, _overloadFrameEnded_T) @[CanBsp.scala 266:81]
    node bitErrExc5 = or(_bitErrExc5_T_1, _bitErrExc5_T_3) @[CanBsp.scala 266:61]
    node _bitErr_T_14 = not(bitErrExc5) @[CanBsp.scala 270:178]
    node _bitErr_T_15 = and(_bitErr_T_13, _bitErr_T_14) @[CanBsp.scala 270:175]
    skip
    node _bitErrExc6_T_1 = and(_io_goRxInter_T_1, rxEof) @[CanBsp.scala 267:44]
    skip
    node bitErrExc6 = and(_bitErrExc6_T_1, _formErr_T_11) @[CanBsp.scala 267:52]
    node _bitErr_T_16 = not(bitErrExc6) @[CanBsp.scala 270:194]
    node _bitErr_T_17 = and(_bitErr_T_15, _bitErr_T_16) @[CanBsp.scala 270:191]
    node _bitErr_T_18 = not(io_resetMode) @[CanBsp.scala 270:210]
    node bitErr = and(_bitErr_T_17, _bitErr_T_18) @[CanBsp.scala 270:207]
    skip
    node _err_T_1 = or(_err_T, bitErr) @[CanBsp.scala 245:39]
    node _err_T_2 = or(_err_T_1, ackErr) @[CanBsp.scala 245:48]
    node _err_T_3 = or(_err_T_2, formErrLatched) @[CanBsp.scala 245:57]
    node _err_T_4 = or(_err_T_3, stuffErrLatched) @[CanBsp.scala 245:74]
    node _err_T_5 = or(_err_T_4, bitErrLatched) @[CanBsp.scala 245:92]
    node _err_T_6 = or(_err_T_5, ackErrLatched) @[CanBsp.scala 245:108]
    node err = or(_err_T_6, crcErr) @[CanBsp.scala 245:124]
    skip
    skip
    node _rTxData_T_2 = shr(io_txData_0, 4) @[Bitwise.scala 103:21]
    node _GEN_196 = pad(_rTxData_T_2, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_3 = and(_GEN_196, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rTxData_T_4 = bits(io_txData_0, 3, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_5 = shl(_rTxData_T_4, 4) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_7 = and(_rTxData_T_5, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rTxData_T_8 = or(_rTxData_T_3, _rTxData_T_7) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_12 = shr(_rTxData_T_8, 2) @[Bitwise.scala 103:21]
    node _GEN_197 = pad(_rTxData_T_12, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_13 = and(_GEN_197, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rTxData_T_14 = bits(_rTxData_T_8, 5, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_15 = shl(_rTxData_T_14, 2) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_17 = and(_rTxData_T_15, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rTxData_T_18 = or(_rTxData_T_13, _rTxData_T_17) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_22 = shr(_rTxData_T_18, 1) @[Bitwise.scala 103:21]
    node _GEN_198 = pad(_rTxData_T_22, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_23 = and(_GEN_198, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rTxData_T_24 = bits(_rTxData_T_18, 6, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_25 = shl(_rTxData_T_24, 1) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_27 = and(_rTxData_T_25, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rTxData_0 = or(_rTxData_T_23, _rTxData_T_27) @[Bitwise.scala 103:39]
    skip
    skip
    node _rTxData_T_31 = shr(io_txData_1, 4) @[Bitwise.scala 103:21]
    node _GEN_199 = pad(_rTxData_T_31, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_32 = and(_GEN_199, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rTxData_T_33 = bits(io_txData_1, 3, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_34 = shl(_rTxData_T_33, 4) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_36 = and(_rTxData_T_34, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rTxData_T_37 = or(_rTxData_T_32, _rTxData_T_36) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_41 = shr(_rTxData_T_37, 2) @[Bitwise.scala 103:21]
    node _GEN_200 = pad(_rTxData_T_41, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_42 = and(_GEN_200, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rTxData_T_43 = bits(_rTxData_T_37, 5, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_44 = shl(_rTxData_T_43, 2) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_46 = and(_rTxData_T_44, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rTxData_T_47 = or(_rTxData_T_42, _rTxData_T_46) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_51 = shr(_rTxData_T_47, 1) @[Bitwise.scala 103:21]
    node _GEN_201 = pad(_rTxData_T_51, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_52 = and(_GEN_201, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rTxData_T_53 = bits(_rTxData_T_47, 6, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_54 = shl(_rTxData_T_53, 1) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_56 = and(_rTxData_T_54, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rTxData_1 = or(_rTxData_T_52, _rTxData_T_56) @[Bitwise.scala 103:39]
    skip
    skip
    node _rTxData_T_60 = shr(io_txData_2, 4) @[Bitwise.scala 103:21]
    node _GEN_202 = pad(_rTxData_T_60, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_61 = and(_GEN_202, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rTxData_T_62 = bits(io_txData_2, 3, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_63 = shl(_rTxData_T_62, 4) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_65 = and(_rTxData_T_63, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rTxData_T_66 = or(_rTxData_T_61, _rTxData_T_65) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_70 = shr(_rTxData_T_66, 2) @[Bitwise.scala 103:21]
    node _GEN_203 = pad(_rTxData_T_70, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_71 = and(_GEN_203, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rTxData_T_72 = bits(_rTxData_T_66, 5, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_73 = shl(_rTxData_T_72, 2) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_75 = and(_rTxData_T_73, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rTxData_T_76 = or(_rTxData_T_71, _rTxData_T_75) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_80 = shr(_rTxData_T_76, 1) @[Bitwise.scala 103:21]
    node _GEN_204 = pad(_rTxData_T_80, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_81 = and(_GEN_204, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rTxData_T_82 = bits(_rTxData_T_76, 6, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_83 = shl(_rTxData_T_82, 1) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_85 = and(_rTxData_T_83, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rTxData_2 = or(_rTxData_T_81, _rTxData_T_85) @[Bitwise.scala 103:39]
    skip
    skip
    node _rTxData_T_89 = shr(io_txData_3, 4) @[Bitwise.scala 103:21]
    node _GEN_205 = pad(_rTxData_T_89, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_90 = and(_GEN_205, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rTxData_T_91 = bits(io_txData_3, 3, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_92 = shl(_rTxData_T_91, 4) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_94 = and(_rTxData_T_92, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rTxData_T_95 = or(_rTxData_T_90, _rTxData_T_94) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_99 = shr(_rTxData_T_95, 2) @[Bitwise.scala 103:21]
    node _GEN_206 = pad(_rTxData_T_99, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_100 = and(_GEN_206, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rTxData_T_101 = bits(_rTxData_T_95, 5, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_102 = shl(_rTxData_T_101, 2) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_104 = and(_rTxData_T_102, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rTxData_T_105 = or(_rTxData_T_100, _rTxData_T_104) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_109 = shr(_rTxData_T_105, 1) @[Bitwise.scala 103:21]
    node _GEN_207 = pad(_rTxData_T_109, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_110 = and(_GEN_207, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rTxData_T_111 = bits(_rTxData_T_105, 6, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_112 = shl(_rTxData_T_111, 1) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_114 = and(_rTxData_T_112, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rTxData_3 = or(_rTxData_T_110, _rTxData_T_114) @[Bitwise.scala 103:39]
    skip
    skip
    node _rTxData_T_118 = shr(io_txData_4, 4) @[Bitwise.scala 103:21]
    node _GEN_208 = pad(_rTxData_T_118, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_119 = and(_GEN_208, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rTxData_T_120 = bits(io_txData_4, 3, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_121 = shl(_rTxData_T_120, 4) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_123 = and(_rTxData_T_121, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rTxData_T_124 = or(_rTxData_T_119, _rTxData_T_123) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_128 = shr(_rTxData_T_124, 2) @[Bitwise.scala 103:21]
    node _GEN_209 = pad(_rTxData_T_128, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_129 = and(_GEN_209, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rTxData_T_130 = bits(_rTxData_T_124, 5, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_131 = shl(_rTxData_T_130, 2) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_133 = and(_rTxData_T_131, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rTxData_T_134 = or(_rTxData_T_129, _rTxData_T_133) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_138 = shr(_rTxData_T_134, 1) @[Bitwise.scala 103:21]
    node _GEN_210 = pad(_rTxData_T_138, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_139 = and(_GEN_210, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rTxData_T_140 = bits(_rTxData_T_134, 6, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_141 = shl(_rTxData_T_140, 1) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_143 = and(_rTxData_T_141, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rTxData_4 = or(_rTxData_T_139, _rTxData_T_143) @[Bitwise.scala 103:39]
    skip
    skip
    node _rTxData_T_147 = shr(io_txData_5, 4) @[Bitwise.scala 103:21]
    node _GEN_211 = pad(_rTxData_T_147, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_148 = and(_GEN_211, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rTxData_T_149 = bits(io_txData_5, 3, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_150 = shl(_rTxData_T_149, 4) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_152 = and(_rTxData_T_150, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rTxData_T_153 = or(_rTxData_T_148, _rTxData_T_152) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_157 = shr(_rTxData_T_153, 2) @[Bitwise.scala 103:21]
    node _GEN_212 = pad(_rTxData_T_157, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_158 = and(_GEN_212, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rTxData_T_159 = bits(_rTxData_T_153, 5, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_160 = shl(_rTxData_T_159, 2) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_162 = and(_rTxData_T_160, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rTxData_T_163 = or(_rTxData_T_158, _rTxData_T_162) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_167 = shr(_rTxData_T_163, 1) @[Bitwise.scala 103:21]
    node _GEN_213 = pad(_rTxData_T_167, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_168 = and(_GEN_213, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rTxData_T_169 = bits(_rTxData_T_163, 6, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_170 = shl(_rTxData_T_169, 1) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_172 = and(_rTxData_T_170, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rTxData_5 = or(_rTxData_T_168, _rTxData_T_172) @[Bitwise.scala 103:39]
    skip
    skip
    node _rTxData_T_176 = shr(io_txData_6, 4) @[Bitwise.scala 103:21]
    node _GEN_214 = pad(_rTxData_T_176, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_177 = and(_GEN_214, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rTxData_T_178 = bits(io_txData_6, 3, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_179 = shl(_rTxData_T_178, 4) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_181 = and(_rTxData_T_179, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rTxData_T_182 = or(_rTxData_T_177, _rTxData_T_181) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_186 = shr(_rTxData_T_182, 2) @[Bitwise.scala 103:21]
    node _GEN_215 = pad(_rTxData_T_186, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_187 = and(_GEN_215, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rTxData_T_188 = bits(_rTxData_T_182, 5, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_189 = shl(_rTxData_T_188, 2) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_191 = and(_rTxData_T_189, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rTxData_T_192 = or(_rTxData_T_187, _rTxData_T_191) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_196 = shr(_rTxData_T_192, 1) @[Bitwise.scala 103:21]
    node _GEN_216 = pad(_rTxData_T_196, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_197 = and(_GEN_216, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rTxData_T_198 = bits(_rTxData_T_192, 6, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_199 = shl(_rTxData_T_198, 1) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_201 = and(_rTxData_T_199, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rTxData_6 = or(_rTxData_T_197, _rTxData_T_201) @[Bitwise.scala 103:39]
    skip
    skip
    node _rTxData_T_205 = shr(io_txData_7, 4) @[Bitwise.scala 103:21]
    node _GEN_217 = pad(_rTxData_T_205, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_206 = and(_GEN_217, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rTxData_T_207 = bits(io_txData_7, 3, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_208 = shl(_rTxData_T_207, 4) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_210 = and(_rTxData_T_208, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rTxData_T_211 = or(_rTxData_T_206, _rTxData_T_210) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_215 = shr(_rTxData_T_211, 2) @[Bitwise.scala 103:21]
    node _GEN_218 = pad(_rTxData_T_215, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_216 = and(_GEN_218, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rTxData_T_217 = bits(_rTxData_T_211, 5, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_218 = shl(_rTxData_T_217, 2) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_220 = and(_rTxData_T_218, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rTxData_T_221 = or(_rTxData_T_216, _rTxData_T_220) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_225 = shr(_rTxData_T_221, 1) @[Bitwise.scala 103:21]
    node _GEN_219 = pad(_rTxData_T_225, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_226 = and(_GEN_219, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rTxData_T_227 = bits(_rTxData_T_221, 6, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_228 = shl(_rTxData_T_227, 1) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_230 = and(_rTxData_T_228, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rTxData_7 = or(_rTxData_T_226, _rTxData_T_230) @[Bitwise.scala 103:39]
    skip
    skip
    node _rTxData_T_234 = shr(io_txData_8, 4) @[Bitwise.scala 103:21]
    node _GEN_220 = pad(_rTxData_T_234, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_235 = and(_GEN_220, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rTxData_T_236 = bits(io_txData_8, 3, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_237 = shl(_rTxData_T_236, 4) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_239 = and(_rTxData_T_237, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rTxData_T_240 = or(_rTxData_T_235, _rTxData_T_239) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_244 = shr(_rTxData_T_240, 2) @[Bitwise.scala 103:21]
    node _GEN_221 = pad(_rTxData_T_244, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_245 = and(_GEN_221, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rTxData_T_246 = bits(_rTxData_T_240, 5, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_247 = shl(_rTxData_T_246, 2) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_249 = and(_rTxData_T_247, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rTxData_T_250 = or(_rTxData_T_245, _rTxData_T_249) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_254 = shr(_rTxData_T_250, 1) @[Bitwise.scala 103:21]
    node _GEN_222 = pad(_rTxData_T_254, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_255 = and(_GEN_222, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rTxData_T_256 = bits(_rTxData_T_250, 6, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_257 = shl(_rTxData_T_256, 1) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_259 = and(_rTxData_T_257, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rTxData_8 = or(_rTxData_T_255, _rTxData_T_259) @[Bitwise.scala 103:39]
    skip
    skip
    node _rTxData_T_263 = shr(io_txData_9, 4) @[Bitwise.scala 103:21]
    node _GEN_223 = pad(_rTxData_T_263, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_264 = and(_GEN_223, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rTxData_T_265 = bits(io_txData_9, 3, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_266 = shl(_rTxData_T_265, 4) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_268 = and(_rTxData_T_266, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rTxData_T_269 = or(_rTxData_T_264, _rTxData_T_268) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_273 = shr(_rTxData_T_269, 2) @[Bitwise.scala 103:21]
    node _GEN_224 = pad(_rTxData_T_273, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_274 = and(_GEN_224, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rTxData_T_275 = bits(_rTxData_T_269, 5, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_276 = shl(_rTxData_T_275, 2) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_278 = and(_rTxData_T_276, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rTxData_T_279 = or(_rTxData_T_274, _rTxData_T_278) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_283 = shr(_rTxData_T_279, 1) @[Bitwise.scala 103:21]
    node _GEN_225 = pad(_rTxData_T_283, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_284 = and(_GEN_225, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rTxData_T_285 = bits(_rTxData_T_279, 6, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_286 = shl(_rTxData_T_285, 1) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_288 = and(_rTxData_T_286, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rTxData_9 = or(_rTxData_T_284, _rTxData_T_288) @[Bitwise.scala 103:39]
    skip
    skip
    node _rTxData_T_292 = shr(io_txData_10, 4) @[Bitwise.scala 103:21]
    node _GEN_226 = pad(_rTxData_T_292, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_293 = and(_GEN_226, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rTxData_T_294 = bits(io_txData_10, 3, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_295 = shl(_rTxData_T_294, 4) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_297 = and(_rTxData_T_295, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rTxData_T_298 = or(_rTxData_T_293, _rTxData_T_297) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_302 = shr(_rTxData_T_298, 2) @[Bitwise.scala 103:21]
    node _GEN_227 = pad(_rTxData_T_302, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_303 = and(_GEN_227, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rTxData_T_304 = bits(_rTxData_T_298, 5, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_305 = shl(_rTxData_T_304, 2) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_307 = and(_rTxData_T_305, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rTxData_T_308 = or(_rTxData_T_303, _rTxData_T_307) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_312 = shr(_rTxData_T_308, 1) @[Bitwise.scala 103:21]
    node _GEN_228 = pad(_rTxData_T_312, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_313 = and(_GEN_228, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rTxData_T_314 = bits(_rTxData_T_308, 6, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_315 = shl(_rTxData_T_314, 1) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_317 = and(_rTxData_T_315, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rTxData_10 = or(_rTxData_T_313, _rTxData_T_317) @[Bitwise.scala 103:39]
    skip
    skip
    node _rTxData_T_321 = shr(io_txData_11, 4) @[Bitwise.scala 103:21]
    node _GEN_229 = pad(_rTxData_T_321, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_322 = and(_GEN_229, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rTxData_T_323 = bits(io_txData_11, 3, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_324 = shl(_rTxData_T_323, 4) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_326 = and(_rTxData_T_324, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rTxData_T_327 = or(_rTxData_T_322, _rTxData_T_326) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_331 = shr(_rTxData_T_327, 2) @[Bitwise.scala 103:21]
    node _GEN_230 = pad(_rTxData_T_331, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_332 = and(_GEN_230, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rTxData_T_333 = bits(_rTxData_T_327, 5, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_334 = shl(_rTxData_T_333, 2) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_336 = and(_rTxData_T_334, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rTxData_T_337 = or(_rTxData_T_332, _rTxData_T_336) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_341 = shr(_rTxData_T_337, 1) @[Bitwise.scala 103:21]
    node _GEN_231 = pad(_rTxData_T_341, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_342 = and(_GEN_231, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rTxData_T_343 = bits(_rTxData_T_337, 6, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_344 = shl(_rTxData_T_343, 1) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_346 = and(_rTxData_T_344, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rTxData_11 = or(_rTxData_T_342, _rTxData_T_346) @[Bitwise.scala 103:39]
    skip
    skip
    node _rTxData_T_350 = shr(io_txData_12, 4) @[Bitwise.scala 103:21]
    node _GEN_232 = pad(_rTxData_T_350, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_351 = and(_GEN_232, UInt<8>("hf")) @[Bitwise.scala 103:31]
    node _rTxData_T_352 = bits(io_txData_12, 3, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_353 = shl(_rTxData_T_352, 4) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_355 = and(_rTxData_T_353, UInt<8>("hf0")) @[Bitwise.scala 103:75]
    node _rTxData_T_356 = or(_rTxData_T_351, _rTxData_T_355) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_360 = shr(_rTxData_T_356, 2) @[Bitwise.scala 103:21]
    node _GEN_233 = pad(_rTxData_T_360, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_361 = and(_GEN_233, UInt<8>("h33")) @[Bitwise.scala 103:31]
    node _rTxData_T_362 = bits(_rTxData_T_356, 5, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_363 = shl(_rTxData_T_362, 2) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_365 = and(_rTxData_T_363, UInt<8>("hcc")) @[Bitwise.scala 103:75]
    node _rTxData_T_366 = or(_rTxData_T_361, _rTxData_T_365) @[Bitwise.scala 103:39]
    skip
    skip
    skip
    node _rTxData_T_370 = shr(_rTxData_T_366, 1) @[Bitwise.scala 103:21]
    node _GEN_234 = pad(_rTxData_T_370, 8) @[Bitwise.scala 103:31]
    node _rTxData_T_371 = and(_GEN_234, UInt<8>("h55")) @[Bitwise.scala 103:31]
    node _rTxData_T_372 = bits(_rTxData_T_366, 6, 0) @[Bitwise.scala 103:46]
    node _rTxData_T_373 = shl(_rTxData_T_372, 1) @[Bitwise.scala 103:65]
    skip
    node _rTxData_T_375 = and(_rTxData_T_373, UInt<8>("haa")) @[Bitwise.scala 103:75]
    node rTxData_12 = or(_rTxData_T_371, _rTxData_T_375) @[Bitwise.scala 103:39]
    skip
    node basicChain_hi_hi_hi = bits(rTxData_1, 7, 4) @[CanBsp.scala 254:41]
    node basicChain_hi_lo = bits(rTxData_1, 3, 0) @[CanBsp.scala 254:68]
    skip
    node basicChain_lo = cat(rTxData_0, UInt<1>("h0")) @[Cat.scala 30:58]
    node basicChain_hi_hi = cat(basicChain_hi_hi_hi, UInt<2>("h0")) @[Cat.scala 30:58]
    node basicChain_hi = cat(basicChain_hi_hi, basicChain_hi_lo) @[Cat.scala 30:58]
    node basicChain = cat(basicChain_hi, basicChain_lo) @[Cat.scala 30:58]
    skip
    skip
    node basicChainData_lo_lo = cat(rTxData_3, rTxData_2) @[Cat.scala 30:58]
    skip
    skip
    node basicChainData_lo_hi = cat(rTxData_5, rTxData_4) @[Cat.scala 30:58]
    node basicChainData_lo = cat(basicChainData_lo_hi, basicChainData_lo_lo) @[Cat.scala 30:58]
    skip
    skip
    node basicChainData_hi_lo = cat(rTxData_7, rTxData_6) @[Cat.scala 30:58]
    skip
    skip
    node basicChainData_hi_hi = cat(rTxData_9, rTxData_8) @[Cat.scala 30:58]
    node basicChainData_hi = cat(basicChainData_hi_hi, basicChainData_hi_lo) @[Cat.scala 30:58]
    node basicChainData = cat(basicChainData_hi, basicChainData_lo) @[Cat.scala 30:58]
    node extendedChainStd_hi_hi_hi = bits(rTxData_0, 7, 4) @[CanBsp.scala 256:47]
    node extendedChainStd_hi_lo = bits(rTxData_0, 1, 1) @[CanBsp.scala 256:74]
    node extendedChainStd_lo_hi_hi = bits(rTxData_2, 2, 0) @[CanBsp.scala 256:89]
    skip
    node extendedChainStd_lo_hi = cat(extendedChainStd_lo_hi_hi, rTxData_1) @[Cat.scala 30:58]
    node extendedChainStd_lo = cat(extendedChainStd_lo_hi, UInt<1>("h0")) @[Cat.scala 30:58]
    node extendedChainStd_hi_hi = cat(extendedChainStd_hi_hi_hi, UInt<2>("h0")) @[Cat.scala 30:58]
    node extendedChainStd_hi = cat(extendedChainStd_hi_hi, extendedChainStd_hi_lo) @[Cat.scala 30:58]
    node extendedChainStd = cat(extendedChainStd_hi, extendedChainStd_lo) @[Cat.scala 30:58]
    skip
    skip
    node extendedChainExt_hi_lo_hi = bits(rTxData_4, 4, 0) @[CanBsp.scala 257:89]
    skip
    node extendedChainExt_lo_hi_hi_hi = bits(rTxData_2, 7, 3) @[CanBsp.scala 257:123]
    skip
    skip
    node extendedChainExt_lo_lo = cat(rTxData_1, UInt<1>("h0")) @[Cat.scala 30:58]
    node extendedChainExt_lo_hi_hi = cat(extendedChainExt_lo_hi_hi_hi, UInt<2>("h3")) @[Cat.scala 30:58]
    node extendedChainExt_lo_hi = cat(extendedChainExt_lo_hi_hi, extendedChainStd_lo_hi_hi) @[Cat.scala 30:58]
    node extendedChainExt_lo = cat(extendedChainExt_lo_hi, extendedChainExt_lo_lo) @[Cat.scala 30:58]
    node extendedChainExt_hi_lo = cat(extendedChainExt_hi_lo_hi, rTxData_3) @[Cat.scala 30:58]
    skip
    skip
    node extendedChainExt_hi = cat(extendedChainStd_hi, extendedChainExt_hi_lo) @[Cat.scala 30:58]
    node extendedChainExt = cat(extendedChainExt_hi, extendedChainExt_lo) @[Cat.scala 30:58]
    node extendedChainDataStd_lo_lo = cat(rTxData_4, rTxData_3) @[Cat.scala 30:58]
    node extendedChainDataStd_lo_hi = cat(rTxData_6, rTxData_5) @[Cat.scala 30:58]
    node extendedChainDataStd_lo = cat(extendedChainDataStd_lo_hi, extendedChainDataStd_lo_lo) @[Cat.scala 30:58]
    node extendedChainDataStd_hi_lo = cat(rTxData_8, rTxData_7) @[Cat.scala 30:58]
    skip
    node extendedChainDataStd_hi_hi = cat(rTxData_10, rTxData_9) @[Cat.scala 30:58]
    node extendedChainDataStd_hi = cat(extendedChainDataStd_hi_hi, extendedChainDataStd_hi_lo) @[Cat.scala 30:58]
    node extendedChainDataStd = cat(extendedChainDataStd_hi, extendedChainDataStd_lo) @[Cat.scala 30:58]
    skip
    skip
    node extendedChainDataExt_lo = cat(extendedChainDataStd_hi_lo, extendedChainDataStd_lo_hi) @[Cat.scala 30:58]
    skip
    skip
    skip
    node extendedChainDataExt_hi_hi = cat(rTxData_12, rTxData_11) @[Cat.scala 30:58]
    node extendedChainDataExt_hi = cat(extendedChainDataExt_hi_hi, extendedChainDataStd_hi_hi) @[Cat.scala 30:58]
    node extendedChainDataExt = cat(extendedChainDataExt_hi, extendedChainDataExt_lo) @[Cat.scala 30:58]
    skip
    node _io_sendAck_T_1 = and(_goEarlyTx_T_2, rxAck) @[CanBsp.scala 261:31]
    node _io_sendAck_T_2 = not(err) @[CanBsp.scala 261:42]
    node _io_sendAck_T_3 = and(_io_sendAck_T_1, _io_sendAck_T_2) @[CanBsp.scala 261:39]
    skip
    skip
    node _errorFlagOver_T = not(io_nodeErrorPassive) @[CanBsp.scala 268:32]
    node _errorFlagOver_T_1 = and(_errorFlagOver_T, io_samplePoint) @[CanBsp.scala 268:54]
    skip
    node _errorFlagOver_T_3 = and(_errorFlagOver_T_1, _bitErrExc4_T) @[CanBsp.scala 268:71]
    node _errorFlagOver_T_4 = and(io_nodeErrorPassive, io_samplePoint) @[CanBsp.scala 268:114]
    node _errorFlagOver_T_5 = eq(passiveCnt, UInt<3>("h6")) @[CanBsp.scala 268:145]
    node _errorFlagOver_T_6 = and(_errorFlagOver_T_4, _errorFlagOver_T_5) @[CanBsp.scala 268:131]
    node _errorFlagOver_T_7 = or(_errorFlagOver_T_3, _errorFlagOver_T_6) @[CanBsp.scala 268:92]
    skip
    node errorFlagOver = and(_errorFlagOver_T_7, _bitErrExc4_T_2) @[CanBsp.scala 268:155]
    skip
    node _overloadFlagOver_T_1 = and(io_samplePoint, _bitErrExc4_T_4) @[CanBsp.scala 269:48]
    skip
    node overloadFlagOver = and(_overloadFlagOver_T_1, _bitErrExc4_T_6) @[CanBsp.scala 269:73]
    skip
    node _rule5_T_1 = and(_errorFlagOver_T, errorFrame) @[CanBsp.scala 271:49]
    skip
    node _rule5_T_3 = and(_rule5_T_1, _bitErrExc3_T_1) @[CanBsp.scala 271:62]
    node _rule5_T_4 = lt(overloadCnt1, UInt<3>("h7")) @[CanBsp.scala 271:117]
    node _rule5_T_5 = and(io_overloadFrame, _rule5_T_4) @[CanBsp.scala 271:101]
    node _rule5_T_6 = or(_rule5_T_3, _rule5_T_5) @[CanBsp.scala 271:82]
    node rule5 = and(bitErr, _rule5_T_6) @[CanBsp.scala 271:19]
    node _limitedTxCntExt_T = bits(io_txData_0, 3, 3) @[CanBsp.scala 272:48]
    node _limitedTxCntExt_T_1 = bits(io_txData_0, 2, 0) @[CanBsp.scala 272:74]
    node _limitedTxCntExt_T_2 = shl(_limitedTxCntExt_T_1, 3) @[CanBsp.scala 272:80]
    node _limitedTxCntExt_T_3 = sub(_limitedTxCntExt_T_2, UInt<6>("h1")) @[CanBsp.scala 272:95]
    node _limitedTxCntExt_T_4 = tail(_limitedTxCntExt_T_3, 1) @[CanBsp.scala 272:95]
    node limitedTxCntExt = mux(_limitedTxCntExt_T, UInt<6>("h3f"), _limitedTxCntExt_T_4) @[CanBsp.scala 272:35]
    node _limitedTxCntStd_T = bits(io_txData_1, 3, 3) @[CanBsp.scala 273:48]
    node _limitedTxCntStd_T_1 = bits(io_txData_1, 2, 0) @[CanBsp.scala 273:74]
    node _limitedTxCntStd_T_2 = shl(_limitedTxCntStd_T_1, 3) @[CanBsp.scala 273:80]
    node _limitedTxCntStd_T_3 = sub(_limitedTxCntStd_T_2, UInt<6>("h1")) @[CanBsp.scala 273:95]
    node _limitedTxCntStd_T_4 = tail(_limitedTxCntStd_T_3, 1) @[CanBsp.scala 273:95]
    node limitedTxCntStd = mux(_limitedTxCntStd_T, UInt<6>("h3f"), _limitedTxCntStd_T_4) @[CanBsp.scala 273:35]
    node _rstTxPointer_T = not(bitDeStuffTx) @[CanBsp.scala 275:31]
    node _rstTxPointer_T_1 = and(_rstTxPointer_T, io_txPoint) @[CanBsp.scala 275:46]
    node _rstTxPointer_T_2 = not(rxData) @[CanBsp.scala 275:62]
    node _rstTxPointer_T_3 = and(_rstTxPointer_T_1, _rstTxPointer_T_2) @[CanBsp.scala 275:59]
    node _rstTxPointer_T_4 = and(_rstTxPointer_T_3, io_extendedMode) @[CanBsp.scala 275:71]
    node _rstTxPointer_T_5 = bits(rTxData_0, 0, 0) @[CanBsp.scala 275:104]
    node _rstTxPointer_T_6 = and(_rstTxPointer_T_4, _rstTxPointer_T_5) @[CanBsp.scala 275:91]
    node _rstTxPointer_T_7 = eq(txPointer, UInt<6>("h26")) @[CanBsp.scala 275:120]
    node _rstTxPointer_T_8 = and(_rstTxPointer_T_6, _rstTxPointer_T_7) @[CanBsp.scala 275:108]
    skip
    skip
    skip
    skip
    skip
    skip
    node _rstTxPointer_T_15 = not(_rstTxPointer_T_5) @[CanBsp.scala 276:93]
    node _rstTxPointer_T_16 = and(_rstTxPointer_T_4, _rstTxPointer_T_15) @[CanBsp.scala 276:91]
    node _rstTxPointer_T_17 = eq(txPointer, UInt<6>("h12")) @[CanBsp.scala 276:120]
    node _rstTxPointer_T_18 = and(_rstTxPointer_T_16, _rstTxPointer_T_17) @[CanBsp.scala 276:108]
    node _rstTxPointer_T_19 = or(_rstTxPointer_T_8, _rstTxPointer_T_18) @[CanBsp.scala 275:130]
    skip
    skip
    skip
    skip
    node _rstTxPointer_T_24 = not(io_extendedMode) @[CanBsp.scala 277:74]
    node _rstTxPointer_T_25 = and(_rstTxPointer_T_3, _rstTxPointer_T_24) @[CanBsp.scala 277:71]
    skip
    node _rstTxPointer_T_27 = and(_rstTxPointer_T_25, _rstTxPointer_T_17) @[CanBsp.scala 277:106]
    node _rstTxPointer_T_28 = or(_rstTxPointer_T_19, _rstTxPointer_T_27) @[CanBsp.scala 276:130]
    skip
    skip
    node _rstTxPointer_T_31 = and(_rstTxPointer_T_1, rxData) @[CanBsp.scala 278:59]
    node _rstTxPointer_T_32 = and(_rstTxPointer_T_31, io_extendedMode) @[CanBsp.scala 278:71]
    node _rstTxPointer_T_33 = eq(txPointer, limitedTxCntExt) @[CanBsp.scala 278:118]
    node _rstTxPointer_T_34 = and(_rstTxPointer_T_32, _rstTxPointer_T_33) @[CanBsp.scala 278:106]
    node _rstTxPointer_T_35 = or(_rstTxPointer_T_28, _rstTxPointer_T_34) @[CanBsp.scala 277:128]
    skip
    skip
    skip
    skip
    node _rstTxPointer_T_40 = and(_rstTxPointer_T_31, _rstTxPointer_T_24) @[CanBsp.scala 279:71]
    node _rstTxPointer_T_41 = eq(txPointer, limitedTxCntStd) @[CanBsp.scala 279:118]
    node _rstTxPointer_T_42 = and(_rstTxPointer_T_40, _rstTxPointer_T_41) @[CanBsp.scala 279:106]
    node _rstTxPointer_T_43 = or(_rstTxPointer_T_35, _rstTxPointer_T_42) @[CanBsp.scala 278:139]
    node _rstTxPointer_T_44 = and(io_txPoint, rxCrcLim) @[CanBsp.scala 280:59]
    node _rstTxPointer_T_45 = or(_rstTxPointer_T_43, _rstTxPointer_T_44) @[CanBsp.scala 279:139]
    node _rstTxPointer_T_46 = or(goRxIdle, errorFrame) @[CanBsp.scala 281:39]
    node _rstTxPointer_T_47 = or(_rstTxPointer_T_46, io_resetMode) @[CanBsp.scala 281:52]
    node _rstTxPointer_T_48 = or(_rstTxPointer_T_47, io_overloadFrame) @[CanBsp.scala 281:67]
    node rstTxPointer = or(_rstTxPointer_T_45, _rstTxPointer_T_48) @[CanBsp.scala 280:139]
    skip
    skip
    skip
    skip
    node _io_goOverloadFrame_T_4 = and(rxEof, _formErr_T_11) @[CanBsp.scala 283:92]
    skip
    node _io_goOverloadFrame_T_6 = and(_io_goOverloadFrame_T_4, _io_goRxInter_T_1) @[CanBsp.scala 283:112]
    node _io_goOverloadFrame_T_7 = or(_io_goOverloadFrame_T_6, errorFrameEnded) @[CanBsp.scala 283:131]
    node _io_goOverloadFrame_T_8 = or(_io_goOverloadFrame_T_7, overloadFrameEnded) @[CanBsp.scala 283:149]
    node _io_goOverloadFrame_T_9 = and(_goRxId1_T_1, _io_goOverloadFrame_T_8) @[CanBsp.scala 283:83]
    skip
    skip
    node _io_goOverloadFrame_T_12 = and(_goRxId1_T_1, io_rxInter) @[CanBsp.scala 284:60]
    skip
    node _io_goOverloadFrame_T_14 = lt(_lastBitOfInter_T, UInt<2>("h2")) @[CanBsp.scala 284:88]
    node _io_goOverloadFrame_T_15 = and(_io_goOverloadFrame_T_12, _io_goOverloadFrame_T_14) @[CanBsp.scala 284:73]
    node _io_goOverloadFrame_T_16 = or(_io_goOverloadFrame_T_9, _io_goOverloadFrame_T_15) @[CanBsp.scala 283:171]
    skip
    skip
    skip
    skip
    node _io_goOverloadFrame_T_21 = or(_errorFrameEnded_T, _overloadFrameEnded_T) @[CanBsp.scala 285:82]
    node _io_goOverloadFrame_T_22 = and(_goRxId1_T_1, _io_goOverloadFrame_T_21) @[CanBsp.scala 285:60]
    skip
    skip
    skip
    skip
    node _io_notFirstBitOfInter_T_1 = orr(_lastBitOfInter_T) @[CanBsp.scala 286:57]
    skip
    skip
    skip
    skip
    node _io_goErrorFrame_T_3 = and(crcErr, goRxEof) @[CanBsp.scala 287:70]
    skip
    node _io_setResetMode_T = not(nodeBusOffQ) @[CanBsp.scala 288:35]
    skip
    node _io_txSuccessful_T = and(transmitter, io_goRxInter) @[CanBsp.scala 289:34]
    skip
    node _io_txSuccessful_T_2 = and(_io_txSuccessful_T, _bitDeStuffSet_T) @[CanBsp.scala 289:49]
    node _io_txSuccessful_T_3 = not(errorFrameEnded) @[CanBsp.scala 289:73]
    node _io_txSuccessful_T_4 = and(_io_txSuccessful_T_2, _io_txSuccessful_T_3) @[CanBsp.scala 289:70]
    node _io_txSuccessful_T_5 = not(overloadFrameEnded) @[CanBsp.scala 289:94]
    node _io_txSuccessful_T_6 = and(_io_txSuccessful_T_4, _io_txSuccessful_T_5) @[CanBsp.scala 289:91]
    node _io_txSuccessful_T_7 = not(arbitrationLost) @[CanBsp.scala 289:118]
    skip
    node _T = or(goRxId1, io_goErrorFrame) @[CanBsp.scala 291:16]
    node _GEN_0 = or(goRxIdle, rxIdle) @[CanBsp.scala 293:24 294:12 83:30]
    node _GEN_1 = mux(_T, UInt<1>("h0"), _GEN_0) @[CanBsp.scala 291:35 292:12]
    node _T_1 = or(goRxRtr1, io_goErrorFrame) @[CanBsp.scala 297:17]
    node _GEN_2 = or(goRxId1, rxId1) @[CanBsp.scala 299:23 300:11 112:29]
    node _GEN_3 = mux(_T_1, UInt<1>("h0"), _GEN_2) @[CanBsp.scala 297:36 298:11]
    node _T_2 = or(goRxIde, io_goErrorFrame) @[CanBsp.scala 303:16]
    node _GEN_4 = or(goRxRtr1, rxRtr1) @[CanBsp.scala 305:24 306:12 113:30]
    node _GEN_5 = mux(_T_2, UInt<1>("h0"), _GEN_4) @[CanBsp.scala 303:35 304:12]
    node _T_3 = or(goRxR0, goRxId2) @[CanBsp.scala 309:16]
    node _T_4 = or(_T_3, io_goErrorFrame) @[CanBsp.scala 309:26]
    node _GEN_6 = or(goRxIde, rxIde) @[CanBsp.scala 311:23 312:11 114:29]
    node _GEN_7 = mux(_T_4, UInt<1>("h0"), _GEN_6) @[CanBsp.scala 309:45 310:11]
    node _T_5 = or(goRxRtr2, io_goErrorFrame) @[CanBsp.scala 315:18]
    node _GEN_8 = or(goRxId2, rxId2) @[CanBsp.scala 317:23 318:11 115:29]
    node _GEN_9 = mux(_T_5, UInt<1>("h0"), _GEN_8) @[CanBsp.scala 315:37 316:11]
    node _T_6 = or(goRxR1, io_goErrorFrame) @[CanBsp.scala 321:15]
    node _GEN_10 = or(goRxRtr2, rxRtr2) @[CanBsp.scala 323:24 324:12 116:30]
    node _GEN_11 = mux(_T_6, UInt<1>("h0"), _GEN_10) @[CanBsp.scala 321:34 322:12]
    node _T_7 = or(goRxR0, io_goErrorFrame) @[CanBsp.scala 327:15]
    node _GEN_12 = or(goRxR1, rxR1) @[CanBsp.scala 329:22 330:10 118:27]
    node _GEN_13 = mux(_T_7, UInt<1>("h0"), _GEN_12) @[CanBsp.scala 327:34 328:10]
    node _T_8 = or(goRxDlc, io_goErrorFrame) @[CanBsp.scala 333:16]
    node _GEN_14 = or(goRxR0, rxR0) @[CanBsp.scala 335:22 336:10 117:28]
    node _GEN_15 = mux(_T_8, UInt<1>("h0"), _GEN_14) @[CanBsp.scala 333:35 334:10]
    skip
    skip
    node _T_9 = or(goRxData, goRxCrc) @[CanBsp.scala 339:17]
    node _T_10 = or(_T_9, io_goErrorFrame) @[CanBsp.scala 339:27]
    node _GEN_16 = or(goRxDlc, rxDlc) @[CanBsp.scala 341:23 342:11 119:29]
    node _GEN_17 = mux(_T_10, UInt<1>("h0"), _GEN_16) @[CanBsp.scala 339:46 340:11]
    node _T_11 = or(goRxCrc, io_goErrorFrame) @[CanBsp.scala 345:16]
    node _GEN_18 = or(goRxData, rxData) @[CanBsp.scala 347:24 348:12 120:30]
    node _GEN_19 = mux(_T_11, UInt<1>("h0"), _GEN_18) @[CanBsp.scala 345:35 346:12]
    node _T_12 = or(goRxCrcLim, io_goErrorFrame) @[CanBsp.scala 351:19]
    node _GEN_20 = or(goRxCrc, rxCrc) @[CanBsp.scala 353:23 354:11 121:29]
    node _GEN_21 = mux(_T_12, UInt<1>("h0"), _GEN_20) @[CanBsp.scala 351:38 352:11]
    skip
    node _GEN_22 = or(goRxCrcLim, rxCrcLim) @[CanBsp.scala 359:26 360:14 122:32]
    node _GEN_23 = mux(_bitDeStuffReset_T, UInt<1>("h0"), _GEN_22) @[CanBsp.scala 357:35 358:14]
    node _T_14 = or(goRxAckLim, io_goErrorFrame) @[CanBsp.scala 363:19]
    node _GEN_24 = or(goRxAck, rxAck) @[CanBsp.scala 365:23 366:11 123:29]
    node _GEN_25 = mux(_T_14, UInt<1>("h0"), _GEN_24) @[CanBsp.scala 363:38 364:11]
    node _T_15 = or(goRxEof, io_goErrorFrame) @[CanBsp.scala 369:16]
    node _GEN_26 = or(goRxAckLim, rxAckLim) @[CanBsp.scala 371:26 372:14 124:32]
    node _GEN_27 = mux(_T_15, UInt<1>("h0"), _GEN_26) @[CanBsp.scala 369:35 370:14]
    node _T_16 = or(io_goRxInter, io_goErrorFrame) @[CanBsp.scala 375:21]
    node _T_17 = or(_T_16, io_goOverloadFrame) @[CanBsp.scala 375:39]
    node _GEN_28 = or(goRxEof, rxEof) @[CanBsp.scala 377:23 378:11 125:29]
    node _GEN_29 = mux(_T_17, UInt<1>("h0"), _GEN_28) @[CanBsp.scala 375:61 376:11]
    node _T_18 = or(goRxIdle, goRxId1) @[CanBsp.scala 381:17]
    node _T_19 = or(_T_18, io_goOverloadFrame) @[CanBsp.scala 381:27]
    node _T_20 = or(_T_19, io_goErrorFrame) @[CanBsp.scala 381:48]
    node _GEN_30 = or(io_goRxInter, rxInter) @[CanBsp.scala 383:28 384:13 89:31]
    node _GEN_31 = mux(_T_20, UInt<1>("h0"), _GEN_30) @[CanBsp.scala 381:67 382:13]
    node _T_21 = or(rxId1, rxId2) @[CanBsp.scala 387:32]
    node _T_22 = and(io_samplePoint, _T_21) @[CanBsp.scala 387:23]
    skip
    node _T_24 = and(_T_22, _goRxRtr1_T) @[CanBsp.scala 387:41]
    node id_hi = bits(id, 27, 0) @[CanBsp.scala 388:17]
    node _id_T = cat(id_hi, io_sampledBit) @[Cat.scala 30:58]
    node _GEN_32 = mux(_T_24, _id_T, id) @[CanBsp.scala 108:26 387:56 388:8]
    node _T_25 = and(io_samplePoint, rxRtr1) @[CanBsp.scala 391:23]
    skip
    node _T_27 = and(_T_25, _goRxRtr1_T) @[CanBsp.scala 391:32]
    node _GEN_33 = mux(_T_27, io_sampledBit, rtr1) @[CanBsp.scala 391:47 392:10 127:28]
    node _T_28 = and(io_samplePoint, rxRtr2) @[CanBsp.scala 395:23]
    skip
    node _T_30 = and(_T_28, _goRxRtr1_T) @[CanBsp.scala 395:32]
    node _GEN_34 = mux(_T_30, io_sampledBit, rtr2) @[CanBsp.scala 395:47 396:10 129:28]
    node _T_31 = and(io_samplePoint, rxIde) @[CanBsp.scala 399:23]
    skip
    node _T_33 = and(_T_31, _goRxRtr1_T) @[CanBsp.scala 399:31]
    node _GEN_35 = mux(_T_33, io_sampledBit, ide) @[CanBsp.scala 128:27 399:46 400:9]
    node _T_34 = and(io_samplePoint, rxDlc) @[CanBsp.scala 403:23]
    skip
    node _T_36 = and(_T_34, _goRxRtr1_T) @[CanBsp.scala 403:31]
    skip
    node _dataLen_T = cat(_goRxData_T_6, io_sampledBit) @[Cat.scala 30:58]
    node _GEN_36 = mux(_T_36, _dataLen_T, dataLen) @[CanBsp.scala 403:46 404:13 107:31]
    node _T_37 = and(io_samplePoint, rxData) @[CanBsp.scala 407:23]
    skip
    node _T_39 = and(_T_37, _goRxRtr1_T) @[CanBsp.scala 407:32]
    node tmpData_hi = bits(tmpData, 6, 0) @[CanBsp.scala 408:27]
    node _tmpData_T = cat(tmpData_hi, io_sampledBit) @[Cat.scala 30:58]
    node _GEN_37 = mux(_T_39, _tmpData_T, tmpData) @[CanBsp.scala 407:47 408:13 131:31]
    skip
    skip
    skip
    node _T_43 = bits(bitCnt, 2, 0) @[CanBsp.scala 411:54]
    node _T_44 = andr(_T_43) @[CanBsp.scala 411:64]
    node _T_45 = and(_T_39, _T_44) @[CanBsp.scala 411:46]
    skip
    node _byteCnt_T = add(byteCnt, UInt<3>("h1")) @[CanBsp.scala 418:24]
    node _byteCnt_T_1 = tail(_byteCnt_T, 1) @[CanBsp.scala 418:24]
    node _T_46 = and(io_samplePoint, goRxCrcLim) @[CanBsp.scala 419:29]
    node _GEN_39 = mux(_T_46, UInt<3>("h0"), byteCnt) @[CanBsp.scala 419:43 420:13 138:31]
    node _GEN_40 = mux(writeDataToTmpFifo, _byteCnt_T_1, _GEN_39) @[CanBsp.scala 417:28 418:13]
    skip
    skip
    skip
    skip
    skip
    node _T_47 = and(io_samplePoint, rxCrc) @[CanBsp.scala 427:23]
    skip
    node _T_49 = and(_T_47, _goRxRtr1_T) @[CanBsp.scala 427:31]
    node crcIn_hi = bits(crcIn, 13, 0) @[CanBsp.scala 428:23]
    node _crcIn_T = cat(crcIn_hi, io_sampledBit) @[Cat.scala 30:58]
    node _GEN_46 = mux(_T_49, _crcIn_T, crcIn) @[CanBsp.scala 427:46 428:11 130:29]
    node _T_50 = or(goRxId1, goRxId2) @[CanBsp.scala 431:16]
    node _T_51 = or(_T_50, goRxDlc) @[CanBsp.scala 431:26]
    node _T_52 = or(_T_51, goRxData) @[CanBsp.scala 431:36]
    node _T_53 = or(_T_52, goRxCrc) @[CanBsp.scala 431:47]
    node _T_54 = or(_T_53, goRxAck) @[CanBsp.scala 431:57]
    node _T_55 = or(_T_54, goRxEof) @[CanBsp.scala 431:67]
    node _T_56 = or(_T_55, io_goRxInter) @[CanBsp.scala 431:77]
    node _T_57 = or(_T_56, io_goErrorFrame) @[CanBsp.scala 431:92]
    node _T_58 = or(_T_57, io_goOverloadFrame) @[CanBsp.scala 431:110]
    skip
    node _T_60 = and(io_samplePoint, _goRxRtr1_T) @[CanBsp.scala 433:29]
    node _bitCnt_T = add(bitCnt, UInt<6>("h1")) @[CanBsp.scala 434:22]
    node _bitCnt_T_1 = tail(_bitCnt_T, 1) @[CanBsp.scala 434:22]
    node _GEN_47 = mux(_T_60, _bitCnt_T_1, bitCnt) @[CanBsp.scala 433:44 434:12 106:30]
    node _GEN_48 = mux(_T_58, UInt<6>("h0"), _GEN_47) @[CanBsp.scala 431:132 432:12]
    skip
    skip
    node _eofCnt_T = add(eofCnt, UInt<3>("h1")) @[CanBsp.scala 440:24]
    node _eofCnt_T_1 = tail(_eofCnt_T, 1) @[CanBsp.scala 440:24]
    node _GEN_49 = mux(rxEof, _eofCnt_T_1, eofCnt) @[CanBsp.scala 439:23 440:14 141:30]
    node _GEN_50 = mux(_T_17, UInt<3>("h0"), _GEN_49) @[CanBsp.scala 437:63 438:14]
    node _GEN_51 = mux(io_samplePoint, _GEN_50, eofCnt) @[CanBsp.scala 436:24 141:30]
    node _GEN_52 = mux(bitDeStuffReset, UInt<1>("h0"), bitStuffCntEn) @[CanBsp.scala 446:31 447:19 139:37]
    node _GEN_53 = or(bitDeStuffSet, _GEN_52) @[CanBsp.scala 444:23 445:19]
    skip
    skip
    skip
    node _bitStuffCnt_T = add(bitStuffCnt, UInt<3>("h1")) @[CanBsp.scala 456:34]
    node _bitStuffCnt_T_1 = tail(_bitStuffCnt_T, 1) @[CanBsp.scala 456:34]
    node _GEN_54 = mux(_stuffErr_T_2, _bitStuffCnt_T_1, UInt<3>("h1")) @[CanBsp.scala 455:50 456:19 458:19]
    node _GEN_55 = mux(bitDeStuff, UInt<3>("h1"), _GEN_54) @[CanBsp.scala 453:31 454:19]
    node _GEN_56 = mux(_stuffErr_T, _GEN_55, bitStuffCnt) @[CanBsp.scala 109:35 452:46]
    node _GEN_57 = mux(bitDeStuffReset, UInt<3>("h1"), _GEN_56) @[CanBsp.scala 450:25 451:17]
    node _T_66 = or(io_resetMode, bitDeStuffReset) @[CanBsp.scala 462:21]
    node _T_67 = and(txPointQ, bitStuffCntEn) @[CanBsp.scala 464:23]
    skip
    node _T_69 = eq(io_tx, txQ) @[CanBsp.scala 467:22]
    node _bitStuffCntTx_T = add(bitStuffCntTx, UInt<3>("h1")) @[CanBsp.scala 468:38]
    node _bitStuffCntTx_T_1 = tail(_bitStuffCntTx_T, 1) @[CanBsp.scala 468:38]
    node _GEN_58 = mux(_T_69, _bitStuffCntTx_T_1, UInt<3>("h1")) @[CanBsp.scala 467:31 468:21 470:21]
    node _GEN_59 = mux(bitDeStuffTx, UInt<3>("h1"), _GEN_58) @[CanBsp.scala 465:33 466:21]
    node _GEN_60 = mux(_T_67, _GEN_59, bitStuffCntTx) @[CanBsp.scala 110:37 464:40]
    node _GEN_61 = mux(_T_66, UInt<3>("h1"), _GEN_60) @[CanBsp.scala 462:40 463:19]
    node _GEN_62 = or(goCrcEnable, crcEnable) @[CanBsp.scala 476:27 477:15 140:33]
    node _GEN_63 = mux(goRxCrc, UInt<1>("h0"), _GEN_62) @[CanBsp.scala 474:22 475:15]
    node _T_70 = or(io_resetMode, errorFrameEnded) @[CanBsp.scala 480:21]
    node _crcErr_T = neq(crcIn, calculatedCrc) @[CanBsp.scala 483:21]
    node _GEN_64 = mux(goRxAck, _crcErr_T, crcErr) @[CanBsp.scala 482:23 483:12 153:30]
    node _GEN_65 = mux(_T_70, UInt<1>("h0"), _GEN_64) @[CanBsp.scala 480:40 481:12]
    skip
    node _T_72 = or(_T_70, io_goOverloadFrame) @[CanBsp.scala 486:39]
    node _GEN_66 = or(ackErr, ackErrLatched) @[CanBsp.scala 488:22 489:19 172:37]
    node _GEN_67 = mux(_T_72, UInt<1>("h0"), _GEN_66) @[CanBsp.scala 486:61 487:19]
    skip
    skip
    node _GEN_68 = or(bitErr, bitErrLatched) @[CanBsp.scala 494:22 495:19 173:37]
    node _GEN_69 = mux(_T_72, UInt<1>("h0"), _GEN_68) @[CanBsp.scala 492:61 493:19]
    node _T_75 = or(io_goErrorFrame, rule3Exc1_1) @[CanBsp.scala 498:24]
    skip
    node _T_77 = and(rule3Exc1_0, _bitErrExc3_T_1) @[CanBsp.scala 500:27]
    node _T_78 = and(_T_77, io_samplePoint) @[CanBsp.scala 500:47]
    skip
    node _T_80 = and(_T_78, _goRxId1_T) @[CanBsp.scala 500:64]
    node _GEN_70 = or(_T_80, rule3Exc1_1) @[CanBsp.scala 500:84 501:18 176:38]
    node _GEN_71 = mux(_T_75, UInt<1>("h0"), _GEN_70) @[CanBsp.scala 498:40 499:18]
    node _T_81 = or(errorFlagOver, rule3Exc1_1) @[CanBsp.scala 504:22]
    node _T_82 = and(io_transmitter, io_nodeErrorPassive) @[CanBsp.scala 506:29]
    node _T_83 = and(_T_82, ackErr) @[CanBsp.scala 506:51]
    node _GEN_72 = or(_T_83, rule3Exc1_0) @[CanBsp.scala 506:61 507:18 176:38]
    node _GEN_73 = mux(_T_81, UInt<1>("h0"), _GEN_72) @[CanBsp.scala 504:38 505:18]
    skip
    skip
    node _GEN_74 = or(stuffErr, stuffErrLatched) @[CanBsp.scala 512:24 513:21 174:39]
    node _GEN_75 = mux(_T_72, UInt<1>("h0"), _GEN_74) @[CanBsp.scala 510:61 511:21]
    skip
    skip
    node _GEN_76 = or(formErr, formErrLatched) @[CanBsp.scala 518:23 519:20 175:38]
    node _GEN_77 = mux(_T_72, UInt<1>("h0"), _GEN_76) @[CanBsp.scala 516:61 517:20]
    node _canCrcRx_io_enable_T = and(crcEnable, io_samplePoint) @[CanBsp.scala 524:35]
    skip
    skip
    node idOk = canAcf.io_idOk @[CanBsp.scala 237:25 544:8]
    node _T_88 = and(io_goRxInter, idOk) @[CanBsp.scala 548:27]
    skip
    node _T_90 = and(_T_88, _io_txSuccessful_T_3) @[CanBsp.scala 548:34]
    skip
    node _T_92 = or(_goEarlyTx_T_2, io_selfRxRequest) @[CanBsp.scala 548:68]
    node _T_93 = and(_T_90, _T_92) @[CanBsp.scala 548:53]
    node _GEN_78 = or(_T_93, wrFifo) @[CanBsp.scala 548:89 549:12 162:30]
    node _GEN_79 = mux(resetWrFifo, UInt<1>("h0"), _GEN_78) @[CanBsp.scala 546:21 547:12]
    node _T_94 = and(wrFifo, storingHeader) @[CanBsp.scala 554:21]
    node _headerCnt_T = add(headerCnt, UInt<3>("h1")) @[CanBsp.scala 555:28]
    node _GEN_80 = mux(_T_94, _headerCnt_T, pad(headerCnt, 4)) @[CanBsp.scala 554:38 555:15 161:33]
    node _GEN_81 = mux(resetWrFifo, UInt<4>("h0"), _GEN_80) @[CanBsp.scala 552:21 553:15]
    node _dataCnt_T = add(dataCnt, UInt<4>("h1")) @[CanBsp.scala 561:24]
    node _GEN_82 = mux(wrFifo, _dataCnt_T, pad(dataCnt, 5)) @[CanBsp.scala 560:22 561:13 160:31]
    node _GEN_83 = mux(resetWrFifo, UInt<5>("h0"), _GEN_82) @[CanBsp.scala 558:21 559:13]
    node fifoSelector_lo = cat(ide, headerCnt) @[Cat.scala 30:58]
    node fifoSelector_hi = cat(storingHeader, io_extendedMode) @[Cat.scala 30:58]
    node fifoSelector = cat(fifoSelector_hi, fifoSelector_lo) @[Cat.scala 30:58]
    node _dataForFifo_T = cat(UInt<1>("h0"), headerLen) @[Cat.scala 30:58]
    node _dataForFifo_T_1 = sub(dataCnt, _dataForFifo_T) @[CanBsp.scala 565:54]
    node _dataForFifo_T_2 = tail(_dataForFifo_T_1, 1) @[CanBsp.scala 565:54]
    skip
    node dataForFifo_lo = cat(UInt<2>("h0"), dataLen) @[Cat.scala 30:58]
    node dataForFifo_hi = cat(UInt<1>("h1"), rtr2) @[Cat.scala 30:58]
    node _dataForFifo_T_4 = cat(dataForFifo_hi, dataForFifo_lo) @[Cat.scala 30:58]
    node _dataForFifo_T_5 = bits(id, 28, 21) @[CanBsp.scala 567:28]
    node _dataForFifo_T_6 = bits(id, 20, 13) @[CanBsp.scala 568:28]
    node _dataForFifo_T_7 = bits(id, 12, 5) @[CanBsp.scala 569:28]
    node dataForFifo_hi_1 = bits(id, 4, 0) @[CanBsp.scala 570:32]
    node _dataForFifo_T_8 = cat(dataForFifo_hi_1, UInt<3>("h0")) @[Cat.scala 30:58]
    skip
    node dataForFifo_hi_2 = cat(UInt<1>("h0"), rtr1) @[Cat.scala 30:58]
    node _dataForFifo_T_9 = cat(dataForFifo_hi_2, dataForFifo_lo) @[Cat.scala 30:58]
    node _dataForFifo_T_10 = bits(id, 10, 3) @[CanBsp.scala 572:28]
    node dataForFifo_hi_hi = bits(id, 2, 0) @[CanBsp.scala 573:32]
    node dataForFifo_hi_3 = cat(dataForFifo_hi_hi, rtr1) @[Cat.scala 30:58]
    node _dataForFifo_T_11 = cat(dataForFifo_hi_3, UInt<4>("h0")) @[Cat.scala 30:58]
    skip
    skip
    skip
    node _dataForFifo_T_13 = cat(dataForFifo_hi_3, dataLen) @[Cat.scala 30:58]
    node _dataForFifo_T_14 = and(fifoSelector, UInt<6>("h37")) @[Lookup.scala 31:38]
    node _dataForFifo_T_15 = eq(UInt<6>("h21"), _dataForFifo_T_14) @[Lookup.scala 31:38]
    skip
    node _dataForFifo_T_17 = eq(UInt<6>("h20"), _dataForFifo_T_14) @[Lookup.scala 31:38]
    skip
    node _dataForFifo_T_19 = eq(UInt<6>("h32"), fifoSelector) @[Lookup.scala 31:38]
    skip
    node _dataForFifo_T_21 = eq(UInt<6>("h31"), fifoSelector) @[Lookup.scala 31:38]
    skip
    node _dataForFifo_T_23 = eq(UInt<6>("h30"), fifoSelector) @[Lookup.scala 31:38]
    skip
    node _dataForFifo_T_25 = eq(UInt<6>("h3c"), fifoSelector) @[Lookup.scala 31:38]
    skip
    node _dataForFifo_T_27 = eq(UInt<6>("h3b"), fifoSelector) @[Lookup.scala 31:38]
    skip
    node _dataForFifo_T_29 = eq(UInt<6>("h3a"), fifoSelector) @[Lookup.scala 31:38]
    skip
    node _dataForFifo_T_31 = eq(UInt<6>("h39"), fifoSelector) @[Lookup.scala 31:38]
    skip
    node _dataForFifo_T_33 = eq(UInt<6>("h38"), fifoSelector) @[Lookup.scala 31:38]
    node _dataForFifo_T_34 = mux(_dataForFifo_T_33, _dataForFifo_T_4, tmpFifo.dataForFifo_MPORT.data) @[Lookup.scala 33:37]
    node _dataForFifo_T_35 = mux(_dataForFifo_T_31, _dataForFifo_T_5, _dataForFifo_T_34) @[Lookup.scala 33:37]
    node _dataForFifo_T_36 = mux(_dataForFifo_T_29, _dataForFifo_T_6, _dataForFifo_T_35) @[Lookup.scala 33:37]
    node _dataForFifo_T_37 = mux(_dataForFifo_T_27, _dataForFifo_T_7, _dataForFifo_T_36) @[Lookup.scala 33:37]
    node _dataForFifo_T_38 = mux(_dataForFifo_T_25, _dataForFifo_T_8, _dataForFifo_T_37) @[Lookup.scala 33:37]
    node _dataForFifo_T_39 = mux(_dataForFifo_T_23, _dataForFifo_T_9, _dataForFifo_T_38) @[Lookup.scala 33:37]
    node _dataForFifo_T_40 = mux(_dataForFifo_T_21, _dataForFifo_T_10, _dataForFifo_T_39) @[Lookup.scala 33:37]
    node _dataForFifo_T_41 = mux(_dataForFifo_T_19, _dataForFifo_T_11, _dataForFifo_T_40) @[Lookup.scala 33:37]
    node _dataForFifo_T_42 = mux(_dataForFifo_T_17, _dataForFifo_T_10, _dataForFifo_T_41) @[Lookup.scala 33:37]
    skip
    skip
    node _T_95 = or(io_setResetMode, errorFrameEnded) @[CanBsp.scala 591:24]
    node _T_96 = or(_T_95, io_goOverloadFrame) @[CanBsp.scala 591:42]
    node _GEN_84 = or(io_goErrorFrame, errorFrame) @[CanBsp.scala 593:31 594:16 143:34]
    node _GEN_85 = mux(_T_96, UInt<1>("h0"), _GEN_84) @[CanBsp.scala 591:64 592:16]
    node _T_97 = or(errorFrameEnded, io_goErrorFrame) @[CanBsp.scala 597:24]
    node _T_98 = or(_T_97, io_goOverloadFrame) @[CanBsp.scala 597:42]
    node _T_99 = and(errorFrame, io_txPoint) @[CanBsp.scala 599:25]
    skip
    node _T_101 = and(_T_99, _bitErrExc3_T_1) @[CanBsp.scala 599:38]
    node _errorCnt1_T = add(errorCnt1, UInt<3>("h1")) @[CanBsp.scala 600:28]
    node _errorCnt1_T_1 = tail(_errorCnt1_T, 1) @[CanBsp.scala 600:28]
    node _GEN_86 = mux(_T_101, _errorCnt1_T_1, errorCnt1) @[CanBsp.scala 599:59 600:15 145:33]
    node _GEN_87 = mux(_T_98, UInt<3>("h0"), _GEN_86) @[CanBsp.scala 597:64 598:15]
    skip
    skip
    node _GEN_88 = or(errorFlagOver, errorFlagOverLatched) @[CanBsp.scala 605:29 606:26 180:44]
    node _GEN_89 = mux(_T_98, UInt<1>("h0"), _GEN_88) @[CanBsp.scala 603:64 604:26]
    skip
    skip
    node _T_106 = and(errorFlagOver, io_sampledBit) @[CanBsp.scala 611:42]
    node _T_107 = and(errorFrame, _T_106) @[CanBsp.scala 611:25]
    node _GEN_90 = or(_T_107, enableErrorCnt2) @[CanBsp.scala 611:60 612:21 144:39]
    node _GEN_91 = mux(_T_98, UInt<1>("h0"), _GEN_90) @[CanBsp.scala 609:64 610:21]
    skip
    skip
    node _T_110 = and(enableErrorCnt2, io_txPoint) @[CanBsp.scala 617:30]
    node _errorCnt2_T = add(errorCnt2, UInt<3>("h1")) @[CanBsp.scala 618:28]
    node _errorCnt2_T_1 = tail(_errorCnt2_T, 1) @[CanBsp.scala 618:28]
    node _GEN_92 = mux(_T_110, _errorCnt2_T_1, errorCnt2) @[CanBsp.scala 617:44 618:15 146:33]
    node _GEN_93 = mux(_T_98, UInt<3>("h0"), _GEN_92) @[CanBsp.scala 615:64 616:15]
    node _T_111 = or(enableErrorCnt2, io_goErrorFrame) @[CanBsp.scala 621:24]
    node _T_112 = or(_T_111, enableOverloadCnt2) @[CanBsp.scala 621:42]
    node _T_113 = or(_T_112, io_goOverloadFrame) @[CanBsp.scala 621:63]
    skip
    skip
    skip
    skip
    node _T_118 = or(_bitErrExc4_T, _bitErrExc4_T_4) @[CanBsp.scala 623:68]
    node _T_119 = and(_goRxId1_T_1, _T_118) @[CanBsp.scala 623:46]
    node _delayedDominantCnt_T = add(delayedDominantCnt, UInt<3>("h1")) @[CanBsp.scala 624:46]
    node _delayedDominantCnt_T_1 = tail(_delayedDominantCnt_T, 1) @[CanBsp.scala 624:46]
    node _GEN_94 = mux(_T_119, _delayedDominantCnt_T_1, delayedDominantCnt) @[CanBsp.scala 623:95 624:24 147:42]
    node _GEN_95 = mux(_T_113, UInt<3>("h0"), _GEN_94) @[CanBsp.scala 621:85 622:24]
    skip
    skip
    node _T_122 = or(_T_98, firstCompareBit) @[CanBsp.scala 627:63]
    node _T_123 = lt(passiveCnt, UInt<3>("h6")) @[CanBsp.scala 629:43]
    node _T_124 = and(io_samplePoint, _T_123) @[CanBsp.scala 629:29]
    skip
    node _T_126 = and(errorFrame, _bitErrExc4_T_2) @[CanBsp.scala 630:21]
    skip
    node _T_128 = and(_T_126, _stuffErr_T_2) @[CanBsp.scala 630:40]
    node _passiveCnt_T = add(passiveCnt, UInt<3>("h1")) @[CanBsp.scala 631:32]
    node _passiveCnt_T_1 = tail(_passiveCnt_T, 1) @[CanBsp.scala 631:32]
    node _GEN_96 = mux(_T_128, _passiveCnt_T_1, UInt<3>("h1")) @[CanBsp.scala 630:78 631:18 633:18]
    node _GEN_97 = mux(_T_124, _GEN_96, passiveCnt) @[CanBsp.scala 142:34 629:51]
    node _GEN_98 = mux(_T_122, UInt<3>("h1"), _GEN_97) @[CanBsp.scala 627:82 628:16]
    node _GEN_99 = mux(io_samplePoint, UInt<1>("h0"), firstCompareBit) @[CanBsp.scala 639:30 640:21 183:39]
    node _GEN_100 = or(io_goErrorFrame, _GEN_99) @[CanBsp.scala 637:25 638:21]
    node _T_129 = or(overloadFrameEnded, io_goErrorFrame) @[CanBsp.scala 643:27]
    node _GEN_101 = or(io_goOverloadFrame, overloadFrame) @[CanBsp.scala 645:34 646:19 79:37]
    node _GEN_102 = mux(_T_129, UInt<1>("h0"), _GEN_101) @[CanBsp.scala 643:46 644:19]
    skip
    node _T_131 = or(_T_129, io_goOverloadFrame) @[CanBsp.scala 649:45]
    node _T_132 = and(io_overloadFrame, io_txPoint) @[CanBsp.scala 651:31]
    skip
    node _T_134 = and(_T_132, _rule5_T_4) @[CanBsp.scala 651:44]
    node _overloadCnt1_T = add(overloadCnt1, UInt<3>("h1")) @[CanBsp.scala 652:34]
    node _overloadCnt1_T_1 = tail(_overloadCnt1_T, 1) @[CanBsp.scala 652:34]
    node _GEN_103 = mux(_T_134, _overloadCnt1_T_1, overloadCnt1) @[CanBsp.scala 651:67 652:18 151:36]
    node _GEN_104 = mux(_T_131, UInt<3>("h0"), _GEN_103) @[CanBsp.scala 649:67 650:18]
    skip
    skip
    node _T_137 = and(overloadFlagOver, io_sampledBit) @[CanBsp.scala 657:51]
    node _T_138 = and(io_overloadFrame, _T_137) @[CanBsp.scala 657:31]
    node _GEN_105 = or(_T_138, enableOverloadCnt2) @[CanBsp.scala 657:68 658:24 148:42]
    node _GEN_106 = mux(_T_131, UInt<1>("h0"), _GEN_105) @[CanBsp.scala 655:67 656:24]
    skip
    skip
    node _T_141 = and(enableOverloadCnt2, io_txPoint) @[CanBsp.scala 663:33]
    node _overloadCnt2_T = add(overloadCnt2, UInt<3>("h1")) @[CanBsp.scala 664:34]
    node _overloadCnt2_T_1 = tail(_overloadCnt2_T, 1) @[CanBsp.scala 664:34]
    node _GEN_107 = mux(_T_141, _overloadCnt2_T_1, overloadCnt2) @[CanBsp.scala 663:46 664:18 152:36]
    node _GEN_108 = mux(_T_131, UInt<3>("h0"), _GEN_107) @[CanBsp.scala 661:67 662:18]
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    node _T_148 = or(io_resetMode, io_nodeBusOff) @[CanBsp.scala 679:22]
    skip
    node _T_150 = lt(errorCnt1, UInt<3>("h6")) @[CanBsp.scala 683:22]
    skip
    node _GEN_114 = mux(_T_150, io_nodeErrorPassive, UInt<1>("h1")) @[CanBsp.scala 683:29 691:19]
    node _T_151 = or(io_goOverloadFrame, io_overloadFrame) @[CanBsp.scala 693:35]
    node _T_152 = lt(overloadCnt1, UInt<3>("h6")) @[CanBsp.scala 694:26]
    node _GEN_115 = mux(_T_152, UInt<1>("h0"), UInt<1>("h1")) @[CanBsp.scala 694:33 695:19 697:19]
    node _T_153 = or(io_goTx, io_txState) @[CanBsp.scala 700:25]
    skip
    skip
    node _txBit_T_1 = dshr(extendedChainDataExt, txPointer) @[CanBsp.scala 723:54]
    node _txBit_T_2 = bits(_txBit_T_1, 0, 0) @[CanBsp.scala 723:54]
    node _txBit_T_3 = dshr(extendedChainDataStd, txPointer) @[CanBsp.scala 723:86]
    node _txBit_T_4 = bits(_txBit_T_3, 0, 0) @[CanBsp.scala 723:86]
    node _txBit_T_5 = mux(_rstTxPointer_T_5, _txBit_T_2, _txBit_T_4) @[CanBsp.scala 723:19]
    node _txBit_T_6 = dshr(rCalculatedCrc, txPointer) @[CanBsp.scala 725:30]
    node _txBit_T_7 = bits(_txBit_T_6, 0, 0) @[CanBsp.scala 725:30]
    skip
    node _txBit_T_9 = dshr(extendedChainExt, txPointer) @[CanBsp.scala 729:50]
    node _txBit_T_10 = bits(_txBit_T_9, 0, 0) @[CanBsp.scala 729:50]
    node _txBit_T_11 = dshr(extendedChainStd, txPointer) @[CanBsp.scala 729:78]
    node _txBit_T_12 = bits(_txBit_T_11, 0, 0) @[CanBsp.scala 729:78]
    node _txBit_T_13 = mux(_rstTxPointer_T_5, _txBit_T_10, _txBit_T_12) @[CanBsp.scala 729:19]
    node _GEN_125 = or(finishMsg, _txBit_T_13) @[CanBsp.scala 726:27 727:13 729:13]
    node _GEN_126 = mux(rxCrc, _txBit_T_7, _GEN_125) @[CanBsp.scala 724:23 725:13]
    node _GEN_127 = mux(rxData, _txBit_T_5, _GEN_126) @[CanBsp.scala 722:18 723:13]
    node _txBit_T_14 = dshr(basicChainData, txPointer) @[CanBsp.scala 733:30]
    node _txBit_T_15 = bits(_txBit_T_14, 0, 0) @[CanBsp.scala 733:30]
    skip
    skip
    node _txBit_T_18 = dshr(basicChain, txPointer) @[CanBsp.scala 739:26]
    node _txBit_T_19 = bits(_txBit_T_18, 0, 0) @[CanBsp.scala 739:26]
    node _GEN_128 = or(finishMsg, _txBit_T_19) @[CanBsp.scala 736:27 737:13 739:13]
    node _GEN_129 = mux(rxCrc, _txBit_T_7, _GEN_128) @[CanBsp.scala 734:23 735:13]
    node _GEN_130 = mux(rxData, _txBit_T_15, _GEN_129) @[CanBsp.scala 732:18 733:13]
    node txBit = mux(io_extendedMode, _GEN_127, _GEN_130) @[CanBsp.scala 721:25]
    skip
    node _io_txNext_T_1 = and(_rstTxPointer_T, txBit) @[CanBsp.scala 701:39]
    node _io_txNext_T_2 = not(txQ) @[CanBsp.scala 701:67]
    node _io_txNext_T_3 = and(bitDeStuffTx, _io_txNext_T_2) @[CanBsp.scala 701:64]
    node _io_txNext_T_4 = or(_io_txNext_T_1, _io_txNext_T_3) @[CanBsp.scala 701:48]
    node _GEN_116 = mux(io_sendAck, UInt<1>("h0"), UInt<1>("h1")) @[CanBsp.scala 702:30 703:17 705:17]
    node _GEN_117 = mux(_T_153, _io_txNext_T_4, _GEN_116) @[CanBsp.scala 700:39 701:19]
    node _GEN_118 = mux(_T_151, _GEN_115, _GEN_117) @[CanBsp.scala 693:55]
    node _GEN_119 = mux(_GEN_84, _GEN_114, _GEN_118) @[CanBsp.scala 682:41]
    skip
    node _GEN_121 = mux(io_txPoint, io_txNext, tx) @[CanBsp.scala 103:26 711:26 712:8]
    node _GEN_122 = or(io_resetMode, _GEN_121) @[CanBsp.scala 709:22 710:8]
    node _txQ_T = not(goEarlyTxLatched) @[CanBsp.scala 718:20]
    node _txQ_T_1 = and(io_tx, _txQ_T) @[CanBsp.scala 718:18]
    node _GEN_123 = mux(io_txPoint, _txQ_T_1, txQ) @[CanBsp.scala 717:26 159:27 718:9]
    node _GEN_124 = mux(io_resetMode, UInt<1>("h0"), _GEN_123) @[CanBsp.scala 715:22 716:9]
    node _T_154 = or(io_txState, io_goTx) @[CanBsp.scala 745:52]
    node _T_155 = and(io_txPoint, _T_154) @[CanBsp.scala 745:38]
    skip
    node _T_157 = and(_T_155, _rstTxPointer_T) @[CanBsp.scala 745:63]
    node _T_158 = or(goEarlyTx, _T_157) @[CanBsp.scala 745:24]
    node _txPointer_T = add(txPointer, UInt<6>("h1")) @[CanBsp.scala 746:28]
    node _txPointer_T_1 = tail(_txPointer_T, 1) @[CanBsp.scala 746:28]
    node _GEN_132 = mux(_T_158, _txPointer_T_1, txPointer) @[CanBsp.scala 745:81 746:15 164:33]
    node _GEN_133 = mux(rstTxPointer, UInt<6>("h0"), _GEN_132) @[CanBsp.scala 743:22 744:15]
    node _T_159 = or(io_txSuccessful, io_resetMode) @[CanBsp.scala 749:24]
    skip
    node _T_161 = and(io_abortTx, errorCaptureCodeDirection) @[CanBsp.scala 749:53]
    node _T_162 = or(_T_159, _T_161) @[CanBsp.scala 749:39]
    skip
    node _T_164 = and(_goEarlyTx_T_2, io_txStateQ) @[CanBsp.scala 749:87]
    node _T_165 = and(_T_164, io_singleShotTransmission) @[CanBsp.scala 749:101]
    node _T_166 = or(_T_162, _T_165) @[CanBsp.scala 749:73]
    node _T_167 = and(io_txRequest, io_samplePoint) @[CanBsp.scala 751:27]
    node _GEN_134 = or(_T_167, needToTx) @[CanBsp.scala 751:45 752:14 93:32]
    node _GEN_135 = mux(_T_166, UInt<1>("h0"), _GEN_134) @[CanBsp.scala 749:130 750:14]
    skip
    skip
    skip
    skip
    skip
    skip
    node _io_goTx_T_6 = and(io_samplePoint, _goEarlyTx_T_5) @[CanBsp.scala 755:95]
    node _io_goTx_T_7 = or(_goEarlyTx_T_4, _io_goTx_T_6) @[CanBsp.scala 755:77]
    node _io_goTx_T_8 = and(_goEarlyTx_T_3, _io_goTx_T_7) @[CanBsp.scala 755:65]
    node _io_goTx_T_9 = or(goEarlyTx, io_rxIdle) @[CanBsp.scala 755:133]
    skip
    node _T_168 = or(io_resetMode, io_txPoint) @[CanBsp.scala 757:21]
    node _GEN_136 = or(goEarlyTx, goEarlyTxLatched) @[CanBsp.scala 759:25 760:22 126:40]
    node _GEN_137 = mux(_T_168, UInt<1>("h0"), _GEN_136) @[CanBsp.scala 757:35 758:22]
    node _T_169 = or(io_resetMode, io_goRxInter) @[CanBsp.scala 763:21]
    node _T_170 = or(_T_169, errorFrame) @[CanBsp.scala 763:36]
    node _T_171 = or(_T_170, arbitrationLost) @[CanBsp.scala 763:49]
    node _GEN_138 = or(io_goTx, txState) @[CanBsp.scala 765:23 766:13 75:31]
    node _GEN_139 = mux(_T_171, UInt<1>("h0"), _GEN_138) @[CanBsp.scala 763:68 764:13]
    node _T_172 = or(io_resetMode, goRxIdle) @[CanBsp.scala 771:27]
    node _T_173 = and(suspend, goRxId1) @[CanBsp.scala 771:48]
    node _T_174 = or(_T_172, _T_173) @[CanBsp.scala 771:38]
    node _GEN_140 = mux(_T_174, UInt<1>("h0"), transmitter) @[CanBsp.scala 771:59 772:17 87:35]
    node _GEN_141 = or(io_goTx, _GEN_140) @[CanBsp.scala 769:17 770:17]
    node _T_175 = or(io_goErrorFrame, io_goOverloadFrame) @[CanBsp.scala 775:24]
    node _T_176 = or(_T_175, io_goTx) @[CanBsp.scala 775:45]
    node _T_177 = or(_T_176, io_sendAck) @[CanBsp.scala 775:55]
    skip
    skip
    node _T_180 = and(goRxId1, _goEarlyTx_T_2) @[CanBsp.scala 777:49]
    node _T_181 = or(_T_172, _T_180) @[CanBsp.scala 777:38]
    node _T_182 = and(arbitrationLost, io_txState) @[CanBsp.scala 777:83]
    node _T_183 = or(_T_181, _T_182) @[CanBsp.scala 777:64]
    node _GEN_142 = mux(_T_183, UInt<1>("h0"), transmitting) @[CanBsp.scala 777:98 778:18 85:36]
    node _GEN_143 = or(_T_177, _GEN_142) @[CanBsp.scala 775:69 776:18]
    skip
    skip
    node _T_186 = or(io_resetMode, _io_goTx_T_6) @[CanBsp.scala 781:21]
    node _T_187 = and(io_notFirstBitOfInter, io_transmitter) @[CanBsp.scala 783:36]
    node _T_188 = and(_T_187, io_nodeErrorPassive) @[CanBsp.scala 783:53]
    node _GEN_144 = or(_T_188, suspend) @[CanBsp.scala 783:76 784:13 177:31]
    node _GEN_145 = mux(_T_186, UInt<1>("h0"), _GEN_144) @[CanBsp.scala 781:63 782:13]
    skip
    skip
    skip
    node _T_192 = and(suspend, io_samplePoint) @[CanBsp.scala 789:22]
    node _T_193 = and(_T_192, lastBitOfInter) @[CanBsp.scala 789:39]
    node _GEN_146 = or(_T_193, suspendCntEn) @[CanBsp.scala 789:57 790:18 178:36]
    node _GEN_147 = mux(_T_186, UInt<1>("h0"), _GEN_146) @[CanBsp.scala 787:64 788:18]
    skip
    skip
    skip
    node _T_197 = and(suspendCntEn, io_samplePoint) @[CanBsp.scala 795:27]
    node _suspendCnt_T = add(suspendCnt, UInt<3>("h1")) @[CanBsp.scala 796:30]
    node _suspendCnt_T_1 = tail(_suspendCnt_T, 1) @[CanBsp.scala 796:30]
    node _GEN_148 = mux(_T_197, _suspendCnt_T_1, suspendCnt) @[CanBsp.scala 795:45 796:16 179:34]
    node _GEN_149 = mux(_T_186, UInt<3>("h0"), _GEN_148) @[CanBsp.scala 793:64 794:16]
    skip
    node _T_199 = or(_T_18, errorFrame) @[CanBsp.scala 799:27]
    node _T_200 = or(_T_199, io_resetMode) @[CanBsp.scala 799:40]
    node _GEN_150 = or(goRxCrcLim, finishMsg) @[CanBsp.scala 801:26 802:15 166:33]
    node _GEN_151 = mux(_T_200, UInt<1>("h0"), _GEN_150) @[CanBsp.scala 799:56 800:15]
    node _T_201 = or(goRxIdle, errorFrameEnded) @[CanBsp.scala 805:17]
    node _T_202 = and(io_transmitter, io_samplePoint) @[CanBsp.scala 807:29]
    node _T_203 = and(_T_202, io_tx) @[CanBsp.scala 807:46]
    node _T_204 = and(_T_203, arbitrationField) @[CanBsp.scala 807:54]
    skip
    node _T_206 = and(_T_204, _goRxId1_T) @[CanBsp.scala 807:73]
    node _GEN_152 = or(_T_206, arbitrationLost) @[CanBsp.scala 807:91 808:21 154:39]
    node _GEN_153 = mux(_T_201, UInt<1>("h0"), _GEN_152) @[CanBsp.scala 805:36 806:21]
    node _GEN_154 = mux(io_samplePoint, arbitrationField, arbitrationFieldD) @[CanBsp.scala 811:24 812:23 156:41]
    node _io_setArbitrationLostIrq_T = not(arbitrationLostQ) @[CanBsp.scala 815:48]
    node _io_setArbitrationLostIrq_T_1 = and(arbitrationLost, _io_setArbitrationLostIrq_T) @[CanBsp.scala 815:45]
    node _io_setArbitrationLostIrq_T_2 = not(arbitrationBlocked) @[CanBsp.scala 815:69]
    skip
    skip
    skip
    node _arbitrationCnt_T = add(arbitrationCnt, UInt<5>("h1")) @[CanBsp.scala 819:40]
    node _arbitrationCnt_T_1 = tail(_arbitrationCnt_T, 1) @[CanBsp.scala 819:40]
    node _GEN_155 = mux(arbitrationFieldD, _arbitrationCnt_T_1, UInt<5>("h0")) @[CanBsp.scala 818:29 819:22 821:22]
    node _GEN_156 = mux(_T_60, _GEN_155, arbitrationCnt) @[CanBsp.scala 157:38 817:38]
    node _GEN_157 = mux(io_setArbitrationLostIrq, arbitrationCnt, arbitrationLostCapture) @[CanBsp.scala 825:34 826:28 99:46]
    node _GEN_158 = or(io_setArbitrationLostIrq, arbitrationBlocked) @[CanBsp.scala 831:40 832:24 158:42]
    node _GEN_159 = mux(io_readArbitrationLostCaptureReg, UInt<1>("h0"), _GEN_158) @[CanBsp.scala 829:42 830:24]
    skip
    node _T_210 = and(io_writeEnReceiveErrorCounter, _goRxIdle_T_2) @[CanBsp.scala 835:38]
    node _rxErrorCount_T = cat(UInt<1>("h0"), io_dataIn) @[Cat.scala 30:58]
    skip
    skip
    node _T_213 = or(_formErr_T_11, arbitrationLost) @[CanBsp.scala 840:48]
    node _T_214 = and(_goEarlyTx_T, _T_213) @[CanBsp.scala 840:29]
    skip
    node _T_216 = and(goRxAckLim, _bitDeStuffSet_T) @[CanBsp.scala 841:23]
    node _T_217 = not(crcErr) @[CanBsp.scala 841:47]
    node _T_218 = and(_T_216, _T_217) @[CanBsp.scala 841:44]
    node _T_219 = gt(io_rxErrorCount, UInt<9>("h0")) @[CanBsp.scala 841:75]
    node _T_220 = and(_T_218, _T_219) @[CanBsp.scala 841:56]
    node _T_221 = gt(io_rxErrorCount, UInt<9>("h7f")) @[CanBsp.scala 842:30]
    node _rxErrorCount_T_1 = sub(io_rxErrorCount, UInt<9>("h1")) @[CanBsp.scala 845:43]
    node _rxErrorCount_T_2 = tail(_rxErrorCount_T_1, 1) @[CanBsp.scala 845:43]
    node _GEN_160 = mux(_T_221, UInt<9>("h7f"), _rxErrorCount_T_2) @[CanBsp.scala 842:39 843:24 845:24]
    node _T_222 = lt(io_rxErrorCount, UInt<9>("h80")) @[CanBsp.scala 847:34]
    skip
    node _T_223 = not(rule5) @[CanBsp.scala 848:32]
    node _T_224 = and(io_goErrorFrame, _T_223) @[CanBsp.scala 848:30]
    node _rxErrorCount_T_3 = add(io_rxErrorCount, UInt<9>("h1")) @[CanBsp.scala 849:43]
    node _rxErrorCount_T_4 = tail(_rxErrorCount_T_3, 1) @[CanBsp.scala 849:43]
    node _T_225 = not(errorFlagOverLatched) @[CanBsp.scala 850:38]
    node _T_226 = and(errorFlagOver, _T_225) @[CanBsp.scala 850:35]
    node _T_227 = and(_T_226, io_samplePoint) @[CanBsp.scala 850:61]
    skip
    node _T_229 = and(_T_227, _goRxId1_T) @[CanBsp.scala 850:78]
    skip
    node _T_231 = and(_T_229, _bitErrExc4_T) @[CanBsp.scala 850:97]
    node _T_232 = and(io_goErrorFrame, rule5) @[CanBsp.scala 851:37]
    node _T_233 = or(_T_231, _T_232) @[CanBsp.scala 850:119]
    skip
    skip
    node _T_236 = eq(delayedDominantCnt, UInt<3>("h7")) @[CanBsp.scala 851:105]
    node _T_237 = and(_goRxId1_T_1, _T_236) @[CanBsp.scala 851:83]
    node _T_238 = or(_T_233, _T_237) @[CanBsp.scala 851:46]
    node _rxErrorCount_T_5 = add(io_rxErrorCount, UInt<9>("h8")) @[CanBsp.scala 852:43]
    node _rxErrorCount_T_6 = tail(_rxErrorCount_T_5, 1) @[CanBsp.scala 852:43]
    node _GEN_161 = mux(_T_238, _rxErrorCount_T_6, rxErrorCount) @[CanBsp.scala 851:116 852:24 95:36]
    node _GEN_162 = mux(_T_224, _rxErrorCount_T_4, _GEN_161) @[CanBsp.scala 848:40 849:24]
    node _GEN_163 = mux(_T_222, _GEN_162, rxErrorCount) @[CanBsp.scala 847:43 95:36]
    node _GEN_164 = mux(_T_220, _GEN_160, _GEN_163) @[CanBsp.scala 841:83]
    node _GEN_165 = mux(_T_214, _GEN_164, rxErrorCount) @[CanBsp.scala 840:68 95:36]
    node _GEN_166 = mux(io_setResetMode, UInt<9>("h0"), _GEN_165) @[CanBsp.scala 837:30 838:18]
    node _GEN_167 = mux(_T_210, _rxErrorCount_T, _GEN_166) @[CanBsp.scala 835:58 836:18]
    skip
    node _T_239 = gt(txErrorCount, UInt<9>("h0")) @[CanBsp.scala 863:30]
    node _T_240 = or(io_txSuccessful, busFree) @[CanBsp.scala 863:56]
    node _T_241 = and(_T_239, _T_240) @[CanBsp.scala 863:37]
    node _txErrorCount_T_1 = sub(txErrorCount, UInt<9>("h1")) @[CanBsp.scala 864:36]
    node _txErrorCount_T_2 = tail(_txErrorCount_T_1, 1) @[CanBsp.scala 864:36]
    skip
    node _T_243 = and(io_transmitter, _io_txSuccessful_T_7) @[CanBsp.scala 865:31]
    skip
    skip
    skip
    skip
    skip
    node _T_249 = or(_T_237, _T_232) @[CanBsp.scala 866:79]
    skip
    skip
    node _T_252 = not(_T_83) @[CanBsp.scala 867:29]
    node _T_253 = and(io_goErrorFrame, _T_252) @[CanBsp.scala 867:26]
    node _T_254 = and(io_transmitter, stuffErr) @[CanBsp.scala 868:29]
    node _T_255 = and(_T_254, arbitrationField) @[CanBsp.scala 868:40]
    node _T_256 = and(_T_255, io_samplePoint) @[CanBsp.scala 868:59]
    node _T_257 = and(_T_256, io_tx) @[CanBsp.scala 868:76]
    skip
    node _T_259 = and(_T_257, _goRxId1_T) @[CanBsp.scala 868:84]
    node _T_260 = not(_T_259) @[CanBsp.scala 868:12]
    node _T_261 = and(_T_253, _T_260) @[CanBsp.scala 867:79]
    node _T_262 = or(_T_249, _T_261) @[CanBsp.scala 866:107]
    node _T_263 = and(errorFrame, rule3Exc1_1) @[CanBsp.scala 869:21]
    node _T_264 = or(_T_262, _T_263) @[CanBsp.scala 868:106]
    node _txErrorCount_T_3 = add(txErrorCount, UInt<9>("h8")) @[CanBsp.scala 870:38]
    node _txErrorCount_T_4 = tail(_txErrorCount_T_3, 1) @[CanBsp.scala 870:38]
    node _GEN_168 = mux(_T_264, _txErrorCount_T_4, txErrorCount) @[CanBsp.scala 869:38 870:22 97:36]
    node _GEN_169 = mux(_T_243, _GEN_168, txErrorCount) @[CanBsp.scala 865:53 97:36]
    node _GEN_170 = mux(_T_241, _txErrorCount_T_2, _GEN_169) @[CanBsp.scala 863:68 864:20]
    node _GEN_171 = mux(io_setResetMode, UInt<9>("h80"), _GEN_170) @[CanBsp.scala 861:27 862:20]
    node _GEN_172 = mux(io_writeEnTransmitErrorCounter, _rxErrorCount_T, _GEN_171) @[CanBsp.scala 858:40 859:18]
    node _T_265 = lt(rxErrorCount, UInt<9>("h80")) @[CanBsp.scala 875:22]
    node _T_266 = lt(txErrorCount, UInt<9>("h80")) @[CanBsp.scala 875:47]
    node _T_267 = and(_T_265, _T_266) @[CanBsp.scala 875:31]
    node _T_268 = geq(rxErrorCount, UInt<9>("h80")) @[CanBsp.scala 877:29]
    node _T_269 = geq(txErrorCount, UInt<9>("h80")) @[CanBsp.scala 877:55]
    node _T_270 = or(_T_268, _T_269) @[CanBsp.scala 877:39]
    skip
    skip
    node _T_273 = and(_bitErr_T_18, resetModeQ) @[CanBsp.scala 877:121]
    node _T_274 = or(_T_97, _T_273) @[CanBsp.scala 877:103]
    node _T_275 = and(_T_270, _T_274) @[CanBsp.scala 877:66]
    skip
    node _T_277 = and(_T_275, _goRxIdle_T_2) @[CanBsp.scala 877:135]
    node _GEN_173 = or(_T_277, nodeErrorPassive) @[CanBsp.scala 877:155 878:22 101:40]
    node _GEN_174 = mux(_T_267, UInt<1>("h0"), _GEN_173) @[CanBsp.scala 875:57 876:22]
    node _io_nodeErrorActive_T = or(io_nodeErrorPassive, io_nodeBusOff) @[CanBsp.scala 881:47]
    skip
    node _T_278 = eq(io_rxErrorCount, UInt<9>("h0")) @[CanBsp.scala 883:25]
    node _T_279 = eq(io_txErrorCount, UInt<9>("h0")) @[CanBsp.scala 883:53]
    node _T_280 = and(_T_278, _T_279) @[CanBsp.scala 883:34]
    skip
    node _T_282 = and(_T_280, _bitErr_T_18) @[CanBsp.scala 883:62]
    node _T_283 = lt(io_dataIn, UInt<8>("hff")) @[CanBsp.scala 883:127]
    node _T_284 = and(io_writeEnTransmitErrorCounter, _T_283) @[CanBsp.scala 883:114]
    node _T_285 = or(_T_282, _T_284) @[CanBsp.scala 883:80]
    node _T_286 = geq(io_txErrorCount, UInt<9>("h100")) @[CanBsp.scala 885:31]
    node _T_287 = eq(io_dataIn, UInt<8>("hff")) @[CanBsp.scala 885:88]
    node _T_288 = and(io_writeEnTransmitErrorCounter, _T_287) @[CanBsp.scala 885:75]
    node _T_289 = or(_T_286, _T_288) @[CanBsp.scala 885:41]
    node _GEN_175 = or(_T_289, nodeBusOff) @[CanBsp.scala 885:101 886:16 91:34]
    node _GEN_176 = mux(_T_285, UInt<1>("h0"), _GEN_175) @[CanBsp.scala 883:138 884:16]
    node _T_290 = and(io_sampledBit, busFreeCntEn) @[CanBsp.scala 890:24]
    node _T_291 = lt(busFreeCnt, UInt<4>("ha")) @[CanBsp.scala 890:52]
    node _T_292 = and(_T_290, _T_291) @[CanBsp.scala 890:39]
    node _busFreeCnt_T = add(busFreeCnt, UInt<4>("h1")) @[CanBsp.scala 891:32]
    node _busFreeCnt_T_1 = tail(_busFreeCnt_T, 1) @[CanBsp.scala 891:32]
    node _GEN_177 = mux(_T_292, _busFreeCnt_T_1, UInt<4>("h0")) @[CanBsp.scala 890:60 891:18 893:18]
    node _GEN_178 = mux(io_samplePoint, _GEN_177, busFreeCnt) @[CanBsp.scala 889:24 167:34]
    skip
    skip
    skip
    node _T_296 = and(nodeBusOffQ, _bitErr_T_18) @[CanBsp.scala 897:51]
    node _T_297 = or(_T_273, _T_296) @[CanBsp.scala 897:37]
    skip
    skip
    skip
    skip
    node _T_302 = and(_busFree_T_2, _goRxIdle_T_2) @[CanBsp.scala 899:69]
    node _GEN_179 = mux(_T_302, UInt<1>("h0"), busFreeCntEn) @[CanBsp.scala 899:87 900:18 168:36]
    node _GEN_180 = or(_T_297, _GEN_179) @[CanBsp.scala 897:68 898:18]
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    node _GEN_182 = or(_T_296, waitingForBusFree) @[CanBsp.scala 911:43 912:23 169:41]
    node _GEN_183 = mux(_goRxIdle_T_3, UInt<1>("h0"), _GEN_182) @[CanBsp.scala 909:34 910:23]
    node _GEN_186 = mux(stuffErr, UInt<2>("h2"), UInt<2>("h3")) @[CanBsp.scala 930:24 931:26 933:26]
    node _GEN_187 = mux(formErr, UInt<2>("h1"), _GEN_186) @[CanBsp.scala 928:23 929:26]
    node errorCaptureCodeType = mux(bitErr, UInt<2>("h0"), _GEN_187) @[CanBsp.scala 926:16 927:26]
    skip
    node errorCaptureCode_hi = cat(errorCaptureCodeType, errorCaptureCodeDirection) @[Cat.scala 30:58]
    node _lo_T = or(rxCrcLim, rxAck) @[CanBsp.scala 920:43]
    node _lo_T_1 = or(_lo_T, rxAckLim) @[CanBsp.scala 920:51]
    node _lo_T_2 = or(_lo_T_1, rxEof) @[CanBsp.scala 920:62]
    node _lo_T_3 = or(_lo_T_2, io_rxInter) @[CanBsp.scala 920:70]
    node _lo_T_4 = or(_lo_T_3, errorFrame) @[CanBsp.scala 920:83]
    node lo_hi_hi_hi = or(_lo_T_4, io_overloadFrame) @[CanBsp.scala 920:96]
    node _lo_T_5 = gt(bitCnt, UInt<6>("h4")) @[CanBsp.scala 921:22]
    node _lo_T_6 = and(rxId2, _lo_T_5) @[CanBsp.scala 921:12]
    node _lo_T_7 = or(_lo_T_6, rxRtr2) @[CanBsp.scala 921:30]
    node _lo_T_8 = or(_lo_T_7, rxR1) @[CanBsp.scala 921:39]
    node _lo_T_9 = or(_lo_T_8, rxR0) @[CanBsp.scala 921:46]
    node _lo_T_10 = or(_lo_T_9, rxDlc) @[CanBsp.scala 921:53]
    node _lo_T_11 = or(_lo_T_10, rxData) @[CanBsp.scala 921:61]
    node _lo_T_12 = or(_lo_T_11, rxCrc) @[CanBsp.scala 921:70]
    node _lo_T_13 = or(_lo_T_12, rxCrcLim) @[CanBsp.scala 921:78]
    node _lo_T_14 = or(_lo_T_13, rxAck) @[CanBsp.scala 921:89]
    node _lo_T_15 = or(_lo_T_14, rxAckLim) @[CanBsp.scala 921:97]
    node _lo_T_16 = or(_lo_T_15, rxEof) @[CanBsp.scala 921:108]
    node lo_hi_hi_lo = or(_lo_T_16, io_overloadFrame) @[CanBsp.scala 921:116]
    node lo_hi_hi = cat(lo_hi_hi_hi, lo_hi_hi_lo) @[Cat.scala 30:58]
    node _lo_T_17 = gt(bitCnt, UInt<6>("h7")) @[CanBsp.scala 922:22]
    node _lo_T_18 = and(rxId1, _lo_T_17) @[CanBsp.scala 922:12]
    node _lo_T_19 = or(_lo_T_18, rxRtr1) @[CanBsp.scala 922:30]
    node _lo_T_20 = or(_lo_T_19, rxIde) @[CanBsp.scala 922:39]
    node _lo_T_21 = or(_lo_T_20, rxId2) @[CanBsp.scala 922:47]
    node _lo_T_22 = or(_lo_T_21, rxRtr2) @[CanBsp.scala 922:55]
    node _lo_T_23 = or(_lo_T_22, rxR1) @[CanBsp.scala 922:64]
    skip
    node _lo_T_25 = or(_lo_T_23, _bitErrExc3_T) @[CanBsp.scala 922:71]
    node lo_hi_lo = or(_lo_T_25, io_overloadFrame) @[CanBsp.scala 922:106]
    node lo_hi = cat(lo_hi_hi, lo_hi_lo) @[Cat.scala 30:58]
    node _lo_T_26 = or(io_rxIdle, rxId1) @[CanBsp.scala 923:13]
    node _lo_T_27 = or(_lo_T_26, rxId2) @[CanBsp.scala 923:21]
    node _lo_T_28 = or(_lo_T_27, rxDlc) @[CanBsp.scala 923:29]
    node _lo_T_29 = or(_lo_T_28, rxData) @[CanBsp.scala 923:37]
    node _lo_T_30 = or(_lo_T_29, rxAckLim) @[CanBsp.scala 923:46]
    node _lo_T_31 = or(_lo_T_30, rxEof) @[CanBsp.scala 923:57]
    node _lo_T_32 = or(_lo_T_31, io_rxInter) @[CanBsp.scala 923:65]
    skip
    node lo_lo_hi = or(_lo_T_32, _bitErrExc3_T) @[CanBsp.scala 923:78]
    node _lo_T_34 = or(io_rxIdle, rxIde) @[CanBsp.scala 924:15]
    node _lo_T_35 = lt(bitCnt, UInt<6>("hd")) @[CanBsp.scala 924:42]
    node _lo_T_36 = and(rxId2, _lo_T_35) @[CanBsp.scala 924:32]
    node _lo_T_37 = or(_lo_T_34, _lo_T_36) @[CanBsp.scala 924:23]
    node _lo_T_38 = or(_lo_T_37, rxR1) @[CanBsp.scala 924:51]
    node _lo_T_39 = or(_lo_T_38, rxR0) @[CanBsp.scala 924:58]
    node _lo_T_40 = or(_lo_T_39, rxDlc) @[CanBsp.scala 924:65]
    node _lo_T_41 = or(_lo_T_40, rxAck) @[CanBsp.scala 924:73]
    node _lo_T_42 = or(_lo_T_41, rxAckLim) @[CanBsp.scala 924:81]
    node _lo_T_43 = and(errorFrame, io_nodeErrorActive) @[CanBsp.scala 924:105]
    node lo_lo_lo = or(_lo_T_42, _lo_T_43) @[CanBsp.scala 924:92]
    node lo_lo = cat(lo_lo_hi, lo_lo_lo) @[Cat.scala 30:58]
    node errorCaptureCodeSegment = cat(lo_hi, lo_lo) @[Cat.scala 30:58]
    skip
    node _errorCaptureCode_T = cat(errorCaptureCode_hi, errorCaptureCodeSegment) @[Cat.scala 30:58]
    node _GEN_184 = mux(io_setBusErrorIrq, _errorCaptureCode_T, errorCaptureCode) @[CanBsp.scala 917:33 918:22 81:40]
    node _GEN_185 = mux(io_readErrorCodeCaptureReg, UInt<8>("h0"), _GEN_184) @[CanBsp.scala 915:36 916:22]
    node _io_transmitStatus_T = and(io_extendedMode, waitingForBusFree) @[CanBsp.scala 936:60]
    skip
    node _io_receiveStatus_T = not(io_rxIdle) @[CanBsp.scala 937:67]
    skip
    node _io_receiveStatus_T_2 = and(_io_receiveStatus_T, errorCaptureCodeDirection) @[CanBsp.scala 937:79]
    node _io_receiveStatus_T_3 = or(waitingForBusFree, _io_receiveStatus_T_2) @[CanBsp.scala 937:63]
    node _io_receiveStatus_T_4 = not(waitingForBusFree) @[CanBsp.scala 938:46]
    skip
    node _io_receiveStatus_T_6 = and(_io_receiveStatus_T_4, _io_receiveStatus_T) @[CanBsp.scala 938:66]
    skip
    node _io_receiveStatus_T_8 = and(_io_receiveStatus_T_6, errorCaptureCodeDirection) @[CanBsp.scala 938:82]
    skip
    node _io_setBusErrorIrq_T = not(errorCaptureCodeBlocked) @[CanBsp.scala 939:43]
    skip
    node _GEN_189 = or(io_setBusErrorIrq, errorCaptureCodeBlocked) @[CanBsp.scala 943:33 944:29 182:47]
    node _GEN_190 = mux(io_readErrorCodeCaptureReg, UInt<1>("h0"), _GEN_189) @[CanBsp.scala 941:36 942:29]
    skip
    node _GEN_235 = pad(io_errorWarningLimit, 9) @[CanBsp.scala 947:58]
    node _io_errorStatus_T = geq(io_rxErrorCount, _GEN_235) @[CanBsp.scala 947:58]
    skip
    node _io_errorStatus_T_1 = geq(io_txErrorCount, _GEN_235) @[CanBsp.scala 947:102]
    node _io_errorStatus_T_2 = or(_io_errorStatus_T, _io_errorStatus_T_1) @[CanBsp.scala 947:83]
    node _io_errorStatus_T_3 = geq(io_rxErrorCount, UInt<9>("h60")) @[CanBsp.scala 948:58]
    node _io_errorStatus_T_4 = geq(io_txErrorCount, UInt<9>("h60")) @[CanBsp.scala 948:86]
    node _io_errorStatus_T_5 = or(_io_errorStatus_T_3, _io_errorStatus_T_4) @[CanBsp.scala 948:67]
    skip
    skip
    skip
    skip
    io_dataOut <= canFifo.io_dataOut @[CanBsp.scala 582:14]
    io_txState <= txState @[CanBsp.scala 76:14]
    io_txStateQ <= txStateQ @[CanBsp.scala 78:15]
    io_overloadFrame <= overloadFrame @[CanBsp.scala 80:20]
    io_errorCaptureCode <= errorCaptureCode @[CanBsp.scala 82:23]
    io_rxIdle <= rxIdle @[CanBsp.scala 84:13]
    io_transmitting <= transmitting @[CanBsp.scala 86:19]
    io_transmitter <= transmitter @[CanBsp.scala 88:18]
    io_goRxInter <= or(_io_goRxInter_T_3, overloadFrameEnded) @[CanBsp.scala 235:84]
    io_notFirstBitOfInter <= and(io_rxInter, _io_notFirstBitOfInter_T_1) @[CanBsp.scala 286:39]
    io_rxInter <= rxInter @[CanBsp.scala 90:14]
    io_setResetMode <= and(nodeBusOff, _io_setResetMode_T) @[CanBsp.scala 288:33]
    io_nodeBusOff <= nodeBusOff @[CanBsp.scala 92:17]
    io_errorStatus <= mux(io_extendedMode, _io_errorStatus_T_2, _io_errorStatus_T_5) @[CanBsp.scala 947:24]
    io_rxErrorCount <= rxErrorCount @[CanBsp.scala 96:19]
    io_txErrorCount <= txErrorCount @[CanBsp.scala 98:19]
    io_transmitStatus <= or(io_transmitting, _io_transmitStatus_T) @[CanBsp.scala 936:41]
    io_receiveStatus <= mux(io_extendedMode, _io_receiveStatus_T_3, _io_receiveStatus_T_8) @[CanBsp.scala 937:27]
    io_txSuccessful <= and(_io_txSuccessful_T_6, _io_txSuccessful_T_7) @[CanBsp.scala 289:115]
    io_needToTx <= needToTx @[CanBsp.scala 94:15]
    io_overrun <= canFifo.io_overrun @[CanBsp.scala 586:14]
    io_infoEmpty <= canFifo.io_infoEmpty @[CanBsp.scala 588:16]
    io_setBusErrorIrq <= and(io_goErrorFrame, _io_setBusErrorIrq_T) @[CanBsp.scala 939:40]
    io_setArbitrationLostIrq <= and(_io_setArbitrationLostIrq_T_1, _io_setArbitrationLostIrq_T_2) @[CanBsp.scala 815:67]
    io_arbitrationLostCapture <= arbitrationLostCapture @[CanBsp.scala 100:29]
    io_nodeErrorPassive <= nodeErrorPassive @[CanBsp.scala 102:23]
    io_nodeErrorActive <= not(_io_nodeErrorActive_T) @[CanBsp.scala 881:25]
    io_rxMessageCounter <= canFifo.io_infoCnt @[CanBsp.scala 587:23]
    io_tx <= tx @[CanBsp.scala 104:9]
    io_txNext <= or(_T_148, _GEN_119) @[CanBsp.scala 679:39 680:15]
    io_busOffOn <= not(io_nodeBusOff) @[CanBsp.scala 946:18]
    io_goOverloadFrame <= or(_io_goOverloadFrame_T_16, _io_goOverloadFrame_T_22) @[CanBsp.scala 284:154]
    io_goErrorFrame <= or(_err_T_2, _io_goErrorFrame_T_3) @[CanBsp.scala 287:60]
    io_goTx <= and(_io_goTx_T_8, _io_goTx_T_9) @[CanBsp.scala 755:120]
    io_sendAck <= and(_io_sendAck_T_3, _goEarlyTx_T) @[CanBsp.scala 261:48]
    txState <= mux(reset, UInt<1>("h0"), _GEN_139) @[CanBsp.scala 75:{31,31}]
    txStateQ <= mux(reset, UInt<1>("h0"), _txStateQ_T) @[CanBsp.scala 77:{32,32,32}]
    overloadFrame <= mux(reset, UInt<1>("h0"), _GEN_102) @[CanBsp.scala 79:{37,37}]
    errorCaptureCode <= mux(reset, UInt<8>("h0"), _GEN_185) @[CanBsp.scala 81:{40,40}]
    rxIdle <= mux(reset, UInt<1>("h0"), _GEN_1) @[CanBsp.scala 83:{30,30}]
    transmitting <= mux(reset, UInt<1>("h0"), _GEN_143) @[CanBsp.scala 85:{36,36}]
    transmitter <= mux(reset, UInt<1>("h0"), _GEN_141) @[CanBsp.scala 87:{35,35}]
    rxInter <= mux(reset, UInt<1>("h0"), _GEN_31) @[CanBsp.scala 89:{31,31}]
    nodeBusOff <= mux(reset, UInt<1>("h0"), _GEN_176) @[CanBsp.scala 91:{34,34}]
    needToTx <= mux(reset, UInt<1>("h0"), _GEN_135) @[CanBsp.scala 93:{32,32}]
    rxErrorCount <= mux(reset, UInt<9>("h0"), _GEN_167) @[CanBsp.scala 95:{36,36}]
    txErrorCount <= mux(reset, UInt<9>("h0"), _GEN_172) @[CanBsp.scala 97:{36,36}]
    arbitrationLostCapture <= mux(reset, UInt<5>("h0"), _GEN_157) @[CanBsp.scala 99:{46,46}]
    nodeErrorPassive <= mux(reset, UInt<1>("h0"), _GEN_174) @[CanBsp.scala 101:{40,40}]
    tx <= or(reset, _GEN_122) @[CanBsp.scala 103:{26,26}]
    resetModeQ <= mux(reset, UInt<1>("h0"), io_resetMode) @[CanBsp.scala 105:{34,34,34}]
    bitCnt <= mux(reset, UInt<6>("h0"), _GEN_48) @[CanBsp.scala 106:{30,30}]
    dataLen <= mux(reset, UInt<4>("h0"), _GEN_36) @[CanBsp.scala 107:{31,31}]
    id <= mux(reset, UInt<29>("h0"), _GEN_32) @[CanBsp.scala 108:{26,26}]
    bitStuffCnt <= mux(reset, UInt<3>("h1"), _GEN_57) @[CanBsp.scala 109:{35,35}]
    bitStuffCntTx <= mux(reset, UInt<3>("h1"), _GEN_61) @[CanBsp.scala 110:{37,37}]
    txPointQ <= mux(reset, UInt<1>("h0"), _txPointQ_T) @[CanBsp.scala 111:{32,32,32}]
    rxId1 <= mux(reset, UInt<1>("h0"), _GEN_3) @[CanBsp.scala 112:{29,29}]
    rxRtr1 <= mux(reset, UInt<1>("h0"), _GEN_5) @[CanBsp.scala 113:{30,30}]
    rxIde <= mux(reset, UInt<1>("h0"), _GEN_7) @[CanBsp.scala 114:{29,29}]
    rxId2 <= mux(reset, UInt<1>("h0"), _GEN_9) @[CanBsp.scala 115:{29,29}]
    rxRtr2 <= mux(reset, UInt<1>("h0"), _GEN_11) @[CanBsp.scala 116:{30,30}]
    rxR0 <= mux(reset, UInt<1>("h0"), _GEN_15) @[CanBsp.scala 117:{28,28}]
    rxR1 <= mux(reset, UInt<1>("h0"), _GEN_13) @[CanBsp.scala 118:{27,27}]
    rxDlc <= mux(reset, UInt<1>("h0"), _GEN_17) @[CanBsp.scala 119:{29,29}]
    rxData <= mux(reset, UInt<1>("h0"), _GEN_19) @[CanBsp.scala 120:{30,30}]
    rxCrc <= mux(reset, UInt<1>("h0"), _GEN_21) @[CanBsp.scala 121:{29,29}]
    rxCrcLim <= mux(reset, UInt<1>("h0"), _GEN_23) @[CanBsp.scala 122:{32,32}]
    rxAck <= mux(reset, UInt<1>("h0"), _GEN_25) @[CanBsp.scala 123:{29,29}]
    rxAckLim <= mux(reset, UInt<1>("h0"), _GEN_27) @[CanBsp.scala 124:{32,32}]
    rxEof <= mux(reset, UInt<1>("h0"), _GEN_29) @[CanBsp.scala 125:{29,29}]
    goEarlyTxLatched <= mux(reset, UInt<1>("h0"), _GEN_137) @[CanBsp.scala 126:{40,40}]
    rtr1 <= mux(reset, UInt<1>("h0"), _GEN_33) @[CanBsp.scala 127:{28,28}]
    ide <= mux(reset, UInt<1>("h0"), _GEN_35) @[CanBsp.scala 128:{27,27}]
    rtr2 <= mux(reset, UInt<1>("h0"), _GEN_34) @[CanBsp.scala 129:{28,28}]
    crcIn <= mux(reset, UInt<15>("h0"), _GEN_46) @[CanBsp.scala 130:{29,29}]
    tmpData <= mux(reset, UInt<8>("h0"), _GEN_37) @[CanBsp.scala 131:{31,31}]
    tmpFifo.canAcf_io_data0_MPORT.addr <= UInt<3>("h0") @[CanBsp.scala 537:29]
    tmpFifo.canAcf_io_data0_MPORT.en <= UInt<1>("h1") @[CanBsp.scala 537:29]
    tmpFifo.canAcf_io_data0_MPORT.clk <= clock @[CanBsp.scala 537:29]
    tmpFifo.canAcf_io_data1_MPORT.addr <= UInt<3>("h1") @[CanBsp.scala 538:29]
    tmpFifo.canAcf_io_data1_MPORT.en <= UInt<1>("h1") @[CanBsp.scala 538:29]
    tmpFifo.canAcf_io_data1_MPORT.clk <= clock @[CanBsp.scala 538:29]
    tmpFifo.dataForFifo_MPORT.addr <= bits(_dataForFifo_T_2, 2, 0) @[CanBsp.scala 565:45]
    tmpFifo.dataForFifo_MPORT.en <= UInt<1>("h1") @[CanBsp.scala 565:45]
    tmpFifo.dataForFifo_MPORT.clk <= clock @[CanBsp.scala 565:45]
    tmpFifo.MPORT.addr <= byteCnt @[CanBsp.scala 423:28]
    tmpFifo.MPORT.en <= writeDataToTmpFifo @[CanBsp.scala 423:28 136:32]
    tmpFifo.MPORT.clk <= clock @[CanBsp.scala 423:28]
    tmpFifo.MPORT.data <= tmpData @[CanBsp.scala 423:28]
    tmpFifo.MPORT.mask <= UInt<1>("h1") @[CanBsp.scala 423:28]
    writeDataToTmpFifo <= mux(reset, UInt<1>("h0"), _T_45) @[CanBsp.scala 137:{42,42}]
    byteCnt <= mux(reset, UInt<3>("h0"), _GEN_40) @[CanBsp.scala 138:{31,31}]
    bitStuffCntEn <= mux(reset, UInt<1>("h0"), _GEN_53) @[CanBsp.scala 139:{37,37}]
    crcEnable <= mux(reset, UInt<1>("h0"), _GEN_63) @[CanBsp.scala 140:{33,33}]
    eofCnt <= mux(reset, UInt<3>("h0"), _GEN_51) @[CanBsp.scala 141:{30,30}]
    passiveCnt <= mux(reset, UInt<3>("h1"), _GEN_98) @[CanBsp.scala 142:{34,34}]
    errorFrame <= mux(reset, UInt<1>("h0"), _GEN_85) @[CanBsp.scala 143:{34,34}]
    enableErrorCnt2 <= mux(reset, UInt<1>("h0"), _GEN_91) @[CanBsp.scala 144:{39,39}]
    errorCnt1 <= mux(reset, UInt<3>("h0"), _GEN_87) @[CanBsp.scala 145:{33,33}]
    errorCnt2 <= mux(reset, UInt<3>("h0"), _GEN_93) @[CanBsp.scala 146:{33,33}]
    delayedDominantCnt <= mux(reset, UInt<3>("h0"), _GEN_95) @[CanBsp.scala 147:{42,42}]
    enableOverloadCnt2 <= mux(reset, UInt<1>("h0"), _GEN_106) @[CanBsp.scala 148:{42,42}]
    skip
    skip
    overloadCnt1 <= mux(reset, UInt<3>("h0"), _GEN_104) @[CanBsp.scala 151:{36,36}]
    overloadCnt2 <= mux(reset, UInt<3>("h0"), _GEN_108) @[CanBsp.scala 152:{36,36}]
    crcErr <= mux(reset, UInt<1>("h0"), _GEN_65) @[CanBsp.scala 153:{30,30}]
    arbitrationLost <= mux(reset, UInt<1>("h0"), _GEN_153) @[CanBsp.scala 154:{39,39}]
    arbitrationLostQ <= mux(reset, UInt<1>("h0"), arbitrationLost) @[CanBsp.scala 155:{40,40,40}]
    arbitrationFieldD <= mux(reset, UInt<1>("h0"), _GEN_154) @[CanBsp.scala 156:{41,41}]
    arbitrationCnt <= mux(reset, UInt<5>("h0"), _GEN_156) @[CanBsp.scala 157:{38,38}]
    arbitrationBlocked <= mux(reset, UInt<1>("h0"), _GEN_159) @[CanBsp.scala 158:{42,42}]
    txQ <= mux(reset, UInt<1>("h0"), _GEN_124) @[CanBsp.scala 159:{27,27}]
    node _GEN_237 = mux(reset, UInt<5>("h0"), _GEN_83) @[CanBsp.scala 160:{31,31}]
    dataCnt <= bits(_GEN_237, 3, 0) @[CanBsp.scala 160:{31,31}]
    node _GEN_238 = mux(reset, UInt<4>("h0"), _GEN_81) @[CanBsp.scala 161:{33,33}]
    headerCnt <= bits(_GEN_238, 2, 0) @[CanBsp.scala 161:{33,33}]
    wrFifo <= mux(reset, UInt<1>("h0"), _GEN_79) @[CanBsp.scala 162:{30,30}]
    txPointer <= mux(reset, UInt<6>("h0"), _GEN_133) @[CanBsp.scala 164:{33,33}]
    finishMsg <= mux(reset, UInt<1>("h0"), _GEN_151) @[CanBsp.scala 166:{33,33}]
    busFreeCnt <= mux(reset, UInt<4>("h0"), _GEN_178) @[CanBsp.scala 167:{34,34}]
    busFreeCntEn <= mux(reset, UInt<1>("h0"), _GEN_180) @[CanBsp.scala 168:{36,36}]
    waitingForBusFree <= or(reset, _GEN_183) @[CanBsp.scala 169:{41,41}]
    busFree <= mux(reset, UInt<1>("h0"), _busFree_T_3) @[CanBsp.scala 170:{31,31}]
    nodeBusOffQ <= mux(reset, UInt<1>("h0"), io_nodeBusOff) @[CanBsp.scala 171:{35,35,35}]
    ackErrLatched <= mux(reset, UInt<1>("h0"), _GEN_67) @[CanBsp.scala 172:{37,37}]
    bitErrLatched <= mux(reset, UInt<1>("h0"), _GEN_69) @[CanBsp.scala 173:{37,37}]
    stuffErrLatched <= mux(reset, UInt<1>("h0"), _GEN_75) @[CanBsp.scala 174:{39,39}]
    formErrLatched <= mux(reset, UInt<1>("h0"), _GEN_77) @[CanBsp.scala 175:{38,38}]
    rule3Exc1_0 <= mux(reset, UInt<1>("h0"), _GEN_73) @[CanBsp.scala 176:{38,38}]
    rule3Exc1_1 <= mux(reset, UInt<1>("h0"), _GEN_71) @[CanBsp.scala 176:{38,38}]
    suspend <= mux(reset, UInt<1>("h0"), _GEN_145) @[CanBsp.scala 177:{31,31}]
    suspendCntEn <= mux(reset, UInt<1>("h0"), _GEN_147) @[CanBsp.scala 178:{36,36}]
    suspendCnt <= mux(reset, UInt<3>("h0"), _GEN_149) @[CanBsp.scala 179:{34,34}]
    errorFlagOverLatched <= mux(reset, UInt<1>("h0"), _GEN_89) @[CanBsp.scala 180:{44,44}]
    errorCaptureCodeBlocked <= mux(reset, UInt<1>("h0"), _GEN_190) @[CanBsp.scala 182:{47,47}]
    firstCompareBit <= mux(reset, UInt<1>("h0"), _GEN_100) @[CanBsp.scala 183:{39,39}]
    canCrcRx.clock <= clock
    canCrcRx.reset <= or(io_hardSync, io_goTx) @[CanBsp.scala 211:40]
    canCrcRx.io_data <= io_sampledBit @[CanBsp.scala 523:20]
    canCrcRx.io_enable <= and(_canCrcRx_io_enable_T, _goRxRtr1_T) @[CanBsp.scala 524:52]
    canAcf.clock <= clock
    canAcf.reset <= reset
    canAcf.io_id <= id @[CanBsp.scala 528:16]
    canAcf.io_resetMode <= io_resetMode @[CanBsp.scala 529:23]
    canAcf.io_acceptanceFilterMode <= io_acceptanceFilterMode @[CanBsp.scala 530:34]
    canAcf.io_extendedMode <= io_extendedMode @[CanBsp.scala 531:26]
    canAcf.io_acceptanceCode_0 <= io_acceptanceCode_0 @[CanBsp.scala 532:28]
    canAcf.io_acceptanceCode_1 <= io_acceptanceCode_1 @[CanBsp.scala 532:28]
    canAcf.io_acceptanceCode_2 <= io_acceptanceCode_2 @[CanBsp.scala 532:28]
    canAcf.io_acceptanceCode_3 <= io_acceptanceCode_3 @[CanBsp.scala 532:28]
    canAcf.io_acceptanceMask_0 <= io_acceptanceMask_0 @[CanBsp.scala 533:28]
    canAcf.io_acceptanceMask_1 <= io_acceptanceMask_1 @[CanBsp.scala 533:28]
    canAcf.io_acceptanceMask_2 <= io_acceptanceMask_2 @[CanBsp.scala 533:28]
    canAcf.io_acceptanceMask_3 <= io_acceptanceMask_3 @[CanBsp.scala 533:28]
    canAcf.io_goRxCrcLim <= and(_goRxCrcLim_T_2, _goRxCrcLim_T_4) @[CanBsp.scala 205:66]
    canAcf.io_goRxInter <= io_goRxInter @[CanBsp.scala 535:23]
    canAcf.io_goErrorFrame <= io_goErrorFrame @[CanBsp.scala 536:26]
    canAcf.io_data0 <= tmpFifo.canAcf_io_data0_MPORT.data @[CanBsp.scala 537:19]
    canAcf.io_data1 <= tmpFifo.canAcf_io_data1_MPORT.data @[CanBsp.scala 538:19]
    canAcf.io_rtr1 <= rtr1 @[CanBsp.scala 539:18]
    canAcf.io_rtr2 <= rtr2 @[CanBsp.scala 540:18]
    canAcf.io_ide <= ide @[CanBsp.scala 541:17]
    canAcf.io_noByte0 <= or(rtr1, _noByte0_T) @[CanBsp.scala 238:29]
    canAcf.io_noByte1 <= or(rtr1, _noByte1_T) @[CanBsp.scala 239:29]
    canFifo.clock <= clock
    canFifo.reset <= reset
    canFifo.io_wr <= wrFifo @[CanBsp.scala 579:17]
    canFifo.io_dataIn <= mux(_dataForFifo_T_15, _dataForFifo_T_13, _dataForFifo_T_42) @[Lookup.scala 33:37]
    canFifo.io_addr <= bits(io_addr, 5, 0) @[CanBsp.scala 581:29]
    canFifo.io_resetMode <= io_resetMode @[CanBsp.scala 583:24]
    canFifo.io_releaseBuffer <= io_releaseBuffer @[CanBsp.scala 584:28]
    canFifo.io_extendedMode <= io_extendedMode @[CanBsp.scala 585:27]

  module CanTop :
    input clock : Clock
    input reset : UInt<1>
    input io_wbClkI : Clock
    input io_wbDatI : UInt<8>
    output io_wbDatO : UInt<8>
    input io_wbCycI : UInt<1>
    input io_wbStbI : UInt<1>
    input io_wbWeI : UInt<1>
    input io_wbAddrI : UInt<8>
    output io_wbAckO : UInt<1>
    input io_canRx : UInt<1>
    output io_canTx : UInt<1>
    output io_busOffOn : UInt<1>
    output io_irqOn : UInt<1>
    output io_clkout : UInt<1>

    inst canRegisters of CanRegisters @[CanTop.scala 101:43]
    inst canBtl of CanBtl @[CanTop.scala 160:31]
    inst canBsp of CanBsp @[CanTop.scala 189:31]
    reg addr : UInt<8>, clock with :
      reset => (UInt<1>("h0"), addr) @[CanTop.scala 95:28]
    reg dataOut : UInt<8>, clock with :
      reset => (UInt<1>("h0"), dataOut) @[CanTop.scala 97:27]
    reg rxSyncTmp : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxSyncTmp) @[CanTop.scala 98:33]
    reg rxSync : UInt<1>, clock with :
      reset => (UInt<1>("h0"), rxSync) @[CanTop.scala 99:30]
    skip
    skip
    node resetMode = canRegisters.io_resetMode @[CanTop.scala 127:13 24:30]
    node _dataOutFifoSelected_T = not(resetMode) @[CanTop.scala 260:52]
    node extendedMode = canRegisters.io_extendedMode @[CanTop.scala 157:16 55:33]
    node _dataOutFifoSelected_T_1 = and(extendedMode, _dataOutFifoSelected_T) @[CanTop.scala 260:49]
    node _dataOutFifoSelected_T_2 = geq(addr, UInt<8>("h10")) @[CanTop.scala 260:73]
    node _dataOutFifoSelected_T_3 = leq(addr, UInt<8>("h1c")) @[CanTop.scala 260:91]
    node _dataOutFifoSelected_T_4 = and(_dataOutFifoSelected_T_2, _dataOutFifoSelected_T_3) @[CanTop.scala 260:82]
    node _dataOutFifoSelected_T_5 = and(_dataOutFifoSelected_T_1, _dataOutFifoSelected_T_4) @[CanTop.scala 260:64]
    node _dataOutFifoSelected_T_6 = not(extendedMode) @[CanTop.scala 260:104]
    node _dataOutFifoSelected_T_7 = geq(addr, UInt<8>("h14")) @[CanTop.scala 260:128]
    node _dataOutFifoSelected_T_8 = leq(addr, UInt<8>("h1d")) @[CanTop.scala 260:146]
    node _dataOutFifoSelected_T_9 = and(_dataOutFifoSelected_T_7, _dataOutFifoSelected_T_8) @[CanTop.scala 260:137]
    node _dataOutFifoSelected_T_10 = and(_dataOutFifoSelected_T_6, _dataOutFifoSelected_T_9) @[CanTop.scala 260:119]
    node dataOutFifoSelected = or(_dataOutFifoSelected_T_5, _dataOutFifoSelected_T_10) @[CanTop.scala 260:101]
    skip
    node _T = not(io_wbWeI) @[CanTop.scala 262:13]
    reg cs_sync2 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), cs_sync2) @[CanTop.scala 267:31]
    reg cs_sync3 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), cs_sync3) @[CanTop.scala 268:31]
    node _cs_T = not(cs_sync3) @[CanTop.scala 277:21]
    node cs = and(cs_sync2, _cs_T) @[CanTop.scala 277:18]
    skip
    node _T_1 = and(cs, _T) @[CanTop.scala 262:11]
    node dataOutFifo = canBsp.io_dataOut @[CanTop.scala 197:15 21:32]
    node dataOutRegs = canRegisters.io_dataOut @[CanTop.scala 106:15 22:32]
    node _dataOut_T = mux(dataOutFifoSelected, dataOutFifo, dataOutRegs) @[CanTop.scala 263:21]
    skip
    reg cs_sync1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), cs_sync1) @[CanTop.scala 266:31]
    reg cs_ack1 : UInt<1>, io_wbClkI with :
      reset => (UInt<1>("h0"), cs_ack1) @[CanTop.scala 270:53]
    reg cs_ack2 : UInt<1>, io_wbClkI with :
      reset => (UInt<1>("h0"), cs_ack2) @[CanTop.scala 271:53]
    reg cs_ack3 : UInt<1>, io_wbClkI with :
      reset => (UInt<1>("h0"), cs_ack3) @[CanTop.scala 272:53]
    reg cs_sync_rst1 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), cs_sync_rst1) @[CanTop.scala 274:35]
    reg cs_sync_rst2 : UInt<1>, clock with :
      reset => (UInt<1>("h0"), cs_sync_rst2) @[CanTop.scala 275:35]
    node _cs_sync1_T = and(io_wbCycI, io_wbStbI) @[CanTop.scala 276:25]
    node _cs_sync1_T_1 = not(cs_sync_rst2) @[CanTop.scala 276:39]
    node _cs_sync1_T_2 = and(_cs_sync1_T, _cs_sync1_T_1) @[CanTop.scala 276:37]
    node _wbAckO_T = not(cs_ack3) @[CanTop.scala 280:24]
    node _wbAckO_T_1 = and(cs_ack2, _wbAckO_T) @[CanTop.scala 280:21]
    reg wbAckO : UInt<1>, io_wbClkI with :
      reset => (UInt<1>("h0"), wbAckO) @[CanTop.scala 280:12]
    node _GEN_1 = mux(io_wbStbI, io_wbAddrI, addr) @[CanTop.scala 285:19 286:10 95:28]
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    skip
    io_wbDatO <= dataOut @[CanTop.scala 289:13]
    io_wbAckO <= wbAckO @[CanTop.scala 283:13]
    io_canTx <= canBsp.io_tx @[CanTop.scala 244:12]
    io_busOffOn <= canBsp.io_busOffOn @[CanTop.scala 246:15]
    io_irqOn <= canRegisters.io_irqN @[CanTop.scala 107:12]
    io_clkout <= canRegisters.io_clkout @[CanTop.scala 158:13]
    addr <= mux(reset, UInt<8>("h0"), _GEN_1) @[CanTop.scala 95:{28,28}]
    dataOut <= mux(_T_1, _dataOut_T, dataOut) @[CanTop.scala 262:23 263:15 97:27]
    rxSyncTmp <= or(reset, io_canRx) @[CanTop.scala 98:{33,33,33}]
    rxSync <= or(reset, rxSyncTmp) @[CanTop.scala 99:{30,30,30}]
    canRegisters.clock <= clock
    canRegisters.reset <= reset
    canRegisters.io_cs <= and(cs_sync2, _cs_T) @[CanTop.scala 277:18]
    canRegisters.io_writeEn <= io_wbWeI @[CanTop.scala 284:11 94:28]
    canRegisters.io_addr <= addr @[CanTop.scala 104:24]
    canRegisters.io_dataIn <= io_wbDatI @[CanTop.scala 288:10 96:27]
    canRegisters.io_samplePoint <= canBtl.io_samplePoint @[CanTop.scala 171:15 59:32]
    canRegisters.io_transmitting <= canBsp.io_transmitting @[CanTop.scala 252:16 66:33]
    canRegisters.io_setResetMode <= canBsp.io_setResetMode @[CanTop.scala 227:16 71:33]
    canRegisters.io_nodeBusOff <= canBsp.io_nodeBusOff @[CanTop.scala 228:14 72:31]
    canRegisters.io_errorStatus <= canBsp.io_errorStatus @[CanTop.scala 229:15 73:31]
    canRegisters.io_rxErrorCount <= bits(canBsp.io_rxErrorCount, 7, 0) @[CanTop.scala 230:41]
    canRegisters.io_txErrorCount <= bits(canBsp.io_txErrorCount, 7, 0) @[CanTop.scala 231:41]
    canRegisters.io_transmitStatus <= canBsp.io_transmitStatus @[CanTop.scala 232:18 76:35]
    canRegisters.io_receiveStatus <= canBsp.io_receiveStatus @[CanTop.scala 233:17 77:34]
    canRegisters.io_txSuccessful <= canBsp.io_txSuccessful @[CanTop.scala 253:16 78:33]
    canRegisters.io_needToTx <= canBsp.io_needToTx @[CanTop.scala 235:12 79:29]
    canRegisters.io_overrun <= canBsp.io_overrun @[CanTop.scala 236:11 80:28]
    canRegisters.io_infoEmpty <= canBsp.io_infoEmpty @[CanTop.scala 237:13 81:30]
    canRegisters.io_setBusErrorIrq <= canBsp.io_setBusErrorIrq @[CanTop.scala 238:18 82:35]
    canRegisters.io_setArbitrationLostIrq <= canBsp.io_setArbitrationLostIrq @[CanTop.scala 239:25 83:42]
    canRegisters.io_arbitrationLostCapture <= canBsp.io_arbitrationLostCapture @[CanTop.scala 240:26 84:43]
    canRegisters.io_nodeErrorPassive <= canBsp.io_nodeErrorPassive @[CanTop.scala 241:20 85:37]
    canRegisters.io_nodeErrorActive <= canBsp.io_nodeErrorActive @[CanTop.scala 242:19 86:36]
    canRegisters.io_rxMessageCounter <= canBsp.io_rxMessageCounter @[CanTop.scala 243:20 87:37]
    canRegisters.io_txState <= canBsp.io_txState @[CanTop.scala 207:11 34:28]
    canRegisters.io_txStateQ <= canBsp.io_txStateQ @[CanTop.scala 208:12 35:29]
    canRegisters.io_overloadFrame <= canBsp.io_overloadFrame @[CanTop.scala 210:17 36:34]
    canRegisters.io_errorCaptureCode <= canBsp.io_errorCaptureCode @[CanTop.scala 215:20 41:37]
    canBtl.clock <= clock
    canBtl.reset <= reset
    canBtl.io_rx <= rxSync @[CanTop.scala 161:16]
    canBtl.io_tx <= io_canTx @[CanTop.scala 162:16]
    canBtl.io_baudRatePrescaler <= canRegisters.io_baudRatePrescaler @[CanTop.scala 146:21 43:38]
    canBtl.io_syncJumpWidth <= canRegisters.io_syncJumpWidth @[CanTop.scala 147:17 44:34]
    canBtl.io_timeSegment1 <= canRegisters.io_timeSegment1 @[CanTop.scala 149:16 45:33]
    canBtl.io_timeSegment2 <= canRegisters.io_timeSegment2 @[CanTop.scala 150:16 46:33]
    canBtl.io_tripleSampling <= canRegisters.io_tripleSampling @[CanTop.scala 151:18 47:35]
    canBtl.io_rxIdle <= canBsp.io_rxIdle @[CanTop.scala 222:10 65:27]
    canBtl.io_rxInter <= canBsp.io_rxInter @[CanTop.scala 226:11 69:28]
    canBtl.io_transmitting <= canBsp.io_transmitting @[CanTop.scala 252:16 66:33]
    canBtl.io_transmitter <= canBsp.io_transmitter @[CanTop.scala 251:15 67:32]
    canBtl.io_goRxInter <= canBsp.io_goRxInter @[CanTop.scala 224:13 68:30]
    canBtl.io_txNext <= canBsp.io_txNext @[CanTop.scala 245:10 88:27]
    canBtl.io_goOverloadFrame <= canBsp.io_goOverloadFrame @[CanTop.scala 247:19 90:36]
    canBtl.io_goErrorFrame <= canBsp.io_goErrorFrame @[CanTop.scala 248:16 91:33]
    canBtl.io_goTx <= canBsp.io_goTx @[CanTop.scala 92:25 249:8]
    canBtl.io_sendAck <= canBsp.io_sendAck @[CanTop.scala 250:11 93:28]
    canBtl.io_nodeErrorPassive <= canBsp.io_nodeErrorPassive @[CanTop.scala 241:20 85:37]
    canBsp.clock <= clock
    canBsp.reset <= reset
    canBsp.io_samplePoint <= canBtl.io_samplePoint @[CanTop.scala 190:25]
    canBsp.io_sampledBit <= canBtl.io_sampledBit @[CanTop.scala 191:24]
    canBsp.io_sampledBitQ <= canBtl.io_sampledBitQ @[CanTop.scala 192:25]
    canBsp.io_txPoint <= canBtl.io_txPoint @[CanTop.scala 193:21]
    canBsp.io_hardSync <= canBtl.io_hardSync @[CanTop.scala 194:22]
    canBsp.io_addr <= addr @[CanTop.scala 195:18]
    canBsp.io_dataIn <= io_wbDatI @[CanTop.scala 288:10 96:27]
    canBsp.io_resetMode <= canRegisters.io_resetMode @[CanTop.scala 127:13 24:30]
    canBsp.io_listenOnlyMode <= canRegisters.io_listenOnlyMode @[CanTop.scala 128:18 25:35]
    canBsp.io_acceptanceFilterMode <= canRegisters.io_acceptanceFilterMode @[CanTop.scala 129:24 26:41]
    canBsp.io_extendedMode <= canRegisters.io_extendedMode @[CanTop.scala 221:26]
    canBsp.io_selfTestMode <= canRegisters.io_selfTestMode @[CanTop.scala 130:16 27:33]
    canBsp.io_releaseBuffer <= canRegisters.io_releaseBuffer @[CanTop.scala 202:27]
    canBsp.io_txRequest <= canRegisters.io_txRequest @[CanTop.scala 204:23]
    canBsp.io_abortTx <= canRegisters.io_abortTx @[CanTop.scala 203:21]
    canBsp.io_selfRxRequest <= canRegisters.io_selfRxRequest @[CanTop.scala 205:27]
    canBsp.io_singleShotTransmission <= canRegisters.io_singleShotTransmission @[CanTop.scala 206:36]
    skip
    canBsp.io_readArbitrationLostCaptureReg <= canRegisters.io_readArbitrationLostCaptureReg @[CanTop.scala 212:43]
    canBsp.io_readErrorCodeCaptureReg <= canRegisters.io_readErrorCodeCaptureReg @[CanTop.scala 214:37]
    canBsp.io_errorWarningLimit <= canRegisters.io_errorWarningLimit @[CanTop.scala 153:21 49:38]
    canBsp.io_writeEnReceiveErrorCounter <= canRegisters.io_writeEnReceiveErrorCounter @[CanTop.scala 218:40]
    canBsp.io_writeEnTransmitErrorCounter <= canRegisters.io_writeEnTransmitErrorCounter @[CanTop.scala 219:41]
    canBsp.io_acceptanceCode_0 <= canRegisters.io_acceptanceCode_0 @[CanTop.scala 255:28]
    canBsp.io_acceptanceCode_1 <= canRegisters.io_acceptanceCode_1 @[CanTop.scala 255:28]
    canBsp.io_acceptanceCode_2 <= canRegisters.io_acceptanceCode_2 @[CanTop.scala 255:28]
    canBsp.io_acceptanceCode_3 <= canRegisters.io_acceptanceCode_3 @[CanTop.scala 255:28]
    canBsp.io_acceptanceMask_0 <= canRegisters.io_acceptanceMask_0 @[CanTop.scala 256:28]
    canBsp.io_acceptanceMask_1 <= canRegisters.io_acceptanceMask_1 @[CanTop.scala 256:28]
    canBsp.io_acceptanceMask_2 <= canRegisters.io_acceptanceMask_2 @[CanTop.scala 256:28]
    canBsp.io_acceptanceMask_3 <= canRegisters.io_acceptanceMask_3 @[CanTop.scala 256:28]
    canBsp.io_txData_0 <= canRegisters.io_txData_0 @[CanTop.scala 258:20]
    canBsp.io_txData_1 <= canRegisters.io_txData_1 @[CanTop.scala 258:20]
    canBsp.io_txData_2 <= canRegisters.io_txData_2 @[CanTop.scala 258:20]
    canBsp.io_txData_3 <= canRegisters.io_txData_3 @[CanTop.scala 258:20]
    canBsp.io_txData_4 <= canRegisters.io_txData_4 @[CanTop.scala 258:20]
    canBsp.io_txData_5 <= canRegisters.io_txData_5 @[CanTop.scala 258:20]
    canBsp.io_txData_6 <= canRegisters.io_txData_6 @[CanTop.scala 258:20]
    canBsp.io_txData_7 <= canRegisters.io_txData_7 @[CanTop.scala 258:20]
    canBsp.io_txData_8 <= canRegisters.io_txData_8 @[CanTop.scala 258:20]
    canBsp.io_txData_9 <= canRegisters.io_txData_9 @[CanTop.scala 258:20]
    canBsp.io_txData_10 <= canRegisters.io_txData_10 @[CanTop.scala 258:20]
    canBsp.io_txData_11 <= canRegisters.io_txData_11 @[CanTop.scala 258:20]
    canBsp.io_txData_12 <= canRegisters.io_txData_12 @[CanTop.scala 258:20]
    cs_sync1 <= mux(reset, UInt<1>("h0"), _cs_sync1_T_2) @[CanTop.scala 266:{31,31} 276:12]
    cs_sync2 <= mux(reset, UInt<1>("h0"), cs_sync1) @[CanTop.scala 267:{31,31,31}]
    cs_sync3 <= mux(reset, UInt<1>("h0"), cs_sync2) @[CanTop.scala 268:{31,31,31}]
    cs_ack1 <= mux(reset, UInt<1>("h0"), cs_sync3) @[CanTop.scala 270:{53,53,53}]
    cs_ack2 <= mux(reset, UInt<1>("h0"), cs_ack1) @[CanTop.scala 271:{53,53,53}]
    cs_ack3 <= mux(reset, UInt<1>("h0"), cs_ack2) @[CanTop.scala 272:{53,53,53}]
    cs_sync_rst1 <= mux(reset, UInt<1>("h0"), cs_ack3) @[CanTop.scala 274:{35,35,35}]
    cs_sync_rst2 <= mux(reset, UInt<1>("h0"), cs_sync_rst1) @[CanTop.scala 275:{35,35,35}]
    wbAckO <= mux(reset, UInt<1>("h0"), _wbAckO_T_1) @[CanTop.scala 280:{12,12,12}]
