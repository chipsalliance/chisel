// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import geny.Writable
import chisel3.{Data, VerificationStatement}
import chisel3.assert.Assert
import chisel3.assume.Assume
import chisel3.cover.Cover
import chisel3.internal.firrtl.ir._

@deprecated("There no CIRCTConverter anymore, use circtpanamaconverter directly", "Chisel 6.0")
abstract class CIRCTConverter {
  val mlirStream:         Writable
  val firrtlStream:       Writable
  val verilogStream:      Writable
  val mlirBytecodeStream: Writable
  def exportSplitVerilog(directory: os.Path): Unit

  def passManager(): CIRCTPassManager
  def om():          CIRCTOM

  def visitCircuit(name: String): Unit

  def visitDefBlackBox(defBlackBox: DefBlackBox): Unit

  def visitDefModule(defModule: DefModule): Unit

  def visitDefIntrinsicModule(defIntrinsicModule: DefIntrinsicModule): Unit

  def visitAttach(attach: Attach): Unit

  def visitConnect(connect: Connect): Unit

  def visitDefWire(defWire: DefWire): Unit

  def visitDefInvalid(defInvalid: DefInvalid): Unit

  def visitWhen(when: When): Unit

  def visitDefSeqMemory(defSeqMemory: DefSeqMemory): Unit

  def visitDefInstance(defInstance: DefInstance): Unit

  def visitDefMemPort[T <: Data](defMemPort: DefMemPort[T]): Unit

  def visitDefMemory(defMemory: DefMemory): Unit

  def visitDefPrim[T <: Data](defPrim: DefPrim[T]): Unit

  def visitDefReg(defReg: DefReg): Unit

  def visitDefRegInit(defRegInit: DefRegInit): Unit

  def visitPrintf(parent: Component, printf: Printf): Unit

  def visitStop(stop: Stop): Unit

  def visitVerification[T <: VerificationStatement](verifi: Verification[T], opName: String, args: Seq[Arg]): Unit

  def visitAssert(assert: Verification[Assert]): Unit

  def visitAssume(assume: Verification[Assume]): Unit

  def visitCover(cover: Verification[Cover]): Unit

  def visitProbeDefine(parent: Component, probeDefine: ProbeDefine): Unit

  def visitProbeForceInitial(parent: Component, probeForceInitial: ProbeForceInitial): Unit

  def visitProbeReleaseInitial(parent: Component, probeReleaseInitial: ProbeReleaseInitial): Unit

  def visitProbeForce(parent: Component, probeForce: ProbeForce): Unit

  def visitProbeRelease(parent: Component, probeRelease: ProbeRelease): Unit

  def visitPropAssign(parent: Component, propAssign: PropAssign): Unit
}
