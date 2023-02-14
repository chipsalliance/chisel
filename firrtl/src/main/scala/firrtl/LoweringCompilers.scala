// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.transforms.IdentityTransform
import firrtl.stage.{Forms, TransformManager}

@deprecated("Use a TransformManager or some other Stage/Phase class. Will be removed in 1.4.", "FIRRTL 1.2")
sealed abstract class CoreTransform extends SeqTransform

/** Emits input circuit with no changes
  *
  * Primarily useful for changing between .fir and .pb serialized formats
  */
@deprecated(
  "Use stage.{FirrtlStage, FirrtlMain} or stage.transforms.Compiler(Seq(Dependency[ChirrtlEmitter]))",
  "FIRRTL 1.3"
)
class NoneCompiler extends Compiler {
  val emitter = new ChirrtlEmitter
  def transforms: Seq[Transform] = Seq(new IdentityTransform(ChirrtlForm))
}

/** Emits input circuit
  * Will replace Chirrtl constructs with Firrtl
  */
@deprecated(
  "Use stage.{FirrtlStage, FirrtlMain} stage.transforms.Compiler(Seq(Dependency[HighFirrtlEmitter]))",
  "FIRRTL 1.3"
)
class HighFirrtlCompiler extends Compiler {
  val emitter = new HighFirrtlEmitter
  def transforms: Seq[Transform] = Forms.HighForm.map(_.getObject())
}

/** Emits middle Firrtl input circuit */
@deprecated(
  "Use stage.{FirrtlStage, FirrtlMain} stage.transforms.Compiler(Dependency[MiddleFirrtlEmitter])",
  "FIRRTL 1.3"
)
class MiddleFirrtlCompiler extends Compiler {
  val emitter = new MiddleFirrtlEmitter
  def transforms: Seq[Transform] = Forms.MidForm.map(_.getObject())
}

/** Emits lowered input circuit */
@deprecated(
  "Use stage.{FirrtlStage, FirrtlMain} stage.transforms.Compiler(Dependency[LowFirrtlEmitter])",
  "FIRRTL 1.3"
)
class LowFirrtlCompiler extends Compiler {
  val emitter = new LowFirrtlEmitter
  def transforms: Seq[Transform] = Forms.LowForm.map(_.getObject())
}
