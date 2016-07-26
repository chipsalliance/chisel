/*
Copyright (c) 2014 - 2016 The Regents of the University of
California (Regents). All Rights Reserved.  Redistribution and use in
source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
   * Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer in the documentation and/or other materials
     provided with the distribution.
   * Neither the name of the Regents nor the names of its contributors
     may be used to endorse or promote products derived from this
     software without specific prior written permission.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.
*/

package firrtl

import com.typesafe.scalalogging.LazyLogging
import java.io.Writer
import firrtl.passes.Pass
import firrtl.ir.Circuit

// ===========================================
//              Utility Traits
// -------------------------------------------
// Valid if all passes in transformation:
//  1) Don't produce annotations
//  2) Don't consume annotations
//  3) No component or module names are renamed
trait SimpleRun extends LazyLogging {
   def run (circuit: Circuit, passes: Seq[Pass]): TransformResult = {
      val result = passes.foldLeft(circuit) {
         (c: Circuit, pass: Pass) => {
            val name = pass.name
            val x = Utils.time(name) { pass.run(c) }
            logger.debug(x.serialize)
            x
         }
      }
      TransformResult(result)
   }
}

// ===========================================
//             Lowering Transforms
// -------------------------------------------
// This transforms "CHIRRTL", the chisel3 IR, to "Firrtl". Note the resulting
//  circuit has only IR nodes, not WIR.
// TODO(izraelevitz): Create RenameMap from RemoveCHIRRTL
class Chisel3ToHighFirrtl () extends Transform with SimpleRun {
   val passSeq = Seq(
      passes.CheckChirrtl,
      passes.CInferTypes,
      passes.CInferMDir,
      passes.RemoveCHIRRTL)
   def execute (circuit: Circuit, annotations: Seq[CircuitAnnotation]): TransformResult =
      run(circuit, passSeq)
}

// Converts from the bare intermediate representation (ir.scala)
//  to a working representation (WIR.scala)
class IRToWorkingIR () extends Transform with SimpleRun {
   val passSeq = Seq(passes.ToWorkingIR)
   def execute (circuit: Circuit, annotations: Seq[CircuitAnnotation]): TransformResult =
      run(circuit, passSeq)
}

// Resolves types, kinds, and genders, and checks the circuit legality.
// Operates on working IR nodes and high Firrtl.
class ResolveAndCheck () extends Transform with SimpleRun {
   val passSeq = Seq(
      passes.CheckHighForm,
      passes.ResolveKinds,
      passes.InferTypes,
      passes.CheckTypes,
      passes.Uniquify,
      passes.ResolveKinds,
      passes.InferTypes,
      passes.ResolveGenders,
      passes.CheckGenders,
      passes.InferWidths,
      passes.CheckWidths)
   def execute (circuit: Circuit, annotations: Seq[CircuitAnnotation]): TransformResult =
      run(circuit, passSeq)
}

// Expands aggregate connects, removes dynamic accesses, and when
//  statements. Checks for uninitialized values. Must accept a
//  well-formed graph.
// Operates on working IR nodes.
class HighFirrtlToMiddleFirrtl () extends Transform with SimpleRun {
   val passSeq = Seq(
      passes.PullMuxes,
      passes.ExpandConnects,
      passes.RemoveAccesses,
      passes.ExpandWhens,
      passes.CheckInitialization,
      passes.ResolveKinds,
      passes.InferTypes,
      passes.ResolveGenders)
      //passes.InferWidths,
      //passes.CheckWidths)
   def execute (circuit: Circuit, annotations: Seq[CircuitAnnotation]): TransformResult =
      run(circuit, passSeq)
}

// Expands all aggregate types into many ground-typed components. Must
//  accept a well-formed graph of only middle Firrtl features.
// Operates on working IR nodes.
// TODO(izraelevitz): Create RenameMap from RemoveCHIRRTL
class MiddleFirrtlToLowFirrtl () extends Transform with SimpleRun {
   val passSeq = Seq(
      passes.Legalize,
      passes.LowerTypes,
      passes.ResolveKinds,
      passes.InferTypes,
      passes.ResolveGenders,
      passes.InferWidths)
   def execute (circuit: Circuit, annotations: Seq[CircuitAnnotation]): TransformResult =
      run(circuit, passSeq)
}

// Emits Verilog.
// First optimizes for verilog width semantics with custom Primops,
//  then splits complex expressions into temporary nodes. Finally,
//  renames names that conflict with Verilog keywords.
// Operates on working IR nodes.
// TODO(izraelevitz): Create RenameMap from VerilogRename
class EmitVerilogFromLowFirrtl (val writer: Writer) extends Transform with SimpleRun {
   val passSeq = Seq(
      passes.RemoveValidIf,
      passes.ConstProp,
      passes.PadWidths,
      passes.ConstProp,
      passes.VerilogWrap,
      passes.SplitExpressions,
      passes.CommonSubexpressionElimination,
      passes.DeadCodeElimination,
      passes.VerilogRename)
   def execute (circuit: Circuit, annotations: Seq[CircuitAnnotation]): TransformResult = {
      val result = run(circuit, passSeq)
      (new VerilogEmitter).run(result.circuit, writer)
      result
   }
}

// Emits Firrtl.
// Operates on WIR/IR nodes.
class EmitFirrtl (val writer: Writer) extends Transform {
   def execute (circuit: Circuit, annotations: Seq[CircuitAnnotation]): TransformResult = {
      FIRRTLEmitter.run(circuit, writer)
      TransformResult(circuit)
   }
}


// ===========================================
//             Lowering Compilers
// -------------------------------------------
// Emits input circuit
// Will replace Chirrtl constructs with Firrtl
class HighFirrtlCompiler extends Compiler {
   def transforms(writer: Writer): Seq[Transform] = Seq(
      new Chisel3ToHighFirrtl(),
      new IRToWorkingIR(),
      new EmitFirrtl(writer)
   )
}

// Emits lowered input circuit
class LowFirrtlCompiler extends Compiler {
   def transforms(writer: Writer): Seq[Transform] = Seq(
      new Chisel3ToHighFirrtl(),
      new IRToWorkingIR(),
      passes.InlineInstances,
      new ResolveAndCheck(),
      new HighFirrtlToMiddleFirrtl(),
      new MiddleFirrtlToLowFirrtl(),
      new EmitFirrtl(writer)
   )
}

// Emits Verilog
class VerilogCompiler extends Compiler {
   def transforms(writer: Writer): Seq[Transform] = Seq(
      new Chisel3ToHighFirrtl(),
      new IRToWorkingIR(),
      new ResolveAndCheck(),
      new HighFirrtlToMiddleFirrtl(),
      new MiddleFirrtlToLowFirrtl(),
      passes.InlineInstances,
      new EmitVerilogFromLowFirrtl(writer)
   )
}
