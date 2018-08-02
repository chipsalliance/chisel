package chisel3.libs.aspect

import firrtl.annotations.ModuleName
import firrtl.ir._
import firrtl.passes.Pass
import firrtl.passes.wiring.WiringInfo
import firrtl.{CircuitForm, CircuitState, HighForm, MidForm, WDefInstance, WRef, WSubAccess, WSubField, WSubIndex}
import firrtl.Mappers._

import scala.collection.mutable

private class AspectTransform extends firrtl.Transform {
  override def inputForm: CircuitForm = MidForm
  override def outputForm: CircuitForm = HighForm

  def onStmt(bps: Seq[AspectAnnotation])(s: Statement): Statement = {
    if(bps.isEmpty) s else {
      val newBody = mutable.ArrayBuffer[Statement]()
      newBody += s
      newBody ++= bps.map(_.instance)
      Block(newBody)
    }
  }

  override def execute(state: CircuitState): CircuitState = {
    val bps = state.annotations.collect{ case b: AspectAnnotation => b}
    val bpMap = bps.groupBy(_.enclosingModule.asInstanceOf[ModuleName].name)

    val moduleMap = state.circuit.modules.map(m => m.name -> m).toMap

    val newModules = state.circuit.modules.flatMap {
      case m: firrtl.ir.Module =>
        val myModuleAspects = bpMap.getOrElse(m.name, Nil)
        val newM = m mapStmt onStmt(myModuleAspects)
        newM +: myModuleAspects.map(_.module)
      case x: ExtModule => Seq(x)
    }

    val newCircuit = state.circuit.copy(modules = newModules)
    val newAnnotations = state.annotations.filter(!_.isInstanceOf[AspectAnnotation])

    val wiringInfos = bps.flatMap{ case bp@AspectAnnotation(refs, enclosingModule, instance, module) =>
      refs.map{ case (from, to) =>
        WiringInfo(from.getComponentName, Seq(to.getComponentName), from.getComponentName.name)
      }
    }


    val transforms = Seq(
      ToIR,
      new firrtl.ChirrtlToHighFirrtl,
      new firrtl.IRToWorkingIR,
      firrtl.passes.CheckHighForm,
      firrtl.passes.ResolveKinds,
      firrtl.passes.InferTypes,
      firrtl.passes.CheckTypes,
      firrtl.passes.Uniquify,
      firrtl.passes.ResolveKinds,
      firrtl.passes.InferTypes,
      firrtl.passes.ResolveGenders,
      firrtl.passes.CheckGenders,
      firrtl.passes.InferWidths,
      new firrtl.passes.wiring.Wiring(wiringInfos.toSeq),
      new firrtl.IRToWorkingIR,
      new firrtl.ResolveAndCheck
    )

    val finalState = transforms.foldLeft(state.copy(annotations = newAnnotations, circuit = newCircuit)) { (newState, xform) =>
      val x = xform.runTransform(newState)
      //println(s"===${xform.getClass.getSimpleName}===")
      //println(x.circuit.serialize)
      x
    }
    //println(finalState.circuit.serialize)
    finalState
  }
}

object ToIR extends Pass {
  def toExp(e: Expression): Expression = e map toExp match {
    case ex: WRef => Reference(ex.name, ex.tpe)
    case ex: WSubField => SubField(ex.expr, ex.name, ex.tpe)
    case ex: WSubIndex => SubIndex(ex.expr, ex.value, ex.tpe)
    case ex: WSubAccess => SubAccess(ex.expr, ex.index, ex.tpe)
    case ex => ex // This might look like a case to use case _ => e, DO NOT!
  }

  def toStmt(s: Statement): Statement = s map toExp match {
    case sx: WDefInstance => DefInstance(sx.info, sx.name, sx.module)
    case sx => sx map toStmt
  }

  def run (c:Circuit): Circuit =
    c copy (modules = c.modules map (_ map toStmt))
}
