// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import scala.tools.nsc
import scala.tools.nsc.{Global, Phase}
import scala.tools.nsc.transform.TypingTransformers

private[plugin] trait ChiselOuterUtils { outerSelf: TypingTransformers =>
  import global._
  trait ChiselInnerUtils { innerSelf: outerSelf.TypingTransformer =>
    def inferType(t: Tree): Type = localTyper.typed(t, nsc.Mode.TYPEmode).tpe

    val baseModuleTpe:   Type = inferType(tq"chisel3.experimental.BaseModule")
    val stringTpe:       Type = inferType(tq"String")
    val bundleTpe:       Type = inferType(tq"chisel3.Bundle")
    val autoTypenameTpe: Type = inferType(tq"chisel3.experimental.HasAutoTypename")
    val recordTpe:       Type = inferType(tq"chisel3.Record")
    val dataTpe:         Type = inferType(tq"chisel3.Data")
    val ignoreSeqTpe:    Type = inferType(tq"chisel3.IgnoreSeqInBundle")
    val seqOfDataTpe:    Type = inferType(tq"scala.collection.Seq[chisel3.Data]")
    val someOfDataTpe:   Type = inferType(tq"scala.Option[chisel3.Data]")
    val itStringAnyTpe:  Type = inferType(tq"scala.collection.Iterable[(String,Any)]")
    val itAnyTpe:        Type = inferType(tq"scala.collection.Iterable[Any]")
    val sourceInfoTpe:   Type = inferType(tq"chisel3.experimental.SourceInfo")

    def stringFromTypeName(name: TypeName): String =
      name.toString.trim() // Remove trailing space (Scalac implementation detail)

    def isAModule(sym: Symbol): Boolean = { sym.tpe <:< baseModuleTpe }
    def isExactBaseModule(sym: Symbol): Boolean = { sym.tpe =:= baseModuleTpe }
    def isABundle(sym: Symbol): Boolean = { sym.tpe <:< bundleTpe }
    def isAutoTypenamed(sym: Symbol): Boolean = { sym.tpe <:< autoTypenameTpe }
    def isARecord(sym: Symbol): Boolean = { sym.tpe <:< recordTpe }
    def isIgnoreSeqInBundle(sym: Symbol): Boolean = { sym.tpe <:< ignoreSeqTpe }
    def isSeqOfData(sym: Symbol): Boolean = {
      val tpe = sym.tpe
      tpe match {
        case NullaryMethodType(resultType) =>
          resultType <:< seqOfDataTpe
        case _ =>
          false
      }
    }

    def isOptionOfData(symbol: Symbol): Boolean = {
      val tpe = symbol.tpe
      tpe match {
        case NullaryMethodType(resultType) =>
          resultType <:< someOfDataTpe
        case _ =>
          false
      }
    }
    def isExactBundle(sym: Symbol): Boolean = { sym.tpe =:= bundleTpe }

    // Cached because this is run on every argument to every Bundle
    val isDataCache = new collection.mutable.HashMap[Type, Boolean]
    def isData(sym: Symbol): Boolean = isDataCache.getOrElseUpdate(sym.tpe, sym.tpe <:< dataTpe)

    def isNullaryMethodNamed(name: String, defdef: DefDef): Boolean =
      defdef.name.decodedName.toString == name && defdef.tparams.isEmpty && defdef.vparamss.isEmpty

  }
}
