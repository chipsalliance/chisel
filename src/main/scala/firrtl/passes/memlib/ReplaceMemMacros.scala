// SPDX-License-Identifier: Apache-2.0

package firrtl.passes
package memlib

import firrtl.Mappers._
import firrtl.Utils._
import firrtl._
import firrtl.annotations._
import firrtl.ir._
import firrtl.passes.MemPortUtils.{MemPortMap, Modules}
import firrtl.passes.memlib.MemTransformUtils._
import firrtl.passes.wiring._
import firrtl.stage.Forms

import scala.collection.mutable.ListBuffer

/** Annotates the name of the pins to add for WiringTransform */
case class PinAnnotation(pins: Seq[String]) extends NoTargetAnnotation

object ReplaceMemMacros {
  class UnsupportedBlackboxMemoryException(msg: String) extends PassException(msg)
}

/** Replace DefAnnotatedMemory with memory blackbox + wrapper + conf file.
  * This will not generate wmask ports if not needed.
  * Creates the minimum # of black boxes needed by the design.
  */
class ReplaceMemMacros extends Transform with DependencyAPIMigration {
  override def prerequisites = Forms.MidForm
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Forms.MidEmitters
  override def invalidates(a: Transform) = false

  /** Return true if mask granularity is per bit, false if per byte or unspecified
    */
  private def getFillWMask(mem: DefAnnotatedMemory) = mem.maskGran match {
    case None    => false
    case Some(v) => v == 1
  }

  private def rPortToBundle(mem: DefAnnotatedMemory) = BundleType(
    defaultPortSeq(mem) :+ Field("data", Flip, mem.dataType)
  )
  private def rPortToFlattenBundle(mem: DefAnnotatedMemory) = BundleType(
    defaultPortSeq(mem) :+ Field("data", Flip, flattenType(mem.dataType))
  )

  /** Catch incorrect memory instantiations when there are masked memories with unsupported aggregate types.
    *
    * Example:
    *
    * val memory = SyncReadMem(N, Vec(M, new Bundle {
    *   val a = Bool()
    *   val b = UInt(3.W)
    * }))
    *
    * This memory wrapper will have created M*NUM_BUNDLE_ENTRIES bits or M*2 since createMask matches the
    * structure of the memory datatype. However, the MemConf output will have
    * a maskGran of 4b and thus M mask bits (since M*4b is the total mem. width and (M*4b)/4b = M).
    * Thus, when connecting the blackbox module created from the MemConf file and the FIRRTL wrapper,
    * there will be a mismatch in port size (M != M*2).
    */
  private def checkMaskDatatype(mem: DefAnnotatedMemory) = {
    mem.dataType match {
      case VectorType(at: AggregateType, _) =>
        val msg = s"${mem.info} : Cannot blackbox masked-write memory ${mem.name} with nested aggregate data type."
        throw new ReplaceMemMacros.UnsupportedBlackboxMemoryException(msg)
      case BundleType(_) =>
        val msg = s"${mem.info} : Cannot blackbox masked-write memory ${mem.name} with bundle data type."
        throw new ReplaceMemMacros.UnsupportedBlackboxMemoryException(msg)
      case _ =>
    }
  }

  private def wPortToBundle(mem: DefAnnotatedMemory) = BundleType(
    (defaultPortSeq(mem) :+ Field("data", Default, mem.dataType)) ++ (mem.maskGran match {
      case None => Nil
      case Some(_) => {
        checkMaskDatatype(mem)
        Seq(Field("mask", Default, createMask(mem.dataType)))
      }
    })
  )
  private def wPortToFlattenBundle(mem: DefAnnotatedMemory) = BundleType(
    (defaultPortSeq(mem) :+ Field("data", Default, flattenType(mem.dataType))) ++ (mem.maskGran match {
      case None                         => Nil
      case Some(_) if getFillWMask(mem) => Seq(Field("mask", Default, flattenType(mem.dataType)))
      case Some(_) => {
        checkMaskDatatype(mem)
        Seq(Field("mask", Default, flattenType(createMask(mem.dataType))))
      }
    })
  )
  // TODO(shunshou): Don't use createMask???

  private def rwPortToBundle(mem: DefAnnotatedMemory) = BundleType(
    defaultPortSeq(mem) ++ Seq(
      Field("wmode", Default, BoolType),
      Field("wdata", Default, mem.dataType),
      Field("rdata", Flip, mem.dataType)
    ) ++ (mem.maskGran match {
      case None => Nil
      case Some(_) => {
        checkMaskDatatype(mem)
        Seq(Field("wmask", Default, createMask(mem.dataType)))
      }
    })
  )
  private def rwPortToFlattenBundle(mem: DefAnnotatedMemory) = BundleType(
    defaultPortSeq(mem) ++ Seq(
      Field("wmode", Default, BoolType),
      Field("wdata", Default, flattenType(mem.dataType)),
      Field("rdata", Flip, flattenType(mem.dataType))
    ) ++ (mem.maskGran match {
      case None                           => Nil
      case Some(_) if (getFillWMask(mem)) => Seq(Field("wmask", Default, flattenType(mem.dataType)))
      case Some(_) => {
        checkMaskDatatype(mem)
        Seq(Field("wmask", Default, flattenType(createMask(mem.dataType))))
      }
    })
  )

  private def memToBundle(s: DefAnnotatedMemory) = BundleType(
    s.readers.map(Field(_, Flip, rPortToBundle(s))) ++
      s.writers.map(Field(_, Flip, wPortToBundle(s))) ++
      s.readwriters.map(Field(_, Flip, rwPortToBundle(s)))
  )

  private def memToFlattenBundle(s: DefAnnotatedMemory) = BundleType(
    s.readers.map(Field(_, Flip, rPortToFlattenBundle(s))) ++
      s.writers.map(Field(_, Flip, wPortToFlattenBundle(s))) ++
      s.readwriters.map(Field(_, Flip, rwPortToFlattenBundle(s)))
  )

  /** Creates a wrapper module and external module to replace a candidate memory
    *  The wrapper module has the same type as the memory it replaces
    *  The external module
    */
  private def createMemModule(
    m:                       DefAnnotatedMemory,
    wrapperName:             String,
    annotatedMemoriesBuffer: ListBuffer[DefAnnotatedMemory]
  ): Seq[DefModule] = {
    assert(m.dataType != UnknownType)
    val wrapperIoType = memToBundle(m)
    val wrapperIoPorts = wrapperIoType.fields.map(f => Port(NoInfo, f.name, Input, f.tpe))
    // Creates a type with the write/readwrite masks omitted if necessary
    val bbIoType = memToFlattenBundle(m)
    val bbIoPorts = bbIoType.fields.map(f => Port(NoInfo, f.name, Input, f.tpe))
    val bbRef = WRef(m.name, bbIoType)
    val hasMask = m.maskGran.isDefined
    val fillMask = getFillWMask(m)
    def portRef(p: String) = WRef(p, field_type(wrapperIoType, p))
    val stmts = Seq(WDefInstance(NoInfo, m.name, m.name, UnknownType)) ++
      (m.readers.flatMap(r => adaptReader(portRef(r), WSubField(bbRef, r)))) ++
      (m.writers.flatMap(w => adaptWriter(portRef(w), WSubField(bbRef, w), hasMask, fillMask))) ++
      (m.readwriters.flatMap(rw => adaptReadWriter(portRef(rw), WSubField(bbRef, rw), hasMask, fillMask)))
    val wrapper = Module(NoInfo, wrapperName, wrapperIoPorts, Block(stmts))
    val bb = ExtModule(NoInfo, m.name, bbIoPorts, m.name, Seq.empty)
    // TODO: Annotate? -- use actual annotation map

    // add to conf file
    annotatedMemoriesBuffer += m
    Seq(bb, wrapper)
  }

  // TODO(shunshou): get rid of copy pasta
  // Connects the clk, en, and addr fields from the wrapperPort to the bbPort
  private def defaultConnects(wrapperPort: WRef, bbPort: WSubField): Seq[Connect] =
    Seq("clk", "en", "addr").map(f => connectFields(bbPort, f, wrapperPort, f))

  // Generates mask bits (concatenates an aggregate to ground type)
  // depending on mask granularity (# bits = data width / mask granularity)
  private def maskBits(mask: WSubField, dataType: Type, fillMask: Boolean): Expression =
    if (fillMask) toBitMask(mask, dataType) else toBits(mask)

  private def adaptReader(wrapperPort: WRef, bbPort: WSubField): Seq[Statement] =
    defaultConnects(wrapperPort, bbPort) :+
      fromBits(WSubField(wrapperPort, "data"), WSubField(bbPort, "data"))

  private def adaptWriter(wrapperPort: WRef, bbPort: WSubField, hasMask: Boolean, fillMask: Boolean): Seq[Statement] = {
    val wrapperData = WSubField(wrapperPort, "data")
    val defaultSeq = defaultConnects(wrapperPort, bbPort) :+
      Connect(NoInfo, WSubField(bbPort, "data"), toBits(wrapperData))
    hasMask match {
      case false => defaultSeq
      case true =>
        defaultSeq :+ Connect(
          NoInfo,
          WSubField(bbPort, "mask"),
          maskBits(WSubField(wrapperPort, "mask"), wrapperData.tpe, fillMask)
        )
    }
  }

  private def adaptReadWriter(
    wrapperPort: WRef,
    bbPort:      WSubField,
    hasMask:     Boolean,
    fillMask:    Boolean
  ): Seq[Statement] = {
    val wrapperWData = WSubField(wrapperPort, "wdata")
    val defaultSeq = defaultConnects(wrapperPort, bbPort) ++ Seq(
      fromBits(WSubField(wrapperPort, "rdata"), WSubField(bbPort, "rdata")),
      connectFields(bbPort, "wmode", wrapperPort, "wmode"),
      Connect(NoInfo, WSubField(bbPort, "wdata"), toBits(wrapperWData))
    )
    hasMask match {
      case false => defaultSeq
      case true =>
        defaultSeq :+ Connect(
          NoInfo,
          WSubField(bbPort, "wmask"),
          maskBits(WSubField(wrapperPort, "wmask"), wrapperWData.tpe, fillMask)
        )
    }
  }

  /** Mapping from (module, memory name) pairs to blackbox names */
  private type NameMap = collection.mutable.HashMap[(String, String), String]

  /** Construct NameMap by assigning unique names for each memory blackbox */
  private def constructNameMap(namespace: Namespace, nameMap: NameMap, mname: String)(s: Statement): Statement = {
    s match {
      case m: DefAnnotatedMemory =>
        m.memRef match {
          case None    => nameMap(mname -> m.name) = namespace.newName(m.name)
          case Some(_) =>
        }
      case _ =>
    }
    s.map(constructNameMap(namespace, nameMap, mname))
  }

  private def updateMemStmts(
    namespace:               Namespace,
    nameMap:                 NameMap,
    mname:                   String,
    memPortMap:              MemPortMap,
    memMods:                 Modules,
    annotatedMemoriesBuffer: ListBuffer[DefAnnotatedMemory]
  )(s:                       Statement
  ): Statement = s match {
    case m: DefAnnotatedMemory =>
      if (m.maskGran.isEmpty) {
        m.writers.foreach { w => memPortMap(s"${m.name}.$w.mask") = EmptyExpression }
        m.readwriters.foreach { w => memPortMap(s"${m.name}.$w.wmask") = EmptyExpression }
      }
      m.memRef match {
        case None =>
          // prototype mem
          val newWrapperName = nameMap(mname -> m.name)
          val newMemBBName = namespace.newName(s"${newWrapperName}_ext")
          val newMem = m.copy(name = newMemBBName)
          memMods ++= createMemModule(newMem, newWrapperName, annotatedMemoriesBuffer)
          WDefInstance(m.info, m.name, newWrapperName, UnknownType)
        case Some((module, mem)) =>
          WDefInstance(m.info, m.name, nameMap(module -> mem), UnknownType)
      }
    case sx => sx.map(updateMemStmts(namespace, nameMap, mname, memPortMap, memMods, annotatedMemoriesBuffer))
  }

  private def updateMemMods(
    namespace:               Namespace,
    nameMap:                 NameMap,
    memMods:                 Modules,
    annotatedMemoriesBuffer: ListBuffer[DefAnnotatedMemory]
  )(m:                       DefModule
  ) = {
    val memPortMap = new MemPortMap

    (m.map(updateMemStmts(namespace, nameMap, m.name, memPortMap, memMods, annotatedMemoriesBuffer))
      .map(updateStmtRefs(memPortMap)))
  }

  def execute(state: CircuitState): CircuitState = {
    val annotatedMemoriesBuffer: collection.mutable.ListBuffer[DefAnnotatedMemory] = ListBuffer[DefAnnotatedMemory]()
    val c = state.circuit
    val namespace = Namespace(c)
    val memMods = new Modules
    val nameMap = new NameMap
    c.modules.map(m => m.map(constructNameMap(namespace, nameMap, m.name)))
    val modules = c.modules.map(updateMemMods(namespace, nameMap, memMods, annotatedMemoriesBuffer))
    state.copy(
      circuit = c.copy(modules = modules ++ memMods),
      annotations =
        state.annotations ++
          (state.annotations.collectFirst { case a: PinAnnotation => a } match {
            case None => Nil
            case Some(PinAnnotation(pins)) =>
              pins.foldLeft(Seq[Annotation]()) { (seq, pin) =>
                seq ++ memMods.collect {
                  case m: ExtModule => SinkAnnotation(ModuleName(m.name, CircuitName(c.main)), pin)
                }
              }
          }) :+
          AnnotatedMemoriesAnnotation(annotatedMemoriesBuffer.toList)
    )
  }
}
