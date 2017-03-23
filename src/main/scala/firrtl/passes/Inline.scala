// See LICENSE for license details.

package firrtl
package passes

import firrtl.ir._
import firrtl.Mappers._
import firrtl.annotations._

// Datastructures
import scala.collection.mutable

// Tags an annotation to be consumed by this pass
object InlineAnnotation {
  def apply(target: Named): Annotation = Annotation(target, classOf[InlineInstances], "")

  def unapply(a: Annotation): Option[Named] = a match {
    case Annotation(named, t, _) if t == classOf[InlineInstances] => Some(named)
    case _ => None
  }
}

// Only use on legal Firrtl. Specifically, the restriction of
//  instance loops must have been checked, or else this pass can
//  infinitely recurse
class InlineInstances extends Transform {
   def inputForm = LowForm
   def outputForm = LowForm
   val inlineDelim = "$"

   private def collectAnns(circuit: Circuit, anns: Iterable[Annotation]): (Set[ModuleName], Set[ComponentName]) =
     anns.foldLeft(Set.empty[ModuleName], Set.empty[ComponentName]) {
       case ((modNames, instNames), ann) => ann match {
         case InlineAnnotation(CircuitName(c)) =>
           (circuit.modules.collect {
             case Module(_, name, _, _) if name != circuit.main => ModuleName(name, CircuitName(c))
           }.toSet, instNames)
         case InlineAnnotation(ModuleName(mod, cir)) => (modNames + ModuleName(mod, cir), instNames)
         case InlineAnnotation(ComponentName(com, mod)) => (modNames, instNames + ComponentName(com, mod))
         case _ => throw new PassException("Annotation must be InlineAnnotation")
       }
     }

   def execute(state: CircuitState): CircuitState = {
     // TODO Add error check for more than one annotation for inlining
     // TODO Propagate other annotations
     getMyAnnotations(state) match {
       case Nil => CircuitState(state.circuit, state.form)
       case myAnnotations =>
         val (modNames, instNames) = collectAnns(state.circuit, myAnnotations)
         run(state.circuit, modNames, instNames, state.annotations)
     }
   }

   // Checks the following properties:
   // 1) All annotated modules exist
   // 2) All annotated modules are InModules (can be inlined)
   // 3) All annotated instances exist, and their modules can be inline
   def check(c: Circuit, moduleNames: Set[ModuleName], instanceNames: Set[ComponentName]): Unit = {
      val errors = mutable.ArrayBuffer[PassException]()
      val moduleMap = (for(m <- c.modules) yield m.name -> m).toMap
      def checkExists(name: String): Unit =
         if (!moduleMap.contains(name))
            errors += new PassException(s"Annotated module does not exist: $name")
      def checkExternal(name: String): Unit = moduleMap(name) match {
            case m: ExtModule => errors += new PassException(s"Annotated module cannot be an external module: $name")
            case _ =>
      }
      def checkInstance(cn: ComponentName): Unit = {
         var containsCN = false
         def onStmt(name: String)(s: Statement): Statement = {
            s match {
               case WDefInstance(_, inst_name, module_name, tpe) =>
                  if (name == inst_name) {
                     containsCN = true
                     checkExternal(module_name)
                  }
               case _ =>
            }
            s map onStmt(name)
         }
         onStmt(cn.name)(moduleMap(cn.module.name).asInstanceOf[Module].body)
         if (!containsCN) errors += new PassException(s"Annotated instance does not exist: ${cn.module.name}.${cn.name}")
      }

      moduleNames.foreach{mn => checkExists(mn.name)}
      if (errors.nonEmpty) throw new PassExceptions(errors)
      moduleNames.foreach{mn => checkExternal(mn.name)}
      if (errors.nonEmpty) throw new PassExceptions(errors)
      instanceNames.foreach{cn => checkInstance(cn)}
      if (errors.nonEmpty) throw new PassExceptions(errors)
   }


  def run(c: Circuit, modsToInline: Set[ModuleName], instsToInline: Set[ComponentName], annos: Option[AnnotationMap]): CircuitState = {
    def getInstancesOf(c: Circuit, modules: Set[String]): Set[String] =
      c.modules.foldLeft(Set[String]()) { (set, d) =>
        d match {
          case e: ExtModule => set
          case m: Module =>
            val instances = mutable.HashSet[String]()
            def findInstances(s: Statement): Statement = s match {
              case WDefInstance(info, instName, moduleName, instTpe) if modules.contains(moduleName) =>
                instances += m.name + "." + instName
                s
              case sx => sx map findInstances
            }
            findInstances(m.body)
            instances.toSet ++ set
        }
      }

    // Check annotations and circuit match up
    check(c, modsToInline, instsToInline)
    val flatModules = modsToInline.map(m => m.name)
    val flatInstances = instsToInline.map(i => i.module.name + "." + i.name) ++ getInstancesOf(c, flatModules)
    val moduleMap = c.modules.foldLeft(Map[String, DefModule]()) { (map, m) => map + (m.name -> m) }

    def appendNamePrefix(prefix: String)(name:String): String = prefix + name
    def appendRefPrefix(prefix: String, currentModule: String)(e: Expression): Expression = e match {
      case WSubField(WRef(ref, _, InstanceKind, _), field, tpe, gen) if flatInstances.contains(currentModule + "." + ref) =>
        WRef(prefix + ref + inlineDelim + field, tpe, WireKind, gen)
      case WRef(name, tpe, kind, gen) => WRef(prefix + name, tpe, kind, gen)
      case ex => ex map appendRefPrefix(prefix, currentModule)
    }

    def onStmt(prefix: String, currentModule: String)(s: Statement): Statement = s match {
      case WDefInstance(info, instName, moduleName, instTpe) =>
        // Rewrites references in inlined statements from ref to inst$ref
        val shouldInline = flatInstances.contains(currentModule + "." + instName)
        // Used memoized instance if available
        if (shouldInline) {
          val toInline = moduleMap(moduleName) match {
            case m: ExtModule => throw new PassException("Cannot inline external module")
            case m: Module => m
          }
          val stmts = toInline.ports.map(p => DefWire(p.info, p.name, p.tpe)) :+ toInline.body
          onStmt(prefix + instName + inlineDelim, moduleName)(Block(stmts))
        } else WDefInstance(info, prefix + instName, moduleName, instTpe)
      case sx => sx map appendRefPrefix(prefix, currentModule) map onStmt(prefix, currentModule) map appendNamePrefix(prefix)
    }

    val flatCircuit = c.copy(modules = c.modules.flatMap { 
      case m if flatModules.contains(m.name) => None
      case m => 
        Some(m map onStmt("", m.name))
    })
    CircuitState(flatCircuit, LowForm, annos, None)
  }
}
