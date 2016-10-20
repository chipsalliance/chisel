package firrtl
package passes

import firrtl.ir._
import firrtl.Mappers._
import firrtl.Annotations._

// Datastructures
import scala.collection.mutable

// Tags an annotation to be consumed by this pass
case class InlineAnnotation(target: Named) extends Annotation with Loose with Unstable {
  def duplicate(n: Named) = this.copy(target=n)
  def transform = classOf[InlineInstances]
}

// Only use on legal Firrtl. Specifically, the restriction of
//  instance loops must have been checked, or else this pass can
//  infinitely recurse
class InlineInstances extends Transform {
   def inputForm = LowForm
   def outputForm = LowForm
   val inlineDelim = "$"
   override def name = "Inline Instances"

   private def collectAnns(anns: Iterable[Annotation]): (Set[ModuleName], Set[ComponentName]) =
     anns.foldLeft(Set.empty[ModuleName], Set.empty[ComponentName]) {
       case ((modNames, instNames), ann) => ann match {
         case InlineAnnotation(ModuleName(mod, cir)) => (modNames + ModuleName(mod, cir), instNames)
         case InlineAnnotation(ComponentName(com, mod)) => (modNames, instNames + ComponentName(com, mod))
         case _ => throw new PassException("Annotation must be InlineAnnotation")
       }
     }

   def execute(state: CircuitState): CircuitState = {
     // TODO Add error check for more than one annotation for inlining
     // TODO Propagate other annotations
     val result = for {
       myAnnotations <- getMyAnnotations(state)
       (modNames, instNames) = collectAnns(myAnnotations.values)
     } yield run(state.circuit, modNames, instNames)
     result getOrElse state // Return state if nothing to do
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

   def run(c: Circuit, modsToInline: Set[ModuleName], instsToInline: Set[ComponentName]): CircuitState = {
      // Check annotations and circuit match up
      check(c, modsToInline, instsToInline)

      // ---- Rename functions/data ----
      val renameMap = mutable.HashMap[Named,Seq[Named]]()
      // Updates renameMap with new names
      def update(name: Named, rename: Named) = {
         val existing = renameMap.getOrElse(name, Seq[Named]())
         if (!existing.contains(rename)) renameMap(name) = existing.:+(rename)
      }
      def set(name: Named, renames: Seq[Named]) = renameMap(name) = renames

      // ---- Pass functions/data ----
      // Contains all unaltered modules
      val originalModules = mutable.HashMap[String,DefModule]()
      // Contains modules whose direct/indirect children modules have been inlined, and whose tagged instances have been inlined.
      val inlinedModules = mutable.HashMap[String,DefModule]()
      val cname = CircuitName(c.main)

      // Recursive.
      def onModule(m: DefModule): DefModule = {
         val inlinedInstances = mutable.ArrayBuffer[String]()
         // Recursive. Replaces inst.port with inst$port
         def onExp(e: Expression): Expression = e match {
            case WSubField(WRef(ref, _, _, _), field, tpe, gen) =>
              // Relies on instance declaration before any instance references
              if (inlinedInstances.contains(ref)) {
                 val newName = ref + inlineDelim + field
                 set(ComponentName(ref, ModuleName(m.name, cname)), Seq.empty)
                 WRef(newName, tpe, WireKind, gen)
              }
              else e
            case ex => ex map onExp
         }
         // Recursive. Inlines tagged instances
         def onStmt(s: Statement): Statement = s match {
               case WDefInstance(info, instName, moduleName, instTpe) =>
                 def rename(name:String): String = {
                    val newName = instName + inlineDelim + name
                    update(ComponentName(name, ModuleName(moduleName, cname)), ComponentName(newName, ModuleName(m.name, cname)))
                    newName
                 }
                 // Rewrites references in inlined statements from ref to inst$ref
                 def renameStmt(s: Statement): Statement = {
                    def renameExp(e: Expression): Expression = {
                       e map renameExp match {
                          case WRef(name, tpe, kind, gen) => WRef(rename(name), tpe, kind, gen)
                          case ex => ex
                       }
                    }
                    s map rename map renameStmt map renameExp
                 }
                 val shouldInline =
                    modsToInline.contains(ModuleName(moduleName, cname)) ||
                       instsToInline.contains(ComponentName(instName, ModuleName(m.name, cname)))
                 // Used memoized instance if available
                 val instModule =
                    if (inlinedModules.contains(name)) inlinedModules(name)
                    else {
                       // Warning - can infinitely recurse if there is an instance loop
                       onModule(originalModules(moduleName))
                    }
                 if (shouldInline) {
                    inlinedInstances += instName
                    val instInModule = instModule match {
                       case m: ExtModule => throw new PassException("Cannot inline external module")
                       case m: Module => m
                    }
                    val stmts = mutable.ArrayBuffer[Statement]()
                    for (p <- instInModule.ports) {
                       stmts += DefWire(p.info, rename(p.name), p.tpe)
                    }
                    stmts += renameStmt(instInModule.body)
                    Block(stmts.toSeq)
                 } else s
               case sx => sx map onExp map onStmt
            }
         m match {
            case Module(info, name, ports, body) =>
              val mx = Module(info, name, ports, onStmt(body))
              inlinedModules(name) = mx
              mx
            case mx: ExtModule =>
              inlinedModules(mx.name) = mx
              mx
         }
      }

      c.modules.foreach{ m => originalModules(m.name) = m}
      val top = c.modules.find(m => m.name == c.main).get
      onModule(top)
      val modulesx = c.modules.map(m => inlinedModules(m.name))
      CircuitState(Circuit(c.info, modulesx, c.main), LowForm, None, Some(RenameMap(renameMap.toMap)))
   }
}
