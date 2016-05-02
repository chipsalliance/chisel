package firrtl
package passes

// Datastructures
import scala.collection.mutable

import firrtl.Mappers.{ExpMap,StmtMap}
import firrtl.Utils.WithAs


// Tags an annotation to be consumed by this pass
case object InlineCAKind extends CircuitAnnotationKind

// Only use on legal Firrtl. Specifically, the restriction of
//  instance loops must have been checked, or else this pass can
//  infinitely recurse
object InlineInstances extends Transform {
   val inlineDelim = "$"
   def name = "Inline Instances"
   def execute(circuit: Circuit, annotations: Seq[CircuitAnnotation]): TransformResult = {
      annotations.count(_.kind == InlineCAKind) match {
         case 0 => TransformResult(circuit, None, None)
         case 1 => {
            // This could potentially be cleaned up, but the best solution is unclear at the moment.
            val myAnnotation = annotations.find(_.kind == InlineCAKind).get match {
               case x: StickyCircuitAnnotation => x
               case _ => throw new PassException("Circuit annotation must be StickyCircuitAnnotation")
            }
            check(circuit, myAnnotation)
            run(circuit, myAnnotation.getModuleNames, myAnnotation.getComponentNames)
         }
         // Default behavior is to error if more than one annotation for inlining
         //  This could potentially change
         case _ => throw new PassException("Found more than one circuit annotation of InlineCAKind!")
      }
   }

   // Checks the following properties:
   // 1) All annotated modules exist
   // 2) All annotated modules are InModules (can be inlined)
   // 3) All annotated instances exist, and their modules can be inline
   def check(c: Circuit, ca: StickyCircuitAnnotation): Unit = {
      val errors = mutable.ArrayBuffer[PassException]()
      val moduleMap = (for(m <- c.modules) yield m.name -> m).toMap
      val annModuleNames = ca.getModuleNames.map(_.name) ++ ca.getComponentNames.map(_.module.name)
      def checkExists(name: String): Unit =
         if (!moduleMap.contains(name))
            errors += new PassException(s"Annotated module does not exist: ${name}")
      def checkExternal(name: String): Unit = moduleMap(name) match {
            case m: ExModule => errors += new PassException(s"Annotated module cannot be an external module: ${name}")
            case _ => {}
         }
      def checkInstance(cn: ComponentName): Unit = {
         var containsCN = false
         def onStmt(name: String)(s: Stmt): Stmt = {
            s match {
               case WDefInstance(_, inst_name, module_name, tpe) =>
                  if (name == inst_name) {
                     containsCN = true
                     checkExternal(module_name)
                  }
               case _ => {}
            }
            s map onStmt(name)
         }
         onStmt(cn.name)(moduleMap(cn.module.name).asInstanceOf[InModule].body)
         if (!containsCN) errors += new PassException(s"Annotated instance does not exist: ${cn.module.name}.${cn.name}")
      }
      annModuleNames.foreach{n => checkExists(n)}
      if (!errors.isEmpty) throw new PassExceptions(errors)
      annModuleNames.foreach{n => checkExternal(n)}
      if (!errors.isEmpty) throw new PassExceptions(errors)
      ca.getComponentNames.foreach{cn => checkInstance(cn)}
      if (!errors.isEmpty) throw new PassExceptions(errors)
   }

   def run(c: Circuit, modsToInline: Seq[ModuleName], instsToInline: Seq[ComponentName]): TransformResult = {
      // ---- Rename functions/data ----
      val renameMap = mutable.HashMap[Named,Seq[Named]]()
      // Updates renameMap with new names
      def update(name: Named, rename: Named) = {
         val existing = renameMap.getOrElse(name, Seq[Named]())
         if (!existing.contains(rename)) renameMap(name) = existing.:+(rename)
      }

      // ---- Pass functions/data ----
      // Contains all unaltered modules
      val originalModules = mutable.HashMap[String,Module]()
      // Contains modules whose direct/indirect children modules have been inlined, and whose tagged instances have been inlined.
      val inlinedModules = mutable.HashMap[String,Module]()

      // Recursive.
      def onModule(m: Module): Module = {
         val inlinedInstances = mutable.ArrayBuffer[String]()
         // Recursive. Replaces inst.port with inst$port
         def onExp(e: Expression): Expression = e match {
            case WSubField(WRef(ref, _, _, _), field, tpe, gen) => {
               // Relies on instance declaration before any instance references
               if (inlinedInstances.contains(ref)) {
                  val newName = ref + inlineDelim + field
                  update(ComponentName(ref, ModuleName(m.name)), ComponentName(newName, ModuleName(m.name)))
                  WRef(newName, tpe, WireKind(), gen)
               }
               else e
            }
            case e => e map onExp
         }
         // Recursive. Inlines tagged instances
         def onStmt(s: Stmt): Stmt = s match {
               case WDefInstance(info, instName, moduleName, instTpe) => {
                  def rename(name:String): String = {
                     val newName = instName + inlineDelim + name
                     update(ComponentName(name, ModuleName(moduleName)), ComponentName(newName, ModuleName(m.name)))
                     newName
                  }
                  // Rewrites references in inlined statements from ref to inst$ref
                  def renameStmt(s: Stmt): Stmt = {
                     def renameExp(e: Expression): Expression = {
                        e map renameExp match {
                           case WRef(name, tpe, kind, gen) => WRef(rename(name), tpe, kind, gen)
                           case e => e
                        }
                     }
                     s map rename map renameStmt map renameExp
                  }
                  val shouldInline =
                     modsToInline.contains(ModuleName(moduleName)) ||
                        instsToInline.contains(ComponentName(instName, ModuleName(m.name)))
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
                        case m: ExModule => throw new PassException("Cannot inline external module")
                        case m: InModule => m
                     }
                     val stmts = mutable.ArrayBuffer[Stmt]()
                     for (p <- instInModule.ports) {
                        stmts += DefWire(p.info, rename(p.name), p.tpe)
                     }
                     stmts += renameStmt(instInModule.body)
                     Begin(stmts.toSeq)
                  } else s
               }
               case s => s map onExp map onStmt
            }
         m match {
            case InModule(info, name, ports, body) => {
               val mx = InModule(info, name, ports, onStmt(body))
               inlinedModules(name) = mx
               mx
            }
            case m: ExModule => {
               inlinedModules(m.name) = m
               m
            }
         }
      }

      c.modules.foreach{ m => originalModules(m.name) = m}
      val top = c.modules.find(m => m.name == c.main).get
      onModule(top)
      val modulesx = c.modules.map(m => inlinedModules(m.name))
      TransformResult(Circuit(c.info, modulesx, c.main), Some(BasicRenameMap(renameMap.toMap)), None)
   }
}
