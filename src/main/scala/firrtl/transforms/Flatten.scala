// See LICENSE for license details.

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.annotations._
import scala.collection.mutable
import firrtl.passes.{InlineInstances,PassException}

/** Tags an annotation to be consumed by this transform */
case class FlattenAnnotation(target: Named) extends SingleTargetAnnotation[Named] {
  def duplicate(n: Named) = FlattenAnnotation(n)
}

/**
 * Takes flatten annotations for module instances and modules and inline the entire hierarchy of modules down from the annotations.
 * This transformation instantiates and is based on the InlineInstances transformation.  
 * Note: Inlining a module means inlining all its children module instances      
 */
class Flatten extends Transform {
   def inputForm = LowForm
   def outputForm = LowForm

   val inlineTransform = new InlineInstances
   
   private def collectAnns(circuit: Circuit, anns: Iterable[Annotation]): (Set[ModuleName], Set[ComponentName]) =
     anns.foldLeft(Set.empty[ModuleName], Set.empty[ComponentName]) {
       case ((modNames, instNames), ann) => ann match {
         case FlattenAnnotation(CircuitName(c)) =>
           (circuit.modules.collect {
             case Module(_, name, _, _) if name != circuit.main => ModuleName(name, CircuitName(c))
           }.toSet, instNames)
         case FlattenAnnotation(ModuleName(mod, cir)) => (modNames + ModuleName(mod, cir), instNames)
         case FlattenAnnotation(ComponentName(com, mod)) => (modNames, instNames + ComponentName(com, mod))
         case _ => throw new PassException("Annotation must be a FlattenAnnotation")
       }
     }

   /**
    *  Modifies the circuit by replicating the hierarchy under the annotated objects (mods and insts) and
    *  by rewriting the original circuit to refer to the new modules that will be inlined later. 
    *  @return modified circuit and ModuleNames to inline
    */
   def duplicateSubCircuitsFromAnno(c: Circuit, mods: Set[ModuleName], insts: Set[ComponentName]): (Circuit, Set[ModuleName]) = {
     val modMap = c.modules.map(m => m.name->m) toMap
     val seedMods = mutable.Map.empty[String, String]
     val newModDefs = mutable.Set.empty[DefModule]
     val nsp = Namespace(c)

     /** 
      *  We start with rewriting DefInstances in the modules with annotations to refer to replicated modules to be created later.  
      *  It populates seedMods where we capture the mapping between the original module name of the instances came from annotation 
      *  to a new module name that we will create as a replica of the original one. 
      *  Note: We replace old modules with it replicas so that other instances of the same module can be left unchanged.
      */
     def rewriteMod(parent: DefModule)(x: Statement): Statement = x match {
       case _: Block => x map rewriteMod(parent)
       case WDefInstance(info, instName, moduleName, instTpe) =>
         if (insts.contains(ComponentName(instName, ModuleName(parent.name, CircuitName(c.main))))
           || mods.contains(ModuleName(parent.name, CircuitName(c.main)))) {
           val newModName = if (seedMods.contains(moduleName)) seedMods(moduleName) else nsp.newName(moduleName+"_TO_FLATTEN")
           seedMods += moduleName -> newModName
           WDefInstance(info, instName, newModName, instTpe)
         } else x
       case _ => x
     }
     
     val modifMods = c.modules map { m => m map rewriteMod(m) }
     
     /** 
      *  Recursively rewrites modules in the hierarchy starting with modules in seedMods (originally annotations). 
      *  Populates newModDefs, which are replicated modules used in the subcircuit that we create 
      *  by recursively traversing modules captured inside seedMods and replicating them
      */
     def recDupMods(mods: Map[String, String]): Unit = {
       val replMods = mutable.Map.empty[String, String]

       def dupMod(x: Statement): Statement = x match {
         case _: Block => x map dupMod
         case WDefInstance(info, instName, moduleName, instTpe) =>
           val newModName = if (replMods.contains(moduleName)) replMods(moduleName) else nsp.newName(moduleName+"_TO_FLATTEN")
           replMods += moduleName -> newModName
           WDefInstance(info, instName, newModName, instTpe)
         case _ => x 
       }
       
       def dupName(name: String): String = mods(name)
       val newMods = mods map { case (origName, newName) => modMap(origName) map dupMod map dupName }
       
       newModDefs ++= newMods
       
       if(replMods.size > 0) recDupMods(replMods.toMap)
       
     }
     recDupMods(seedMods.toMap)

     //convert newly created modules to ModuleName for inlining next (outside this function)
     val modsToInline = newModDefs map { m => ModuleName(m.name, CircuitName(c.main)) }
     (c.copy(modules = modifMods ++ newModDefs), modsToInline.toSet)
   }
   
   override def execute(state: CircuitState): CircuitState = {
     val annos = state.annotations.collect { case a @ FlattenAnnotation(_) => a }
     annos match {
       case Nil => CircuitState(state.circuit, state.form)
       case myAnnotations =>
         val c = state.circuit
         val (modNames, instNames) = collectAnns(state.circuit, myAnnotations)
         // take incoming annotation and produce annotations for InlineInstances, i.e. traverse circuit down to find all instances to inline
         val (newc, modsToInline) = duplicateSubCircuitsFromAnno(state.circuit, modNames, instNames)
         inlineTransform.run(newc, modsToInline.toSet, Set.empty[ComponentName], state.annotations)
     }
   }
}
