// SPDX-License-Identifier: Apache-2.0

package chisel3.debug

import scala.reflect.runtime.universe._

private[debug] object CtorParamsPlatform {

  def ctorParams(obj: Any): Seq[(String, String)] = {
    val ctor = typeOfInstance(obj).typeSymbol.asClass.primaryConstructor
    if (ctor == NoSymbol) Seq.empty
    else
      ctor.asMethod.paramLists.flatten
        .filter(!_.name.toString.contains("$outer"))
        .map(a => (a.name.toString.trim, a.info.typeSymbol.name.decodedName.toString.trim))
  }

  private def typeOfInstance(obj: Any): Type =
    runtimeMirror(obj.getClass.getClassLoader).classSymbol(obj.getClass).toType
}
