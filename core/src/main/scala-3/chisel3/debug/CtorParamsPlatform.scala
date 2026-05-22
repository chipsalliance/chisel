// SPDX-License-Identifier: Apache-2.0

package chisel3.debug

import logger.LazyLogging

// Scala 3 has no runtime mirror to pick a primary out of multiple ctors,
// so emit params only for single-ctor classes.
private[debug] object CtorParamsPlatform extends LazyLogging {

  def ctorParams(obj: Any): Seq[(String, String)] = {
    val cls = obj.getClass
    val ctors = cls.getDeclaredConstructors
    if (ctors.length != 1) {
      if (ctors.length > 1)
        logger.warn(s"ctorParams: ${cls.getName} has ${ctors.length} constructors; omitting `params`")
      Seq.empty
    } else {
      val rawParams = ctors.head.getParameters.toSeq.filter(!_.getName.contains("$outer"))
      val namesSynthetic = rawParams.nonEmpty && rawParams.forall(!_.isNamePresent)
      val params = rawParams.map(p => (p.getName, simpleTypeName(p.getParameterizedType.getTypeName)))
      if (namesSynthetic)
        logger.warn(
          s"ctorParams: ${cls.getName} has only synthetic parameter names " +
            s"(${params.map(_._1).mkString(", ")}); emitted debug metadata will be of limited use. " +
            s"Compile user code with a flag that retains parameter names " +
            s"(e.g. javac `-parameters`) to recover real names."
        )
      params
    }
  }

  private def simpleTypeName(raw: String): String = {
    val noGeneric = raw.takeWhile(_ != '<')
    val afterDot = noGeneric.substring(noGeneric.lastIndexOf('.') + 1)
    val lastDollar = afterDot.lastIndexOf('$')
    val name = if (lastDollar >= 0) afterDot.substring(lastDollar + 1) else afterDot
    name.capitalize
  }
}
