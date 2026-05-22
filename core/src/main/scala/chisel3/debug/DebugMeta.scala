// SPDX-License-Identifier: Apache-2.0

package chisel3.debug

import logger.LazyLogging

import chisel3._

import upickle.{default => json}

import scala.collection.mutable
import scala.language.existentials
import scala.util.Try
import scala.util.control.NonFatal

private[debug] case class ClassParam(
  name:     String,
  typeName: String,
  value:    Option[ujson.Value] = None
)

private[debug] object ClassParam {
  implicit val rw: json.ReadWriter[ClassParam] = {
    // Default upickle serializes Option as array; use value-or-null instead.
    implicit val optRw: json.ReadWriter[Option[ujson.Value]] = json
      .readwriter[ujson.Value]
      .bimap[Option[ujson.Value]](
        p => p.getOrElse(ujson.Null),
        j => Option.when(j != ujson.Null)(j)
      )
    json.macroRW
  }
}

private[debug] object CtorParamExtractor {
  private[debug] val MaxParamDepth = 8
  private[debug] val MaxRenderedLen = 256
  private[debug] val TruncatedSuffix = "...[truncated]"

  private[debug] def dataToTypeName(data: Data): String = sanitize(data match {
    case t: Record =>
      t.topBindingOpt match {
        case Some(binding) => s"${t._bindingToString(binding)}[${t.className}]"
        case None          => t.className
      }
    case t => t.toString.split(" ").last
  })

  private def sanitize(s: String): String =
    s.replaceAll("[\\p{Cntrl}\"\\\\]", "")

  private[debug] def getCtorParams(target: Any): Seq[ClassParam] =
    new CtorParamExtractor().getCtorParams(target)
}

private[debug] final class CtorParamExtractor extends LazyLogging {
  import CtorParamExtractor.{dataToTypeName, MaxParamDepth, MaxRenderedLen, TruncatedSuffix}

  private case class ClassDescriptor(
    params:    Seq[(String, String)],
    accessors: Map[String, java.lang.reflect.Method]
  )

  private val descriptorCache = mutable.HashMap.empty[Class[_], ClassDescriptor]
  // Identity-keyed; depth bounded by MaxParamDepth so linear scan beats hashing.
  // Reset at every getCtorParams entry; safe because elaboration is single-threaded.
  private val visited = mutable.ArrayBuffer.empty[AnyRef]

  private def descriptor(target: Any): ClassDescriptor = {
    val cls = target.getClass
    descriptorCache.getOrElseUpdate(cls, buildDescriptor(target, cls))
  }

  private def buildDescriptor(target: Any, cls: Class[_]): ClassDescriptor = {
    val params =
      try CtorParamsPlatform.ctorParams(target)
      catch {
        case NonFatal(e) =>
          logger.debug(s"ctorParams failed on ${cls.getName}: ${e.getMessage}")
          Seq.empty[(String, String)]
      }
    val accessors = params.iterator.flatMap { case (name, _) =>
      try {
        val m = cls.getDeclaredMethod(name)
        m.setAccessible(true)
        Some(name -> m)
      } catch { case NonFatal(_) => None }
    }.toMap
    ClassDescriptor(params, accessors)
  }

  private[debug] def getCtorParams(target: Any): Seq[ClassParam] = {
    visited.clear()
    target match { case ref: AnyRef => visited += ref; case _ => }
    getCtorParamsImpl(target, 0)
  }

  private def getCtorParamsImpl(target: Any, depth: Int): Seq[ClassParam] = {
    val d = descriptor(target)
    d.params.map { case (name, typeName) =>
      ClassParam(name, typeName, paramValue(target, d, name, typeName, depth))
    }
  }

  private def paramValue(
    obj:      Any,
    desc:     ClassDescriptor,
    name:     String,
    typeName: String,
    depth:    Int
  ): Option[ujson.Value] =
    desc.accessors.get(name).flatMap { method =>
      Try(method.invoke(obj.asInstanceOf[AnyRef])).fold(
        e => { logger.debug(s"paramValue: cannot reflect $name: ${e.getMessage}"); None },
        v => Some(renderValue(v, typeName, depth))
      )
    }

  private def renderValue(v: Any, typeName: String, depth: Int): ujson.Value = v match {
    case s: scala.collection.Seq[_] if s.exists(_.isInstanceOf[Data]) =>
      ujson.Str(s.collect { case d: Data => dataToTypeName(d) }.mkString("[", ", ", "]"))
    case d: Data    => ujson.Str(dataToTypeName(d))
    case b: Boolean => ujson.Bool(b)
    case _: Byte | _: Short | _: Int | _: Long | _: Float | _: Double => ujson.Str(v.toString)
    case null                                                         => ujson.Str("null")
    case ref: AnyRef =>
      if (depth >= MaxParamDepth || visited.exists(_ eq ref) || isOpaqueStdlibClass(ref.getClass))
        ujson.Str(capped(ref.toString))
      else {
        visited += ref
        val nested =
          try getCtorParamsImpl(ref, depth + 1)
          finally visited.dropRightInPlace(1)
        if (nested.exists(_.value.isDefined))
          ujson.Str(
            capped(
              s"$typeName(${nested.map(p => p.value.fold(p.name)(vv => s"${p.name}: ${renderJson(vv)}")).mkString(", ")})"
            )
          )
        else ujson.Str(capped(ref.toString))
      }
    case other => ujson.Str(capped(other.toString))
  }

  private def renderJson(v: ujson.Value): String = v match {
    case ujson.Str(s)  => s
    case ujson.Bool(b) => b.toString
    case other         => other.toString
  }

  private def capped(s: String): String =
    if (s.length <= MaxRenderedLen) s else s.substring(0, MaxRenderedLen) + TruncatedSuffix

  private def isOpaqueStdlibClass(cls: Class[_]): Boolean = {
    val name = cls.getName
    name.startsWith("java.") || name.startsWith("javax.") ||
    name.startsWith("sun.") || name.startsWith("scala.")
  }
}
