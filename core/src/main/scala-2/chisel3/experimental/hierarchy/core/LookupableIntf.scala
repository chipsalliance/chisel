// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import chisel3._
import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.experimental.hierarchy.core.Lookupable.{Aux, Simple}

private[chisel3] trait Lookupable$Intf { self: Lookupable.type =>

  /** Factory method for creating Lookupable for user-defined types
    */
  def product1[X, T1: Lookupable](
    in:  X => T1,
    out: T1 => X
  )(
    implicit sourceInfo: SourceInfo
  ): Simple[X] = _product1Impl(in, out)

  /** Factory method for creating Lookupable for user-defined types
    */
  def product2[X, T1: Lookupable, T2: Lookupable](
    in:  X => (T1, T2),
    out: (T1, T2) => X
  )(
    implicit sourceInfo: SourceInfo
  ): Simple[X] = _product2Impl(in, out)

  /** Factory method for creating Lookupable for user-defined types
    */
  def product3[X, T1: Lookupable, T2: Lookupable, T3: Lookupable](
    in:  X => (T1, T2, T3),
    out: (T1, T2, T3) => X
  )(
    implicit sourceInfo: SourceInfo
  ): Simple[X] = _product3Impl(in, out)

  /** Factory method for creating Lookupable for user-defined types
    */
  def product4[X, T1: Lookupable, T2: Lookupable, T3: Lookupable, T4: Lookupable](
    in:  X => (T1, T2, T3, T4),
    out: (T1, T2, T3, T4) => X
  )(
    implicit sourceInfo: SourceInfo
  ): Simple[X] = _product4Impl(in, out)

  /** Factory method for creating Lookupable for user-defined types
    */
  def product5[X, T1: Lookupable, T2: Lookupable, T3: Lookupable, T4: Lookupable, T5: Lookupable](
    in:  X => (T1, T2, T3, T4, T5),
    out: (T1, T2, T3, T4, T5) => X
  )(
    implicit sourceInfo: SourceInfo
  ): Simple[X] = _product5Impl(in, out)

  implicit def lookupInstance[B <: BaseModule](implicit sourceInfo: SourceInfo): Simple[Instance[B]] =
    _lookupInstanceImpl[B]

  @deprecated("Looking up Modules is deprecated, please cast to Instance instead (.toInstance).", "Chisel 7.0.0")
  implicit def lookupModule[B <: BaseModule](implicit sourceInfo: SourceInfo): Aux[B, Instance[B]] =
    _lookupModuleImpl[B]

  implicit def lookupData[B <: Data](implicit sourceInfo: SourceInfo): Simple[B] =
    _lookupDataImpl[B]

  implicit def lookupMem[B <: MemBase[_]](implicit sourceInfo: SourceInfo): Simple[B] =
    _lookupMemImpl[B]

  implicit def lookupHasTarget(implicit sourceInfo: SourceInfo): Simple[HasTarget] =
    _lookupHasTargetImpl

  import scala.language.higherKinds
  implicit def lookupIterable[B, F[_] <: Iterable[_]](
    implicit sourceInfo: SourceInfo,
    lookupable:          Lookupable[B]
  ): Aux[F[B], F[lookupable.C]] = _lookupIterableImpl[B, F]

  implicit def lookupOption[B](
    implicit sourceInfo: SourceInfo,
    lookupable:          Lookupable[B]
  ): Aux[Option[B], Option[lookupable.C]] = _lookupOptionImpl[B]

  implicit def lookupEither[L, R](
    implicit sourceInfo: SourceInfo,
    lookupableL:         Lookupable[L],
    lookupableR:         Lookupable[R]
  ): Aux[Either[L, R], Either[lookupableL.C, lookupableR.C]] = _lookupEitherImpl[L, R]

  implicit def lookupTuple2[T1, T2](
    implicit sourceInfo: SourceInfo,
    lookupableT1:        Lookupable[T1],
    lookupableT2:        Lookupable[T2]
  ): Aux[(T1, T2), (lookupableT1.C, lookupableT2.C)] = _lookupTuple2Impl[T1, T2]

  implicit def lookupTuple3[T1, T2, T3](
    implicit sourceInfo: SourceInfo,
    lookupableT1:        Lookupable[T1],
    lookupableT2:        Lookupable[T2],
    lookupableT3:        Lookupable[T3]
  ): Aux[(T1, T2, T3), (lookupableT1.C, lookupableT2.C, lookupableT3.C)] = _lookupTuple3Impl[T1, T2, T3]

  implicit def lookupTuple4[T1, T2, T3, T4](
    implicit sourceInfo: SourceInfo,
    lookupableT1:        Lookupable[T1],
    lookupableT2:        Lookupable[T2],
    lookupableT3:        Lookupable[T3],
    lookupableT4:        Lookupable[T4]
  ): Aux[(T1, T2, T3, T4), (lookupableT1.C, lookupableT2.C, lookupableT3.C, lookupableT4.C)] =
    _lookupTuple4Impl[T1, T2, T3, T4]

  implicit def lookupTuple5[T1, T2, T3, T4, T5](
    implicit sourceInfo: SourceInfo,
    lookupableT1:        Lookupable[T1],
    lookupableT2:        Lookupable[T2],
    lookupableT3:        Lookupable[T3],
    lookupableT4:        Lookupable[T4],
    lookupableT5:        Lookupable[T5]
  ): Aux[(T1, T2, T3, T4, T5), (lookupableT1.C, lookupableT2.C, lookupableT3.C, lookupableT4.C, lookupableT5.C)] =
    _lookupTuple5Impl[T1, T2, T3, T4, T5]

  @deprecated(
    "Use of @instantiable on user-defined types is deprecated. Implement Lookupable for your type instead.",
    "Chisel 7.0.0"
  )
  implicit def lookupIsInstantiable[B <: IsInstantiable](
    implicit sourceInfo: SourceInfo
  ): Aux[B, Instance[B]] = _lookupIsInstantiableImpl[B]

  implicit def lookupIsLookupable[B <: IsLookupable](implicit sourceInfo: SourceInfo): Simple[B] =
    _lookupIsLookupableImpl[B]
}
