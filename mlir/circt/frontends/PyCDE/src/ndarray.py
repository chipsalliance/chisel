#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .dialects import hw, sv
from .signals import BitVectorSignal, ArraySignal
from .types import BitVectorType, dim

from .circt import ir

import numpy as np
from functools import lru_cache
from dataclasses import dataclass
from typing import Union


@dataclass
class _TargetShape:
  """
  A small helper class for representing shapes which might be n-dimensional
  matrices (len(dims) > 0) or unary types (len(dims) == 0).
  """

  dims: list
  dtype: type

  @property
  def num_dims(self):
    return len(self.dims)

  @property
  def type(self):
    if len(self.dims) != 0:
      return dim(self.dtype, *self.dims)
    else:
      return self.dtype

  def __str__(self):
    return str(self.type)


class NDArray(np.ndarray):
  """
  A PyCDE Matrix serves as a Numpy view of a multidimensional CIRCT array
  (ArrayType).  The Matrix ensures that all assignments to itself have been
  properly converted to conform with insertion into the numpy array
  (circt_to_arr).  Once filled, a user can treat the Matrix as a numpy array.
  The underlying CIRCT array is not materialized until to_circt is called.
  """

  __slots__ = ["name", "pycde_dtype", "circt_output"]

  def __array_finalize__(self, obj):
    """
    Ensure Matrix-class slots are propagated upon copying.
    See https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_finalize__
    """
    if obj is not None:
      for slot in NDArray.__slots__:
        setattr(self, slot, getattr(obj, slot))

  def __new__(cls,
              shape: list = None,
              dtype=None,
              name: str = None,
              from_value=None) -> None:
    """
    Construct a matrix with the given shape and dtype.
    This is a __new__ function since np.ndarray does not have an __init__ function.

    Args:
        shape (list, optional): A tuple of integers representing the shape of the matrix.
        dtype (_type_, optional):
          the inner type of the matrix. This is a PyCDE type - the Numpy
          matrix contains 'object'-typed values.
        from_value (_type_, optional):
          A ListValue which this matrix should be initialized from.

    Raises:
        ValueError: _description_
        TypeError: _description_
    """

    from pycde.signals import ArraySignal

    if (from_value is not None) and (shape is not None or dtype is not None):
      raise ValueError(
          "Must specify either shape and dtype, or initialize from a value, but not both."
      )

    if from_value is not None:
      if isinstance(from_value, ArraySignal):
        shape = from_value.type.shape
        dtype = from_value.type.inner_type
        name = from_value.name
      elif isinstance(from_value, np.ndarray):
        shape = from_value.shape
        # Sample the first element to infer the type. This assumes that the
        # numpy array has already been filled.
        dtype = from_value.item(0).type
      else:
        raise TypeError(
            f"Cannot inititalize NDArray from value of type {type(from_value)}")

    # Initialize the underlying np.ndarray
    self = np.ndarray.__new__(cls, shape=shape, dtype=object)

    if name is None:
      name = "matrix"
    self.name = name
    self.pycde_dtype = dtype

    # The SSA value of this matrix after it has been materialized.
    # Once set, the matrix is immutable.
    self.circt_output = None

    if from_value is not None:
      if isinstance(from_value, ArraySignal):
        # PyCDE and numpy do not play nicely when doing np.arr(Matrix._circt_to_arr(...))
        # but individual assignments work.
        target_shape = self._target_shape_for_idxs(0)
        value_arr = NDArray._circt_to_arr(from_value, target_shape)
        for i, v in enumerate(value_arr):
          super().__setitem__(self, i, v)
      elif isinstance(from_value, np.ndarray):
        for i, v in enumerate(from_value):
          super().__setitem__(self, i, v)

    return self

  @lru_cache
  def _get_constant(self, value: int, width: int = 32):
    """ Get an IR constant backed by a constant cache."""
    return hw.ConstantOp(ir.IntegerType.get_signless(width), value)

  @staticmethod
  def _circt_to_arr(value: Union[BitVectorSignal, ArraySignal],
                    target_shape: _TargetShape):
    """Converts a CIRCT value into a numpy array."""
    from .signals import (BitVectorSignal, ArraySignal)

    if isinstance(value, BitVectorSignal) and isinstance(
        target_shape.dtype, BitVectorType):
      # Direct match on the target shape?
      if value.type == target_shape.type:
        return value

      # Is it feasible to extract values to the target shape?
      if target_shape.num_dims > 1:
        raise ValueError(
            f"Cannot extract BitVectorValue of type {value.type} to a multi-dimensional array of type {target_shape.type}."
        )

      if len(target_shape.dims) == 0:
        target_shape_bits = target_shape.dtype.width
      else:
        target_shape_bits = target_shape.dims[0] * target_shape.dtype.width

      if target_shape_bits != value.type.width:
        raise ValueError(
            f"Width mismatch between provided BitVectorValue ({value.type}) and target shape ({target_shape.type})."
        )

      # Extract to the target type
      n = len(value) / target_shape.dtype.width
      if n != int(n) or n < 1:
        raise ValueError("Bitvector must be a multiple of the provided dtype")
      n = int(n)
      slice_elem_width = int(len(value) / n)
      arr = []
      if n == 1:
        return value
      else:
        for i in range(n):
          startbit = i * slice_elem_width
          endbit = i * slice_elem_width + slice_elem_width
          arr.append(value[startbit:endbit])
    elif isinstance(value, ArraySignal):
      # Recursively convert the list.
      arr = []
      for i in range(value.type.size):
        # Pop the outer dimension of the target shape.
        inner_dims = target_shape.dims.copy()[1:]
        arr.append(
            NDArray._circt_to_arr(value[i],
                                  _TargetShape(inner_dims, target_shape.dtype)))
    elif isinstance(value, NDArray):
      # Check that the shape is compatible and that we have an identical dtype.
      if list(value.shape) != target_shape.dims:
        raise ValueError(
            f"Shape mismatch between provided NDArray ({value.shape}) and target shape ({target_shape.shape})."
        )
      if value.item(0).type != target_shape.dtype:
        raise ValueError(
            f"Dtype mismatch between provided NDArray ({value.item(0).type}) and target shape ({target_shape.dtype})."
        )
      # Compatible NDArray!
      return value
    else:
      raise ValueError(f"Cannot convert value {value} to numpy array.")

    return arr

  def _target_shape_for_idxs(self, idxs):
    """
    Get the TargetShape for the given indexing into the array.
    
    This function can be used for determining the type that right-hand side values
    to a given matrix assignment should have.
    """
    target_v = self[idxs]
    target_shape = _TargetShape([], self.pycde_dtype)
    if isinstance(target_v, np.ndarray):
      target_shape.dims = list(target_v.shape)
    return target_shape

  def __setitem__(self, np_access, value):
    if self.circt_output is not None:
      raise ValueError("Cannot assign to a materialized matrix.")

    # Todo: We should allow for 1 extra dimension in the access slice which is
    # not passed to numpy. This dimension would instead refer to an access
    # into the inner data type.
    # This access would allow for
    # - bitwise access for integer data types
    # - struct access for structs
    # This issue, however, is a more general one, since we don't currently
    # support lhs assignment of PyCDE Value's, which would be require for this.

    # Infer the target shape based on the access to the numpy array.
    # circt_to_arr will then try to convert the value to this shape.
    v = NDArray._circt_to_arr(value, self._target_shape_for_idxs(np_access))
    super().__setitem__(np_access, v)

  def check_is_fully_assigned(self):
    """ Checks that all sub-matrices have been fully assigned. """
    unassigned = np.argwhere(self == None)
    if len(unassigned) > 0:
      raise ValueError(f"Unassigned sub-matrices: \n{unassigned}")

  def assign_default_driver(self, value):
    """Assigns a default driver to any unassigned value in the matrix"""
    for arg in np.argwhere(self == None):
      super().__setitem__(tuple(arg), value)

  def to_circt(self, create_wire=True, dtype=None, default_driver=None):
    """
    Materializes this matrix to a ListValue through hw.array_create operations.
    
    if 'create_wire' is True, the matrix will be materialized to an sv.wire operation
    and the returned value will be a read-only reference to the wire.
    This wire acts as a barrier in CIRCT to prevent dataflow optimizations
    from reordering/optimizing the materialization of the matrix, which might
    reduce debugability.

    If default_driver is set, any unassigned value will be assigned to this. The
    default driver is expected to be of equal type as the dtype of this matrix.
    """
    if self.circt_output:
      return self.circt_output

    if dtype == None:
      dtype = self.pycde_dtype

    if default_driver:
      if default_driver.type != self.pycde_dtype:
        raise ValueError(
            f"Default driver {default_driver} is not of type {dtype}.")
      self.assign_default_driver(default_driver)

    # Check that the entire matrix has been assigned. If not, an exception is
    # thrown.
    self.check_is_fully_assigned()

    def build_subarray(lstOrVal):
      from .signals import BitVectorSignal
      # Recursively converts this matrix into ListValues through hw.array_create
      # operations.
      if not isinstance(lstOrVal, BitVectorSignal):
        subarrays = [build_subarray(v) for v in lstOrVal]
        return hw.ArrayCreateOp(subarrays)
      return lstOrVal

    # Materialize the matrix. Flip the matrix before materialization to match
    # SystemVerilog ordering.

    self.circt_output = build_subarray(np.flip(self))

    if create_wire:
      wire = sv.WireOp(self.circt_output.type, self.name + "_wire")
      sv.AssignOp(wire, self.circt_output)
      self.circt_output = wire.read

    return self.circt_output

  @staticmethod
  def to_ndarrays(lst):
    """
    Ensures that all ListValues in a lst have been converted to ndarrays.
    """
    ndarrays = []
    for l in lst:
      if isinstance(l, ArraySignal):
        ndarrays.append(NDArray(from_value=l))
      else:
        if not isinstance(l, np.ndarray):
          raise ValueError(f"Expected NDArray or ListValue, got {type(l)}")
        ndarrays.append(l)
    return ndarrays
