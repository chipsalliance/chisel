#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from ._om_ops_gen import *
from .._mlir_libs._circt._om import Evaluator as BaseEvaluator, Object as BaseObject, ClassType, ReferenceAttr, ListAttr

from ..ir import Attribute, Diagnostic, DiagnosticSeverity, Module, StringAttr
from ..support import attribute_to_var, var_to_attribute

import sys
import logging
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Sequence, TypeVar

if TYPE_CHECKING:
  from _typeshed.stdlib.dataclass import DataclassInstance


# Define the Object class by inheriting from the base implementation in C++.
class Object(BaseObject):

  def __init__(self, obj: BaseObject) -> None:
    super().__init__(obj)

  def __getattr__(self, name: str):
    # Call the base method to get a field.
    field = super().__getattr__(name)

    # For primitives, return a Python value.
    if isinstance(field, Attribute):
      return attribute_to_var(field)

    # For objects, return an Object, wrapping the base implementation.
    assert isinstance(field, BaseObject)
    return Object(field)

  # Support iterating over an Object by yielding its fields.
  def __iter__(self):
    for name in self.field_names:
      yield (name, getattr(self, name))


# Define the Evaluator class by inheriting from the base implementation in C++.
class Evaluator(BaseEvaluator):

  def __init__(self, mod: Module) -> None:
    """Instantiate an Evaluator with a Module."""

    # Call the base constructor.
    super().__init__(mod)

    # Set up logging for diagnostics.
    logging.basicConfig(
        format="[%(asctime)s] %(name)s (%(levelname)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )
    self._logger = logging.getLogger("Evaluator")

    # Attach our Diagnostic handler.
    mod.context.attach_diagnostic_handler(self._handle_diagnostic)

  def instantiate(self, cls: str, *args: Any) -> Object:
    """Instantiate an Object with a class name and actual parameters."""

    # Convert the class name and actual parameters to Attributes within the
    # Evaluator's context.
    with self.module.context:
      # Get the class name from the class name.
      class_name = StringAttr.get(cls)

      # Get the actual parameter Attributes from the supplied variadic
      # arguments. This relies on the circt.support helpers to convert from
      # Python objects to Attributes.
      actual_params = var_to_attribute(list(args))

    # Call the base instantiate method.
    obj = super().instantiate(class_name, actual_params)

    # Return the Object, wrapping the base implementation.
    return Object(obj)

  def _handle_diagnostic(self, diagnostic: Diagnostic) -> bool:
    """Handle MLIR Diagnostics by logging them."""

    # Log the diagnostic message at the appropriate level.
    if diagnostic.severity == DiagnosticSeverity.ERROR:
      self._logger.error(diagnostic.message)
    elif diagnostic.severity == DiagnosticSeverity.WARNING:
      self._logger.warning(diagnostic.message)
    else:
      self._logger.info(diagnostic.message)

    # Log any diagnostic notes at the info level.
    for note in diagnostic.notes:
      self._logger.info(str(note))

    # Flush the stdout stream to ensure logs appear when expected.
    sys.stdout.flush()

    # Return True, indicating this diagnostic has been fully handled.
    return True
