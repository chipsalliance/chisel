# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

from circt.ir import ArrayAttr, Context, StringAttr
from circt.support import attribute_to_var

with Context():
  string_attr = StringAttr.get("foo")
  array_attr = ArrayAttr.get([string_attr])
  array = attribute_to_var(array_attr)
  # CHECK: ['foo']
  print(array)
