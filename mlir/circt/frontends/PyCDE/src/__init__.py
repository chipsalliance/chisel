#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .circt import ir
from . import circt
import atexit

# Push a default context onto the context stack at import time.
DefaultContext = ir.Context()
DefaultContext.__enter__()
circt.register_dialects(DefaultContext)
DefaultContext.allow_unregistered_dialects = True


@atexit.register
def __exit_ctxt():
  DefaultContext.__exit__(None, None, None)


from .common import (AppID, Clock, Reset, Input, InputChannel, Output,
                     OutputChannel)
from .module import (generator, modparams, Module)
from .system import (System)
from .types import (dim, types)
from .signals import (Signal)

# Until we get source location based on Python stack traces, default to unknown
# locations.
DefaultLocation = ir.Location.unknown()
DefaultLocation.__enter__()


@atexit.register
def __exit_loc():
  DefaultLocation.__exit__(None, None, None)
