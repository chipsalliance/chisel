#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._esi_ops_gen import *
from .._mlir_libs._circt._esi import *


class ChannelSignaling:
  ValidReady: int = 0
  FIFO0: int = 1
