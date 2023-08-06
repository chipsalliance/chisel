#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..signals import wrap_opviews_with_values
from ..circt.dialects import sv

wrap_opviews_with_values(sv, __name__)
