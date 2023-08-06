from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Type

from . import esi
from .. import support

from .. import ir


class RequestToServerConnectionOp:

  @property
  def clientNamePath(self) -> List[str]:
    return [
        ir.StringAttr(x).value
        for x in ir.ArrayAttr(self.attributes["clientNamePath"])
    ]


class RequestToClientConnectionOp:

  @property
  def clientNamePath(self) -> List[str]:
    return [
        ir.StringAttr(x).value
        for x in ir.ArrayAttr(self.attributes["clientNamePath"])
    ]


class RandomAccessMemoryDeclOp:

  @property
  def innerType(self):
    return ir.TypeAttr(self.attributes["innerType"])


class ESIPureModuleOp:

  def add_entry_block(self):
    if len(self.body.blocks) > 0:
      raise IndexError('The module already has an entry block')
    self.body.blocks.append()
    return self.body.blocks[0]
