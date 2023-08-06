import json
import os
from pathlib import Path
import typing

__dir__ = Path(__file__).parent


class _XrtNode:

  def __init__(self, root, prefix: typing.List[str]):
    self._root: Xrt = root
    self._endpoint_prefix = prefix

  def supports_impl(self, impl_type: str) -> bool:
    return False

  def get_child(self, child_name: str):
    """When instantiating a child instance, get the backend node with which it
    is associated."""
    child_path = self._endpoint_prefix + [child_name]
    return _XrtNode(self._root, child_path)


class Xrt(_XrtNode):

  def __init__(self,
               xclbin: os.PathLike = None,
               kernel: str = None,
               chan_desc_path: os.PathLike = None,
               hw_emu: bool = False) -> None:
    if xclbin is None:
      xclbin_files = list(__dir__.glob("*.xclbin"))
      if len(xclbin_files) == 0:
        raise RuntimeError("Could not find FPGA image.")
      if len(xclbin_files) > 1:
        raise RuntimeError("Found multiple FPGA images.")
      xclbin = __dir__ / xclbin_files[0]
    if kernel is None:
      xclbin_fn = os.path.basename(xclbin)
      kernel = xclbin_fn.split('.')[0]
    super().__init__(self, [])

    if hw_emu:
      os.environ["XCL_EMULATION_MODE"] = "hw_emu"

    from .esiXrtPython import Accelerator
    self._acc = Accelerator(os.path.abspath(str(xclbin)), kernel)
