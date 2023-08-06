#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from io import FileIO
import json
import pathlib
import re
import shutil
from typing import Dict, List

__dir__ = pathlib.Path(__file__).parent


def _camel_to_snake(camel: str):
  if camel.upper() == camel:
    return camel.lower()
  return re.sub(r'(?<!^)(?=[A-Z])', '_', camel).lower()


def _get_ports_for_clients(clients):
  # Assemble lists of clients for each service port.
  ports = {}
  for client in clients:
    port = client['port']['inner']
    if port not in ports:
      ports[port] = []
    ports[port].append(client)
  return ports


class SoftwareApiBuilder:
  """Parent class for all software API builders. Defines an interfaces and tries
  to encourage code sharing and API consistency (between languages)."""

  class Module:
    """Bookkeeping about modules."""

    def __init__(self, name: str):
      self.name = name
      self.instances: Dict[str, SoftwareApiBuilder.Module] = {}
      self.services: List[Dict] = []

  def __init__(self, services_json: str):
    """Read in the system descriptor and set up bookkeeping structures."""
    self.services = json.loads(services_json)
    self.types: Dict[str, Dict] = {}
    self.modules: Dict[str, SoftwareApiBuilder.Module] = {}

    # Get all the modules listed in the service hierarchy. Populate their
    # 'instances' properly.
    for top in self.services["top_levels"]:
      top_mod = self._get_module(top["module"][1:])
      for svc in top["services"]:
        parent: SoftwareApiBuilder.Module = top_mod
        for inner_ref in [
            (inst["outer_sym"], inst["inner"]) for inst in svc["instance_path"]
        ]:
          m = self._get_module(inner_ref[0])
          parent.instances[inner_ref[1]] = m
          parent = m

    # For any modules which have services, add them as appropriate.
    for mod in self.services["modules"]:
      m = self._get_module(mod["symbol"])
      for svc in mod["services"]:
        m.services.append(svc)

  def _get_module(self, mod_sym: str):
    """Get a module adding an entry if it doesn't exist."""
    if mod_sym not in self.modules:
      self.modules[mod_sym] = SoftwareApiBuilder.Module(mod_sym)
    return self.modules[mod_sym]

  def build(self, os: FileIO, tmpl_file: str):
    """Output the API (in a pre-determined order) via callbacks. Encourages some
    level of consistency between language APIs."""

    env = Environment(loader=FileSystemLoader(str(__dir__)),
                      undefined=StrictUndefined)
    env.globals.update(camel_to_snake=_camel_to_snake,
                       get_ports_for_clients=_get_ports_for_clients,
                       get_type_name=self.get_type_name,
                       type_str_of=self.get_str_type)

    template = env.get_template(tmpl_file)
    top_levels = [
        self._get_module(t["module"][1:]) for t in self.services["top_levels"]
    ]
    os.write(
        template.render(services=self.services,
                        modules=self.modules.values(),
                        types=self.types,
                        tops=top_levels))

  def get_type_name(self, type: Dict):
    """Create a name for 'type', record it, and return it."""
    if "capnp_name" in type:
      name = type["capnp_name"]
    else:
      name = "".join([c if c.isalnum() else '_' for c in type["mlir_name"]])
    self.types[name] = type
    return name

  def get_str_type(self, type: Dict):
    assert False, "unimplemented"


class PythonApiBuilder(SoftwareApiBuilder):

  def __init__(self, services_json: str):
    super().__init__(services_json)

  def build(self, system_name: str, sw_dir: pathlib.Path):
    """Emit a Python ESI runtime library into 'output_dir'."""
    libdir = sw_dir / system_name
    if not libdir.exists():
      libdir.mkdir()

    common_file = libdir / "common.py"
    shutil.copy(__dir__ / "esi_runtime_common.py", common_file)

    # Emit the system-specific API.
    main = libdir / "__init__.py"
    super().build(main.open("w"), "esi_api.py.j2")

  def get_str_type(self, type_dict: Dict):
    """Get a Python code string instantiating 'type'."""

    def py_type(type: Dict):
      dialect = type["dialect"]
      mn: str = type["mnemonic"]
      if dialect == "esi" and mn == "channel":
        return py_type(type["inner"])
      if dialect == "builtin":
        if mn.startswith("i") or mn.startswith("ui"):
          width = int(mn.strip("ui"))
          signed = False
        elif mn.startswith("si"):
          width = int(mn.strip("si"))
          signed = True
        if width == 0:
          return "VoidType()"
        return f"IntType({width}, {signed})"
      elif dialect == "hw":
        if mn == "struct":
          fields = [
              f"('{x['name']}', {py_type(x['type'])})" for x in type["fields"]
          ]
          fields_str = ", ".join(fields)
          return "StructType([" + fields_str + "])"

      assert False, "unimplemented type"

    return py_type(type_dict["type_desc"])
