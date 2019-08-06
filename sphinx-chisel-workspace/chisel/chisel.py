import docutils
from docutils import nodes
import sphinx
from docutils.parsers import rst
from docutils.parsers.rst import directives 
from sphinx.domains import Domain, Index
from sphinx.domains.std import StandardDomain
from sphinx.roles import XRefRole
from sphinx.directives import ObjectDescription
from sphinx.util.nodes import make_refnode
from sphinx import addnodes

class ChiselNode(ObjectDescription):
    """A custom node that describes a Chisel function."""

    required_arguments = 1

    option_spec = {
        'params': directives.unchanged_required
    }

    def handle_signature(self, sig, signode):
        signode += addnodes.desc_name(text=sig)
        return sig

    def add_target_and_index(self, name_cls, sig, signode):
        signode['ids'].append('attr-' + sig)
        if 'noindex' not in self.options:
            name = u"{}.{}.{}".format('chisel', type(self).__name__, sig)
            '''
            pmap = self.env.domaindata['chisel']['obj2param']
            if pmap != None:
                pmap[name] = list(self.options.get('params').split(' '))
            '''
            objs = self.env.domaindata['chisel']['objects']
            if objs != None:
                objs.append((name, sig, 'Chisel', self.env.docname, 'attr-' + sig, 0))
            else:
                self.env.domaindata['chisel']['objects'] = [(name, sig, 'Chisel', self.env.docname, 'attr-' + sig, 0)]

class ChiselDomain(Domain):

    name = 'chisel'
    label = 'ChiselDomain'

    roles = {
        'reref': XRefRole()
    }

    directives = {
        'attr': ChiselNode
    }

    indices = {
    }

    initial_data = {
        'objects': [], # list of objects
        'obj2param': {} # dict of name -> object dicts
    }

    def get_objects(self):
        for obj in self.data['objects']:
            yield obj

    def get_full_qualified_name(self, node):
        """Return full qualified name for a node."""
        return "{}.{}.{}".format('chisel', type(node).__name__, node.arguments[0])

    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        match = [(docname, anchor)
                 for name, sig, typ, docname, anchor, prio
                 in self.get_objects() if ("class " + target in sig or "def " + target in sig) and not ".scala" in target]
        if len(match) > 0:
            todocname = match[0][0]
            targ = match[0][1]
            return make_refnode(builder, fromdocname, todocname, targ, contnode, targ)
        else:
            return None 

def setup(app):
    app.add_domain(ChiselDomain)
    return {'version': '0.1'}
