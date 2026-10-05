"""Labelled SYNTHETIC graph contracts with deliberately missing inputs; no model training."""
from graph_core.build import GraphBuilder, ModuleBuilder


def diagnostic_graph(kind='nested'):
    def leaf(name):
        m = ModuleBuilder(name, description='SYNTHETIC teaching contract: constructor.b intentionally unconnected')
        x = m.in_('x', [None, 4])
        m.node('constructor', 'tensor.add')
        m.wire(x, 'constructor.a')
        m.out('y', 'constructor.output', [None, 4])
        return m.build()
    g = GraphBuilder()
    g.modules = [leaf('leaf')]
    g.input('input', ['N', 4])
    if kind == 'nested':
        middle = ModuleBuilder('middle', description='SYNTHETIC nested repeated definition')
        x = middle.in_('x', [None, 4])
        middle.node('loop', 'core.repeat', module='leaf', count=2, carry=[{'input': 'x', 'output': 'y'}])
        middle.wire(x, 'loop.x')
        middle.out('y', 'loop.y', [None, 4])
        g.modules.append(middle.build())
        g.instance('call', 'middle')
        g.wire('input', 'call.x')
    elif kind == 'repeat':
        g.node('loop', 'core.repeat', module='leaf', count=2, carry=[{'input': 'x', 'output': 'y'}])
        g.wire('input', 'loop.x')
    elif kind == 'select':
        g.modules.append(leaf('alternate'))
        g.node('flag', 'tensor.constant', value=True, dtype='bool', label='SYNTHETIC declared predicate; no learned value')
        g.node('pick', 'core.select', then={'module': 'leaf'}, otherwise={'module': 'alternate'})
        g.wire('flag', 'pick.pred')
        g.wire('input', 'pick.x')
    else:
        raise ValueError('Choose nested, repeat or select.')
    return g.build()
