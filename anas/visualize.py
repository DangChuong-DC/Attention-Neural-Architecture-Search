import sys
import genotypes
from graphviz import Digraph


def plot(genotype, filename):
    g = Digraph(
        format='png',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("input_1", fillcolor='darkseagreen2')
    g.node("input_2", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2*i, 2*i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "input_1"
            elif j == 1:
                u = "input_2"
            else:
                u = str(j-2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="black", style="bold")

    g.node("mixed attention", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "mixed attention", fillcolor="purple", color="purple")

    g.node("mask", fillcolor='palegoldenrod')
    g.edge("mixed attention", "mask", label="sigmoid", fillcolor="black")

    g.render(filename, view=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)

    genotype_name = sys.argv[1]
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    plot(genotype.att, "attention_block_1")
