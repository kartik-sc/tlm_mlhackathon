import networkx as nx
class GraphBuilder:
  def __init__(self):
    self.graph = nx.DiGraph()

  def build(self,events):
    for e in events:
            self.graph.add_node(e.patient, type="Patient")
            self.graph.add_node(e.target)

            self.graph.add_edge(
                e.patient,
                e.target,
                relation=e.relation,
                time=e.time
            )

    return self.graph