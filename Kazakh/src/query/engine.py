class QueryEngine:

    def __init__(self, graph):
        self.graph = graph

    def diagnosed_with(self, patient, disease):
        for _, target, data in self.graph.edges(patient, data=True):
            if data["relation"] == "diagnosed_with" and target == disease:
                return "Yes"
        return "No"

    def first_drug(self, patient):
        drugs = []

        for _, target, data in self.graph.edges(patient, data=True):
            if data["relation"] == "treated_with":
                drugs.append((target, data["time"]))

        if not drugs:
            return None

        drugs.sort(key=lambda x: x[1])
        return drugs[0][0]