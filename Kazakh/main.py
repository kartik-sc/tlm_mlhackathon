from src.extraction.extractor import ClinicalExtractor
from src.timeline.temporal_resolver import TemporalResolver
from src.graph.builder import GraphBuilder
from src.query.engine import QueryEngine

def main():
    text = """
    Patient was diagnosed with lung cancer in 2022.
    In 2022-03, patient was treated with cisplatin.
    In 2022-06, patient was treated with paclitaxel.
    """

    patient_id = "patient_001"

    # Step 1: Extract
    extractor = ClinicalExtractor()
    events = extractor.extract(text, patient_id)

    # Step 2: Fix timeline
    resolver = TemporalResolver()
    events = resolver.fix_timeline(events)

    # Step 3: Build graph
    builder = GraphBuilder()
    graph = builder.build(events)

    # Step 4: Query
    engine = QueryEngine(graph)
    
    print(engine.diagnosed_with(patient_id, "lung cancer"))
    print(engine.first_drug(patient_id))


if __name__ == "__main__":
    main()