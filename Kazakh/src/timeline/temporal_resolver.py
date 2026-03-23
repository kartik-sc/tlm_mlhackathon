class TemporalResolver:

    def __init__(self):
        pass

    def fix_timeline(self, events):
        """
        Fix inconsistent timestamps
        """
        # Sort by time (basic version)
        events = sorted(events, key=lambda x: x.time)

        # Fix backward time (simple heuristic)
        for i in range(1, len(events)):
            if events[i].time < events[i-1].time:
                events[i].time = events[i-1].time

        return events