from email.mime import text
import re
from .schemas import Event

class ClinicalExtractor:

    def __init__(self):
        pass

    def extract(self, text, patient_id):
        events = []

        sentences = text.split(".")  # split into sentences

        for sent in sentences:
            sent = sent.strip().lower()

            if "diagnosed with" in sent:
                disease = self._extract_after(sent, "diagnosed with")
                time = self._extract_date(sent)

                events.append(Event(
                    patient=patient_id,
                    relation="diagnosed_with",
                    target=disease,
                    time=time,
                    source=sent
                ))

            if "treated with" in sent:
                drug = self._extract_after(sent, "treated with")
                time = self._extract_date(sent)

                events.append(Event(
                    patient=patient_id,
                    relation="treated_with",
                    target=drug,
                    time=time,
                    source=sent
                ))

        return events

    def _extract_after(self, text, phrase):
        try:
            val = text.split(phrase)[1].strip()
            val = val.split(" in ")[0]  # remove time
            val = val.split(",")[0]
            return val.strip()
        except:
            return "unknown"

    def _extract_date(self, text):
    # match YYYY-MM
        match = re.search(r"(20\d{2})-(\d{2})", text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"

        # match just year
        match = re.search(r"(20\d{2})", text)
        if match:
            return f"{match.group(1)}-01"

        return "9999-99"  # fallback (so sorting works)