"""Demo knowledge base: a small clinic. Replace with your own documents (see guide, step 10)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str
    modality: str = "text"   # text | image | audio (images/audio stored as caption/transcript)
    acl: str = "public"      # public | staff  -> enforced at retrieval time
    source: str = "kb"       # kb | web


KB = [
    Doc("d1", "Refund policy", "Refunds are issued within 14 days of purchase. Refund requests need the order number. Digital items are refundable only if unused."),
    Doc("d2", "Cancellation policy", "Appointments can be cancelled up to 24 hours before the start time at no charge. Late cancellations incur a 50 percent fee."),
    Doc("d3", "Opening hours", "The clinic is open Monday to Friday 9am to 6pm and Saturday 10am to 2pm. Closed on Sunday."),
    Doc("d4", "Delivery", "Orders ship within 2 business days. Standard delivery takes 3 to 5 days. Express delivery costs 9 dollars."),
    Doc("d5", "Dental checkups", "A routine checkup takes 30 minutes and is performed by Dr. Rivera. Checkups are recommended every six months."),
    Doc("d6", "Insurance", "We accept Aetna and Cigna. Dr. Rivera works at the Downtown branch. Insurance claims are filed within 7 days."),
    Doc("d7", "Staff payroll note", "Staff payroll runs on the 25th. Dr. Rivera salary band is confidential.", acl="staff"),
    Doc("i1", "Floor plan photo", "Photo of the Downtown branch floor plan: reception, waiting room, three treatment rooms, X-ray room next to room 3.", modality="image"),
    Doc("i2", "Price list scan", "Scanned price list: checkup 60 dollars, cleaning 90 dollars, X-ray 45 dollars, whitening 250 dollars.", modality="image"),
    Doc("a1", "Voicemail 0412", "Voicemail transcript: patient asks to move the Thursday appointment and says the parking garage entrance is on Pine Street.", modality="audio"),
]

WEB = [
    Doc("w1", "Whitening aftercare", "After whitening, avoid coffee and red wine for 48 hours.", source="web"),
    Doc("w2", "Root canal", "A root canal usually takes 60 to 90 minutes.", source="web"),
    Doc("w3", "Fluoride", "Fluoride treatment is usually recommended twice a year.", source="web"),
]

# Knowledge graph: entity -> trigger words, and (subject, relation, object) edges.
ENTITIES = {
    "Dr. Rivera": ["rivera"], "Downtown branch": ["downtown", "branch"], "Dental checkup": ["checkup"],
    "Aetna": ["aetna"], "Cigna": ["cigna"], "X-ray room": ["x-ray", "xray"], "Whitening": ["whitening"],
}
EDGES = [
    ("Dr. Rivera", "works at", "Downtown branch"), ("Dr. Rivera", "performs", "Dental checkup"),
    ("Dental checkup", "takes", "30 minutes"), ("Dental checkup", "costs", "60 dollars"),
    ("Downtown branch", "accepts", "Aetna"), ("Downtown branch", "accepts", "Cigna"),
    ("Downtown branch", "has", "X-ray room"), ("X-ray room", "costs", "45 dollars"),
    ("Whitening", "costs", "250 dollars"), ("Downtown branch", "is near", "Pine Street garage"),
]
RELATION_HINTS = {
    "accepts": r"insur|accept|plan", "costs": r"cost|price|much|dollar|pay", "takes": r"long|minute|time|take",
    "works at": r"where|branch|work", "performs": r"who|doctor|perform", "has": r"room|have|has|equip",
    "is near": r"near|park|garage|street",
}
QUERY_EXPANSIONS = [
    (r"money|back|return", "refund purchase"), (r"cancel|late", "cancellation fee hours"),
    (r"cost|price|much", "dollars price"), (r"open|hours|when", "monday friday open closed"),
    (r"insur", "aetna cigna claims"), (r"doctor|dentist", "dr rivera"),
]
