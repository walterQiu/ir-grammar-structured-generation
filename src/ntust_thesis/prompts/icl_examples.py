"""In-context examples for IR generation prompts."""

from __future__ import annotations


def build_one_stage_ir_in_context_examples(ir_grammar: str) -> str:
    """Return one-stage IR-generation ICL examples for the requested grammar."""
    if ir_grammar == "json":
        return _build_one_stage_json_icl_examples()
    if ir_grammar == "incremental_assignment_ir":
        return _build_one_stage_incremental_assignment_icl_examples()
    if ir_grammar == "code4struct_ir":
        return _build_one_stage_code4struct_icl_examples()
    msg = f"Unsupported one-stage IR grammar for ICL examples: {ir_grammar}"
    raise ValueError(msg)


def _build_one_stage_json_icl_examples() -> str:
    """Return one-stage JSON in-context examples."""
    return r"""
In-context Examples

Example 1

[Input]

Event type (reference): artifactexistence.damagedestroy.n/a
Allowed roles and multiplicities: damagerdestroyer=1, artifact=1, instrument=1, place=1
Sentence: damagerdestroyer is Protesters ; artifact is the offices of the holding company of Ukraine 's richest man , Rinat Akhmetov .

[Output]

{
  "damagerdestroyer": ["Protesters"],
  "artifact": ["the offices of the holding company of Ukraine 's richest man , Rinat Akhmetov"]
}

---

Example 2

[Input]

Event type (reference): contact.collaborate.correspondence
Allowed roles and multiplicities: participant=2, place=1
Sentence: participant is campaign .

[Output]

{
  "participant": ["campaign"]
}

---

Example 3

[Input]

Event type (reference): contact.discussion.n/a
Allowed roles and multiplicities: participant=2, place=1
Sentence: participant is Hillary ; participant is Alinsky .

[Output]

{
  "participant": ["Hillary", "Alinsky"]
}

---

Example 4

[Input]

Event type (reference): conflict.demonstrate.n/a
Allowed roles and multiplicities: demonstrator=1, place=1
Sentence: no valid role-span pairs are present .

[Output]

{}

"""


def _build_one_stage_incremental_assignment_icl_examples() -> str:
    """Return one-stage incremental-assignment in-context examples."""
    return r"""
In-context Examples

Example 1

[Input]

Event type (reference): artifactexistence.damagedestroy.n/a
Allowed roles and multiplicities: damagerdestroyer=1, artifact=1, instrument=1, place=1
Sentence: damagerdestroyer is Protesters ; artifact is the offices of the holding company of Ukraine 's richest man , Rinat Akhmetov .

[Output]

damagerdestroyer += Protesters
artifact += the offices of the holding company of Ukraine 's richest man , Rinat Akhmetov

---

Example 2

[Input]

Event type (reference): contact.collaborate.correspondence
Allowed roles and multiplicities: participant=2, place=1
Sentence: participant is campaign .

[Output]

participant += campaign

---

Example 3

[Input]

Event type (reference): contact.discussion.n/a
Allowed roles and multiplicities: participant=2, place=1
Sentence: participant is Hillary ; participant is Alinsky .

[Output]

participant += Hillary
participant += Alinsky

---

Example 4

[Input]

Event type (reference): conflict.demonstrate.n/a
Allowed roles and multiplicities: demonstrator=1, place=1
Sentence: no valid role-span pairs are present .

[Output]

"""


def _build_one_stage_code4struct_icl_examples() -> str:
    """Return one-stage CODE4STRUCT in-context examples."""
    return r'''
In-context Examples

Example 1

[Input]

from typing import List

class Entity:
    def __init__(self, name: str):
        self.name = name

class Event:
    def __init__(self, name: str):
        self.name = name

class ArtifactexistenceDamagedestroyNa(Event):
    def __init__(
        self,
        damagerdestroyer: List[Entity] = [],
        artifact: List[Entity] = [],
        instrument: List[Entity] = [],
        place: List[Entity] = [],
    ):
        self.damagerdestroyer = damagerdestroyer
        self.artifact = artifact
        self.instrument = instrument
        self.place = place

"""
Argument value limits: damagerdestroyer=1, artifact=1, instrument=1, place=1
Convert the following sentence into an instance of ArtifactexistenceDamagedestroyNa.
"damagerdestroyer is Protesters ; artifact is the offices of the holding company of Ukraine 's richest man , Rinat Akhmetov ."
"""
artifactexistencedamagedestroyna_event = ArtifactexistenceDamagedestroyNa(

[Output]

damagerdestroyer=[Entity("Protesters"),],
artifact=[Entity("the offices of the holding company of Ukraine 's richest man , Rinat Akhmetov"),],
)

---

Example 2

[Input]

from typing import List

class Entity:
    def __init__(self, name: str):
        self.name = name

class Event:
    def __init__(self, name: str):
        self.name = name

class ContactCollaborateCorrespondence(Event):
    def __init__(
        self,
        participant: List[Entity] = [],
        place: List[Entity] = [],
    ):
        self.participant = participant
        self.place = place

"""
Argument value limits: participant=2, place=1
Convert the following sentence into an instance of ContactCollaborateCorrespondence.
"participant is campaign ."
"""
contactcollaboratecorrespondence_event = ContactCollaborateCorrespondence(

[Output]

participant=[Entity("campaign"),],
)

---

Example 3

[Input]

from typing import List

class Entity:
    def __init__(self, name: str):
        self.name = name

class Event:
    def __init__(self, name: str):
        self.name = name

class ContactDiscussionNa(Event):
    def __init__(
        self,
        participant: List[Entity] = [],
        place: List[Entity] = [],
    ):
        self.participant = participant
        self.place = place

"""
Argument value limits: participant=2, place=1
Convert the following sentence into an instance of ContactDiscussionNa.
"participant is Hillary ; participant is Alinsky ."
"""
contactdiscussionna_event = ContactDiscussionNa(

[Output]

participant=[
    Entity("Hillary"),
    Entity("Alinsky"),
],
)

---

Example 4

[Input]

from typing import List

class Entity:
    def __init__(self, name: str):
        self.name = name

class Event:
    def __init__(self, name: str):
        self.name = name

class ConflictDemonstrateNa(Event):
    def __init__(
        self,
        demonstrator: List[Entity] = [],
        place: List[Entity] = [],
    ):
        self.demonstrator = demonstrator
        self.place = place

"""
Argument value limits: demonstrator=1, place=1
Convert the following sentence into an instance of ConflictDemonstrateNa.
"no valid role-span pairs are present ."
"""
conflictdemonstratena_event = ConflictDemonstrateNa(

[Output]

)

'''


def build_two_stage_ir_in_context_examples(ir_grammar: str) -> str:
    """Return two-stage IR-generation ICL examples for the requested grammar."""
    if ir_grammar == "json":
        return _build_two_stage_json_icl_examples()
    if ir_grammar == "incremental_assignment_ir":
        return _build_two_stage_incremental_assignment_icl_examples()
    if ir_grammar == "code4struct_ir":
        return _build_two_stage_code4struct_icl_examples()
    msg = f"Unsupported IR grammar for ICL examples: {ir_grammar}"
    raise ValueError(msg)


def _build_two_stage_json_icl_examples() -> str:
    """Return two-stage JSON-generation in-context learning examples."""
    return r"""
In-context Examples

Example 1

[Input]

Allowed roles and multiplicities: damagerdestroyer=1, artifact=1, instrument=1, place=1
Extraction notes:
In the sentence, "Protesters" serves as the damagerdestroyer, and "the offices of the holding company of Ukraine 's richest man , Rinat Akhmetov" is the artifact that was vandalized.

[Output]

{
  "damagerdestroyer": ["Protesters"],
  "artifact": ["the offices of the holding company of Ukraine 's richest man , Rinat Akhmetov"]
}

---

Example 2

[Input]

Allowed roles and multiplicities: participant=2, place=1
Extraction notes:
The sentence indicates that the "campaign" is a participant in the correspondence.

[Output]

{
  "participant": ["campaign"]
}

---

Example 3

[Input]

Allowed roles and multiplicities: participant=2, place=1
Extraction notes:
In the sentence, the participants involved in the meeting are Hillary and Alinsky.

[Output]

{
  "participant": ["Hillary", "Alinsky"]
}

---

Example 4

[Input]

Allowed roles and multiplicities: demonstrator=1, place=1
Extraction notes:
The sentence does not explicitly provide any information about the demonstrator or the place related to the protest. Therefore, there are no arguments to identify for the allowed roles.

[Output]

{}

"""


def _build_two_stage_incremental_assignment_icl_examples() -> str:
    """Return two-stage incremental-assignment IR in-context examples."""
    return r"""
In-context Examples

Example 1

[Input]

Allowed roles and multiplicities: damagerdestroyer=1, artifact=1, instrument=1, place=1
Extraction notes:
In the sentence, "Protesters" serves as the damagerdestroyer, and "the offices of the holding company of Ukraine 's richest man , Rinat Akhmetov" is the artifact that was vandalized.

[Output]

damagerdestroyer += Protesters
artifact += the offices of the holding company of Ukraine 's richest man , Rinat Akhmetov

---

Example 2

[Input]

Allowed roles and multiplicities: participant=2, place=1
Extraction notes:
The sentence indicates that the "campaign" is a participant in the correspondence.

[Output]

participant += campaign

---

Example 3

[Input]

Allowed roles and multiplicities: participant=2, place=1
Extraction notes:
In the sentence, the participants involved in the meeting are Hillary and Alinsky.

[Output]

participant += Hillary
participant += Alinsky

---

Example 4

[Input]

Allowed roles and multiplicities: demonstrator=1, place=1
Extraction notes:
The sentence does not explicitly provide any information about the demonstrator or the place related to the protest. Therefore, there are no arguments to identify for the allowed roles.

[Output]

"""


def _build_two_stage_code4struct_icl_examples() -> str:
    """Return two-stage CODE4STRUCT in-context learning examples."""
    return r'''
In-context Examples

Example 1

[Input]

from typing import List

class Entity:
    def __init__(self, name: str):
        self.name = name

class Event:
    def __init__(self, name: str):
        self.name = name

class ConflictAttackDamage(Event):
    def __init__(
        self,
        damagerdestroyer: List[Entity] = [],
        artifact: List[Entity] = [],
        instrument: List[Entity] = [],
        place: List[Entity] = [],
    ):
        self.damagerdestroyer = damagerdestroyer
        self.artifact = artifact
        self.instrument = instrument
        self.place = place

"""
Argument value limits: damagerdestroyer=1, artifact=1, instrument=1, place=1
Convert the following extraction notes into an instance of ConflictAttackDamage.
"In the sentence, "Protesters" serves as the damagerdestroyer, and "the offices of the holding company of Ukraine 's richest man , Rinat Akhmetov" is the artifact that was vandalized."
"""
conflictattackdamage_event = ConflictAttackDamage(

[Output]

damagerdestroyer=[Entity("Protesters"),],
artifact=[Entity("the offices of the holding company of Ukraine 's richest man , Rinat Akhmetov"),],
)

---

Example 2

[Input]

from typing import List

class Entity:
    def __init__(self, name: str):
        self.name = name

class Event:
    def __init__(self, name: str):
        self.name = name

class ContactCorrespondence(Event):
    def __init__(
        self,
        participant: List[Entity] = [],
        place: List[Entity] = [],
    ):
        self.participant = participant
        self.place = place

"""
Argument value limits: participant=2, place=1
Convert the following extraction notes into an instance of ContactCorrespondence.
"The sentence indicates that the "campaign" is a participant in the correspondence."
"""
contactcorrespondence_event = ContactCorrespondence(

[Output]

participant=[Entity("campaign"),],
)

---

Example 3

[Input]

from typing import List

class Entity:
    def __init__(self, name: str):
        self.name = name

class Event:
    def __init__(self, name: str):
        self.name = name

class ContactMeet(Event):
    def __init__(
        self,
        participant: List[Entity] = [],
        place: List[Entity] = [],
    ):
        self.participant = participant
        self.place = place

"""
Argument value limits: participant=2, place=1
Convert the following extraction notes into an instance of ContactMeet.
"In the sentence, the participants involved in the meeting are Hillary and Alinsky."
"""
contactmeet_event = ContactMeet(

[Output]

participant=[
    Entity("Hillary"),
    Entity("Alinsky"),
],
)

---

Example 4

[Input]

from typing import List

class Entity:
    def __init__(self, name: str):
        self.name = name

class Event:
    def __init__(self, name: str):
        self.name = name

class ConflictDemonstrateMarchprotestpoliticalgathering(Event):
    def __init__(
        self,
        demonstrator: List[Entity] = [],
        place: List[Entity] = [],
    ):
        self.demonstrator = demonstrator
        self.place = place

"""
Argument value limits: demonstrator=1, place=1
Convert the following extraction notes into an instance of ConflictDemonstrateMarchprotestpoliticalgathering.
"The sentence does not explicitly provide any information about the demonstrator or the place related to the protest. Therefore, there are no arguments to identify for the allowed roles."
"""
conflictdemonstratemarchprotestpoliticalgathering_event = ConflictDemonstrateMarchprotestpoliticalgathering(

[Output]

)


'''
