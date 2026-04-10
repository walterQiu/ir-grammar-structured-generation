"""In-context examples for IR generation prompts."""

from __future__ import annotations


def build_one_stage_ir_in_context_examples(ir_grammar: str = "json") -> str:
    """Return one-stage IR-generation ICL examples for the requested grammar."""
    if ir_grammar == "json":
        return _build_one_stage_json_icl_examples()
    msg = f"Unsupported one-stage IR grammar for ICL examples: {ir_grammar}"
    raise ValueError(msg)


def _build_one_stage_json_icl_examples() -> str:
    """Return one-stage JSON in-context examples."""
    return r"""Example 1:
Event type (reference): life.die.n/a
Allowed roles and multiplicities: victim=1, place=1
Sentence: Three specific points illustrate why Americans see Trump as the problem : 1 ) Trump has trouble working with people beyond his base . In Saddam Hussein 's Iraq that might work when opponents can be thrown in jail or **exterminated** . In the United States that wo n't fly : presidents must build bridges within and beyond their core support to resolve challenges . Without alliances , a president ca n't get approval to get things done .
Output:
{"arguments":[{"role":"victim","span":"opponents"},{"role":"place","span":"Saddam Hussein's Iraq"}]}

Example 2:
Event type (reference): contact.collaborate.n/a
Allowed roles and multiplicities: participant=2, place=1
Sentence: Polar bears are not uncommon in the area , which is surrounded by pack ice in the winter , but the local population has more than doubled this year to around a dozen . And the stranded meteorologists have run out of the flares they use to scare off the beasts . " The bears live in the Arctic , you know — we can't ban them from hanging around , " station supervisor Vasily Shevchenko told NBC News by **telephone** from the northern city of Arkhangelsk . " Worst case , the station chief has a gun . " Some of the bears have taken to sleeping right outside the windows of the remote outpost , according to Russian news agency TASS , which spoke to some of the meteorologists via satellite phone .
Output:
{"arguments":[{"role":"participant","span":"station supervisor Vasily Shevchenko"},{"role":"participant","span":"NBC News"},{"role":"place","span":"the northern city of Arkhangelsk"}]}

Example 3:
Event type (reference): personnel.elect.winelection
Allowed roles and multiplicities: voter=1, candidate=1, place=1
Sentence: In addition to working alongside super - PACs , there 's the latest saga of two Democratic operatives losing their posts because of a leaked video . The Chicago Tribune explains the impact of this video in a piece titled Two local Democratic operatives lose jobs after video sting on voter fraud : Robert Creamer , husband of Rep. Jan Schakowsky , D - Ill . , and Scott Foval -- two little - known but influential Democratic political operatives -- have left their jobs after video investigations by James O'Keefe 's Project Veritas Action found them entertaining dark notions about how to **win elections** . Foval was laid off on Monday by Americans United for Change , where he had been national field director . Creamer announced Tuesday night that he was " stepping back " from the work he was doing for the unified Democratic campaign for Hillary Clinton .
Output:
{"arguments":[]}
"""


def build_two_stage_ir_in_context_examples(ir_grammar: str = "dot_notation_ir") -> str:
    """Return two-stage IR-generation ICL examples for the requested grammar."""
    if ir_grammar == "json":
        return _build_two_stage_json_icl_examples()
    if ir_grammar == "dot_notation_ir":
        return _build_two_stage_dot_notation_icl_examples()
    if ir_grammar == "code4struct_ir":
        return _build_two_stage_code4struct_icl_examples()
    msg = f"Unsupported IR grammar for ICL examples: {ir_grammar}"
    raise ValueError(msg)


def _build_two_stage_json_icl_examples() -> str:
    """Return two-stage JSON-generation in-context learning examples."""
    return r"""Example 1:
Allowed roles and multiplicities: victim=1, place=1
Extraction notes:
The trigger word is "exterminated". This event type is life.die.n/a, and the candidate roles are victim and place.

The sentence states "In Saddam Hussein's Iraq that might work when opponents can be thrown in jail or exterminated." The word "exterminated" refers to the act of killing. The most likely victims of this extermination are "opponents". Therefore, "opponents" is the victim. The place where this extermination might occur is "Saddam Hussein's Iraq". Therefore, "Saddam Hussein's Iraq" is the place.
Output:
{"arguments":[{"role":"victim","span":"opponents"},{"role":"place","span":"Saddam Hussein's Iraq"}]}

Example 2:
Allowed roles and multiplicities: participant=2, place=1
Extraction notes:
The trigger word is "telephone". The event type is contact.collaborate.n/a. The candidate roles are participant and place.

The sentence states: "station supervisor Vasily Shevchenko told NBC News by telephone from the northern city of Arkhangelsk."

The trigger word is "telephone", which refers to a method of communication.

The first participant in this communication is "station supervisor Vasily Shevchenko". This is the person initiating the communication. The second participant is "NBC News". This is the entity receiving the communication.

The place where the communication is originating from is "the northern city of Arkhangelsk".
Output:
{"arguments":[{"role":"participant","span":"station supervisor Vasily Shevchenko"},{"role":"participant","span":"NBC News"},{"role":"place","span":"the northern city of Arkhangelsk"}]}

Example 3:
Allowed roles and multiplicities: voter=1, candidate=1, place=1
Extraction notes:
The trigger word is "win elections". This event is about winning elections.

The sentence mentions "two Democratic operatives losing their posts because of a leaked video". This is background information and not directly related to the act of winning elections. The sentence then discusses "Robert Creamer" and "Scott Foval" leaving their jobs after video investigations. This is also background information about why they left their jobs, not about the act of winning elections itself. The sentence states that the video investigations "found them entertaining dark notions about how to win elections". This phrase directly describes the actions of the operatives in relation to winning elections. The sentence does not explicitly mention who the voters are in this context, nor does it specify a particular place where the elections are being won.

Therefore, given the candidate roles (voter, candidate, place), there are no explicit mentions of a voter, a candidate, or a place in relation to the act of "win elections". No arguments can be identified for this event based on the provided sentence and candidate roles.
Output:
{"arguments":[]}
"""


def _build_two_stage_dot_notation_icl_examples() -> str:
    """Return two-stage dot-notation IR in-context examples."""
    return r"""# In-context Examples
Example 1:
Allowed roles and multiplicities: victim=1, place=1
Extraction notes:
The trigger word is "exterminated". This event type is life.die.n/a, and the candidate roles are victim and place.

The sentence states "In Saddam Hussein's Iraq that might work when opponents can be thrown in jail or exterminated." The word "exterminated" refers to the act of killing. The most likely victims of this extermination are "opponents". Therefore, "opponents" is the victim. The place where this extermination might occur is "Saddam Hussein's Iraq". Therefore, "Saddam Hussein's Iraq" is the place.
Output:
arguments.victim += opponents
arguments.place += Saddam Hussein's Iraq

Example 2:
Allowed roles and multiplicities: participant=2, place=1
Extraction notes:
The trigger word is "telephone". The event type is contact.collaborate.n/a. The candidate roles are participant and place.

The sentence states: "station supervisor Vasily Shevchenko told NBC News by telephone from the northern city of Arkhangelsk."

The trigger word is "telephone", which refers to a method of communication.

The first participant in this communication is "station supervisor Vasily Shevchenko". This is the person initiating the communication. The second participant is "NBC News". This is the entity receiving the communication.

The place where the communication is originating from is "the northern city of Arkhangelsk".
Output:
arguments.participant += station supervisor Vasily Shevchenko
arguments.participant += NBC News
arguments.place += the northern city of Arkhangelsk

Example 3:
Allowed roles and multiplicities: voter=1, candidate=1, place=1
Extraction notes:
The trigger word is "win elections". This event is about winning elections.

The sentence mentions "two Democratic operatives losing their posts because of a leaked video". This is background information and not directly related to the act of winning elections. The sentence then discusses "Robert Creamer" and "Scott Foval" leaving their jobs after video investigations. This is also background information about why they left their jobs, not about the act of winning elections itself. The sentence states that the video investigations "found them entertaining dark notions about how to win elections". This phrase directly describes the actions of the operatives in relation to winning elections. The sentence does not explicitly mention who the voters are in this context, nor does it specify a particular place where the elections are being won.

Therefore, given the candidate roles (voter, candidate, place), there are no explicit mentions of a voter, a candidate, or a place in relation to the act of "win elections". No arguments can be identified for this event based on the provided sentence and candidate roles.
Output:
"""


def _build_two_stage_code4struct_icl_examples() -> str:
    """Return two-stage CODE4STRUCT in-context learning examples."""
    return r'''# In-context Examples
# Example 1
from typing import List

class Entity:
    def __init__(self, name: str):
        self.name = name

class Event:
    def __init__(self, name: str = ""):
        self.name = name

class LifeDieNa(Event):
    def __init__(
        self,
        victim: List[Entity] = [],
        place: List[Entity] = [],
    ):
        self.victim = victim
        self.place = place

Allowed roles and multiplicities: victim=1, place=1
"""
Convert the following extraction notes into an instance of LifeDieNa.
The trigger word is "exterminated". This event type is life.die.n/a, and the candidate roles are victim and place.

The sentence states "In Saddam Hussein's Iraq that might work when opponents can be thrown in jail or exterminated." The word "exterminated" refers to the act of killing. The most likely victims of this extermination are "opponents". Therefore, "opponents" is the victim. The place where this extermination might occur is "Saddam Hussein's Iraq". Therefore, "Saddam Hussein's Iraq" is the place.
"""
lifediena_event = LifeDieNa(
    victim=[Entity("opponents"),],
    place=[Entity("Saddam Hussein's Iraq"),],
)

# Example 2
from typing import List

class Entity:
    def __init__(self, name: str):
        self.name = name

class Event:
    def __init__(self, name: str = ""):
        self.name = name

class ContactCollaborateNa(Event):
    def __init__(
        self,
        participant: List[Entity] = [],
        place: List[Entity] = [],
    ):
        self.participant = participant
        self.place = place

Allowed roles and multiplicities: participant=2, place=1
"""
Convert the following extraction notes into an instance of ContactCollaborateNa.
The trigger word is "telephone". The event type is contact.collaborate.n/a. The candidate roles are participant and place.

The sentence states: "station supervisor Vasily Shevchenko told NBC News by telephone from the northern city of Arkhangelsk."

The trigger word is "telephone", which refers to a method of communication.

The first participant in this communication is "station supervisor Vasily Shevchenko". This is the person initiating the communication. The second participant is "NBC News". This is the entity receiving the communication.

The place where the communication is originating from is "the northern city of Arkhangelsk".
"""
contactcollaboratena_event = ContactCollaborateNa(
    participant=[
        Entity("station supervisor Vasily Shevchenko"),
        Entity("NBC News"),
    ],
    place=[Entity("the northern city of Arkhangelsk"),],
)

# Example 3
from typing import List

class Entity:
    def __init__(self, name: str):
        self.name = name

class Event:
    def __init__(self, name: str = ""):
        self.name = name

class PersonnelElectWinelection(Event):
    def __init__(
        self,
        voter: List[Entity] = [],
        candidate: List[Entity] = [],
        place: List[Entity] = [],
    ):
        self.voter = voter
        self.candidate = candidate
        self.place = place

Allowed roles and multiplicities: voter=1, candidate=1, place=1
"""
Convert the following extraction notes into an instance of PersonnelElectWinelection.
The trigger word is "win elections". This event is about winning elections.

The sentence mentions "two Democratic operatives losing their posts because of a leaked video". This is background information and not directly related to the act of winning elections. The sentence then discusses "Robert Creamer" and "Scott Foval" leaving their jobs after video investigations. This is also background information about why they left their jobs, not about the act of winning elections itself. The sentence states that the video investigations "found them entertaining dark notions about how to win elections". This phrase directly describes the actions of the operatives in relation to winning elections. The sentence does not explicitly mention who the voters are in this context, nor does it specify a particular place where the elections are being won.

Therefore, given the candidate roles (voter, candidate, place), there are no explicit mentions of a voter, a candidate, or a place in relation to the act of "win elections". No arguments can be identified for this event based on the provided sentence and candidate roles.
"""
personnelelectwinelection_event = PersonnelElectWinelection()
'''
