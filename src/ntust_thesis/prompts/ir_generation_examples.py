"""In-context examples for IR generation prompts."""

from __future__ import annotations


def build_ir_generation_in_context_examples(ir_grammar: str = "dot_notation_ir") -> str:
    """Return IR-generation ICL examples for the requested grammar."""
    if ir_grammar == "dot_notation_ir":
        return _build_dot_notation_icl_examples()
    if ir_grammar == "code4struct_ir":
        return _build_code4struct_icl_examples()
    msg = f"Unsupported IR grammar for ICL examples: {ir_grammar}"
    raise ValueError(msg)


def _build_dot_notation_icl_examples() -> str:
    """Return dot-notation IR in-context examples."""
    return (
        "Example 1:\n"
        "Allowed roles and multiplicities: victim=1\n"
        "Extraction notes:\n"
        "The sentence describes a fatal incident. "
        "The victim is clearly identified as a local shopkeeper. "
        "No other explicit role spans are given.\n"
        "Output:\n"
        "arguments.victim += a local shopkeeper\n"
        "\n"
        "Example 2:\n"
        "Allowed roles and multiplicities: attacker=1, target=1, place=1\n"
        "Extraction notes:\n"
        "The sentence describes an attack. "
        "The attacker is clearly identified as rebel fighters. "
        "The target is a police outpost. "
        "The place is stated as the northern district of Halmin. "
        "The notes also mention rising tensions in the region, which is background context and not an argument.\n"
        "Output:\n"
        "arguments.attacker += rebel fighters\n"
        "arguments.target += a police outpost\n"
        "arguments.place += the northern district of Halmin\n"
        "\n"
        "Example 3:\n"
        "Allowed roles and multiplicities: attacker=1, target=3, instrument=2\n"
        "Extraction notes:\n"
        "The sentence reports an assault. "
        "The attacker is clearly identified as government troops. "
        "Three targets are explicitly mentioned: the radio station, the central market, and a nearby warehouse. "
        "Two instruments are explicitly described: mortars and heavy machine guns. "
        "The notes also mention panic among residents, which is a consequence and not an argument.\n"
        "Output:\n"
        "arguments.attacker += government troops\n"
        "arguments.target += the radio station\n"
        "arguments.target += the central market\n"
        "arguments.target += a nearby warehouse\n"
        "arguments.instrument += mortars\n"
        "arguments.instrument += heavy machine guns\n"
        "\n"
        "Example 4:\n"
        "Allowed roles and multiplicities: attacker=1, target=1, instrument=1, place=1\n"
        "Extraction notes:\n"
        "The sentence describes a bombing. "
        "The attacker is clearly identified as an unidentified militant cell. "
        "The target is a commuter bus. "
        "The place is given as the eastern entrance of the capital. "
        "No explicit instrument span is stated in the sentence. "
        "The notes also discuss earlier security warnings, which are background context and not arguments.\n"
        "Output:\n"
        "arguments.attacker += an unidentified militant cell\n"
        "arguments.target += a commuter bus\n"
        "arguments.place += the eastern entrance of the capital\n"
        "\n"
        "Example 5:\n"
        "Allowed roles and multiplicities: person.name=1, person.title=1, organization=1\n"
        "Extraction notes:\n"
        "The sentence identifies a person involved in the event. "
        "The person's name is Dr. Leoran Vesk. "
        "The person's title is chief medical officer. "
        "The associated organization is North Harbor General Hospital. "
        "The notes also mention public concern after the event, which is explanatory context and not an argument.\n"
        "Output:\n"
        "arguments.person.name += Dr. Leoran Vesk\n"
        "arguments.person.title += chief medical officer\n"
        "arguments.organization += North Harbor General Hospital\n"
        "\n"
        "Example 6:\n"
        "Allowed roles and multiplicities: attacker=1, target=1, place=1\n"
        "Extraction notes:\n"
        "The sentence does not provide any explicit valid argument span for the event. "
        "The notes mention speculation about who might have been involved and where it may have happened, "
        "but nothing is clearly supported. "
        "No explicit attacker, target, or place span can be extracted.\n"
        "Output:\n"
    )


def _build_code4struct_icl_examples() -> str:
    """Return CODE4STRUCT in-context learning examples."""
    return (
        "# k In-context Examples\n"
        "# Example 1\n"
        "from typing import List\n\n"
        "class Entity:\n"
        "    def __init__(self, name: str):\n"
        "        self.name = name\n\n"
        "class Event:\n"
        '    def __init__(self, name: str = ""):\n'
        "        self.name = name\n\n"
        "class LifeDieNa(Event):\n"
        "    def __init__(\n"
        "        self,\n"
        "        victim: List[Entity] = [],\n"
        "        place: List[Entity] = [],\n"
        "    ):\n"
        "        self.victim = victim\n"
        "        self.place = place\n\n"
        "Allowed roles and multiplicities: victim=1, place=1\n"
        '"""\n'
        "Convert the following extraction notes into an instance of LifeDieNa.\n"
        'The trigger word is "exterminated". This event type is life.die.n/a, and the candidate roles are victim and place.\n\n'
        'The sentence states "In Saddam Hussein\'s Iraq that might work when opponents can be thrown in jail or exterminated." '
        'The word "exterminated" refers to the act of killing. The most likely victims of this extermination are "opponents". '
        'Therefore, "opponents" is the victim. The place where this extermination might occur is "Saddam Hussein\'s Iraq". '
        'Therefore, "Saddam Hussein\'s Iraq" is the place.\n'
        '"""\n'
        "lifediena_event = LifeDieNa(\n"
        '    victim=[Entity("opponents"),],\n'
        '    place=[Entity("Saddam Hussein\'s Iraq"),],\n'
        ")\n\n"
        "# Example 2\n"
        "from typing import List\n\n"
        "class Entity:\n"
        "    def __init__(self, name: str):\n"
        "        self.name = name\n\n"
        "class Event:\n"
        '    def __init__(self, name: str = ""):\n'
        "        self.name = name\n\n"
        "class ContactCollaborateNa(Event):\n"
        "    def __init__(\n"
        "        self,\n"
        "        participant: List[Entity] = [],\n"
        "        place: List[Entity] = [],\n"
        "    ):\n"
        "        self.participant = participant\n"
        "        self.place = place\n\n"
        "Allowed roles and multiplicities: participant=2, place=1\n"
        '"""\n'
        "Convert the following extraction notes into an instance of ContactCollaborateNa.\n"
        'The trigger word is "telephone". The event type is contact.collaborate.n/a. The candidate roles are participant and place.\n\n'
        'The sentence states: "station supervisor Vasily Shevchenko told NBC News by telephone from the northern city of Arkhangelsk."\n\n'
        'The trigger word is "telephone", which refers to a method of communication.\n\n'
        'The first participant in this communication is "station supervisor Vasily Shevchenko". This is the person initiating the communication. '
        'The second participant is "NBC News". This is the entity receiving the communication.\n\n'
        'The place where the communication is originating from is "the northern city of Arkhangelsk".\n'
        '"""\n'
        "contactcollaboratena_event = ContactCollaborateNa(\n"
        "    participant=[\n"
        '        Entity("station supervisor Vasily Shevchenko"),\n'
        '        Entity("NBC News"),\n'
        "    ],\n"
        '    place=[Entity("the northern city of Arkhangelsk"),],\n'
        ")\n\n"
        "# Example 3\n"
        "from typing import List\n\n"
        "class Entity:\n"
        "    def __init__(self, name: str):\n"
        "        self.name = name\n\n"
        "class Event:\n"
        '    def __init__(self, name: str = ""):\n'
        "        self.name = name\n\n"
        "class PersonnelElectWinelection(Event):\n"
        "    def __init__(\n"
        "        self,\n"
        "        voter: List[Entity] = [],\n"
        "        candidate: List[Entity] = [],\n"
        "        place: List[Entity] = [],\n"
        "    ):\n"
        "        self.voter = voter\n"
        "        self.candidate = candidate\n"
        "        self.place = place\n\n"
        "Allowed roles and multiplicities: voter=1, candidate=1, place=1\n"
        '"""\n'
        "Convert the following extraction notes into an instance of PersonnelElectWinelection.\n"
        'The trigger word is "win elections". This event is about winning elections.\n\n'
        'The sentence mentions "two Democratic operatives losing their posts because of a leaked video". This is background information and not directly related to the act of winning elections. '
        'The sentence then discusses "Robert Creamer" and "Scott Foval" leaving their jobs after video investigations. This is also background information about why they left their jobs, not about the act of winning elections itself. '
        'The sentence states that the video investigations "found them entertaining dark notions about how to win elections". This phrase directly describes the actions of the operatives in relation to winning elections. '
        "The sentence does not explicitly mention who the voters are in this context, nor does it specify a particular place where the elections are being won.\n\n"
        'Therefore, given the candidate roles (voter, candidate, place), there are no explicit mentions of a voter, a candidate, or a place in relation to the act of "win elections". '
        "No arguments can be identified for this event based on the provided sentence and candidate roles.\n"
        '"""\n'
        "personnelelectwinelection_event = PersonnelElectWinelection()\n"
    )
