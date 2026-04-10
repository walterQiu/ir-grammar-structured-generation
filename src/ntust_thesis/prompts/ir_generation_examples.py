"""In-context examples for IR generation prompts."""

from __future__ import annotations


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
    return (
        "Example 1:\n"
        "Allowed roles and multiplicities: victim=1, place=1\n"
        "Extraction notes:\n"
        'The trigger word is "exterminated". This event type is life.die.n/a, and the candidate roles are victim and place.\n\n'
        'The sentence states "In Saddam Hussein\'s Iraq that might work when opponents can be thrown in jail or exterminated." '
        'The word "exterminated" refers to the act of killing. The most likely victims of this extermination are "opponents". '
        'Therefore, "opponents" is the victim. The place where this extermination might occur is "Saddam Hussein\'s Iraq". '
        'Therefore, "Saddam Hussein\'s Iraq" is the place.\n'
        "Output:\n"
        '{"arguments":[{"role":"victim","span":"opponents"},{"role":"place","span":"Saddam Hussein\'s Iraq"}]}\n'
        "\n"
        "Example 2:\n"
        "Allowed roles and multiplicities: participant=2, place=1\n"
        "Extraction notes:\n"
        'The trigger word is "telephone". The event type is contact.collaborate.n/a. The candidate roles are participant and place.\n\n'
        'The sentence states: "station supervisor Vasily Shevchenko told NBC News by telephone from the northern city of Arkhangelsk."\n\n'
        'The trigger word is "telephone", which refers to a method of communication.\n\n'
        'The first participant in this communication is "station supervisor Vasily Shevchenko". This is the person initiating the communication. '
        'The second participant is "NBC News". This is the entity receiving the communication.\n\n'
        'The place where the communication is originating from is "the northern city of Arkhangelsk".\n'
        "Output:\n"
        '{"arguments":[{"role":"participant","span":"station supervisor Vasily Shevchenko"},{"role":"participant","span":"NBC News"},{"role":"place","span":"the northern city of Arkhangelsk"}]}\n'
        "\n"
        "Example 3:\n"
        "Allowed roles and multiplicities: voter=1, candidate=1, place=1\n"
        "Extraction notes:\n"
        'The trigger word is "win elections". This event is about winning elections.\n\n'
        'The sentence mentions "two Democratic operatives losing their posts because of a leaked video". This is background information and not directly related to the act of winning elections. '
        'The sentence then discusses "Robert Creamer" and "Scott Foval" leaving their jobs after video investigations. This is also background information about why they left their jobs, not about the act of winning elections itself. '
        'The sentence states that the video investigations "found them entertaining dark notions about how to win elections". This phrase directly describes the actions of the operatives in relation to winning elections. '
        "The sentence does not explicitly mention who the voters are in this context, nor does it specify a particular place where the elections are being won.\n\n"
        'Therefore, given the candidate roles (voter, candidate, place), there are no explicit mentions of a voter, a candidate, or a place in relation to the act of "win elections". '
        "No arguments can be identified for this event based on the provided sentence and candidate roles.\n"
        "Output:\n"
        '{"arguments":[]}\n'
    )


def _build_two_stage_dot_notation_icl_examples() -> str:
    """Return two-stage dot-notation IR in-context examples."""
    return (
        "# In-context Examples\n"
        "Example 1:\n"
        "Allowed roles and multiplicities: victim=1, place=1\n"
        "Extraction notes:\n"
        'The trigger word is "exterminated". This event type is life.die.n/a, and the candidate roles are victim and place.\n\n'
        'The sentence states "In Saddam Hussein\'s Iraq that might work when opponents can be thrown in jail or exterminated." '
        'The word "exterminated" refers to the act of killing. The most likely victims of this extermination are "opponents". '
        'Therefore, "opponents" is the victim. The place where this extermination might occur is "Saddam Hussein\'s Iraq". '
        'Therefore, "Saddam Hussein\'s Iraq" is the place.\n'
        "Output:\n"
        "arguments.victim += opponents\n"
        "arguments.place += Saddam Hussein's Iraq\n"
        "\n"
        "Example 2:\n"
        "Allowed roles and multiplicities: participant=2, place=1\n"
        "Extraction notes:\n"
        'The trigger word is "telephone". The event type is contact.collaborate.n/a. The candidate roles are participant and place.\n\n'
        'The sentence states: "station supervisor Vasily Shevchenko told NBC News by telephone from the northern city of Arkhangelsk."\n\n'
        'The trigger word is "telephone", which refers to a method of communication.\n\n'
        'The first participant in this communication is "station supervisor Vasily Shevchenko". This is the person initiating the communication. '
        'The second participant is "NBC News". This is the entity receiving the communication.\n\n'
        'The place where the communication is originating from is "the northern city of Arkhangelsk".\n'
        "Output:\n"
        "arguments.participant += station supervisor Vasily Shevchenko\n"
        "arguments.participant += NBC News\n"
        "arguments.place += the northern city of Arkhangelsk\n"
        "\n"
        "Example 3:\n"
        "Allowed roles and multiplicities: voter=1, candidate=1, place=1\n"
        "Extraction notes:\n"
        'The trigger word is "win elections". This event is about winning elections.\n\n'
        'The sentence mentions "two Democratic operatives losing their posts because of a leaked video". This is background information and not directly related to the act of winning elections. '
        'The sentence then discusses "Robert Creamer" and "Scott Foval" leaving their jobs after video investigations. This is also background information about why they left their jobs, not about the act of winning elections itself. '
        'The sentence states that the video investigations "found them entertaining dark notions about how to win elections". This phrase directly describes the actions of the operatives in relation to winning elections. '
        "The sentence does not explicitly mention who the voters are in this context, nor does it specify a particular place where the elections are being won.\n\n"
        'Therefore, given the candidate roles (voter, candidate, place), there are no explicit mentions of a voter, a candidate, or a place in relation to the act of "win elections". '
        "No arguments can be identified for this event based on the provided sentence and candidate roles.\n"
        "Output:\n"
    )


def _build_two_stage_code4struct_icl_examples() -> str:
    """Return two-stage CODE4STRUCT in-context learning examples."""
    return (
        "# In-context Examples\n"
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
