from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
import json

## Reducers
from typing import Annotated, Dict, Any
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

import os
print("DB PATH:", os.path.abspath("PatientCareDB.db"))

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

from Tools import get_doctor_schedule,find_available_doctors
tools=[get_doctor_schedule,find_available_doctors]
llm_tools = llm.bind_tools(tools)

'''
response = llm_tools.invoke([HumanMessage(content="Hi, how are you?")])
print(response.content)

response_tool = llm_tools.invoke([HumanMessage(content="Please suggest me a doctor for Cardiology")])
print(response_tool.content)
print(response_tool.content_blocks)
print("TOOL CALLS:", response_tool.tool_calls)
'''

class ClinicalWorkflowState(TypedDict):

    messages: Annotated[list, add_messages]
    triage: Dict[str, Any]

def triage_reasoning_agent(state: ClinicalWorkflowState):

    # Infer speciality from symptom and the find doctor based on the derieved speciality
    system_prompt = """
    You are a triage reasoning agent.

    Your tasks: Execute the following task sequentially
    - If doctor's name is provided in the user message, check if doctor's schedule matches with preferred date and time.
    - If schedule does NOT match, identify the doctor's specility and then find another doctor with the same specility matches best with preferred date and time.
    - If doctor name is NOT provided, but specility is provided in user message, then find a doctor based on the specility who matches best with preferred date and time.
    - If doctor name or specility is NOT provided in user message, Infer speciality based on the symptom. Then find a doctor based on the specility who matches best with preferred date and time.
    - If NO match, suggest the earliest availble doctor with the same specility.
    - DO NOT call the same tool repeatedly. Call the tool ONLY ONCE.
    - After showing the result, do not confirm the appointment immediately. Instead request user to confirm.

    Available Speciality: 
        - General Medicine
        - Cardiology
        - Orthopedics
        - Gynecology
        - Pediatrics
        - Dermatology
        - Neurology
        - Psychiatry
        - ENT
        - Ophthalmology
        - Pulmonology
        - Gastroenterology
        - Endocrinology
        - Nephrology
    """

    llm_response = llm_tools.invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"]  
    )

    system_prompt = """
    Extract and reply only in valid JSON. No tool calls.
    JSON schema:
    {
        "specility": "...",
        "doctor": "...",
        "available_time": "HH:mm"
    }
    """

    llm_response_json = llm.invoke(
        [{"role": "system", "content": system_prompt}] + [{"role": "user", "content": llm_response.content}]   
    )
    
    '''
    json_response = json.loads(llm_response.content)
            
    triage = {
                "speciality": json_response.get("speciality", "None"),
                "doctor_name": json_response.get("doctor", "None")
             }
    
    return {triage}
    '''

    return {"messages": llm_response}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ClinicalWorkflowState)

# Register agents
graph.add_node("triage", triage_reasoning_agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "triage")
graph.add_conditional_edges(
    "triage",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
graph.add_edge("tools","triage")
graph.add_edge("triage", END)

graph_builder=graph.compile(checkpointer=checkpointer)

CONFIG = {'configurable': {'thread_id': 'thread-5'}}

# NO doctor name or specility - Schedule Match
# response = graph_builder.invoke({"messages": "I have High BP. Please suggest me a doctor who is available on Monday at 3 PM"}, config=CONFIG, debug=True)

# NO doctor name or specility - Schedule NOT Match
# response = graph_builder.invoke({"messages": "I have High BP. Please suggest me a doctor who is available on Sunday at 11 AM"}, config=CONFIG, debug=True)

# Specility Mentioned - Schedule Match
# response = graph_builder.invoke({"messages": "I have High BP. Please suggest me a General physician who is available on Monday at 1:30 PM"}, config=CONFIG, debug=True)

# Specility Mentioned - Schedule NOT Match
# response = graph_builder.invoke({"messages": "I have High BP. Please suggest me a General physician who is available on Monday at 3:30 PM"}, config=CONFIG, debug=True)

# Doctor Mentioned - Schedule Match
# response = graph_builder.invoke({"messages": "I have High BP. I want to see Dr. Sandeep Singh on Monday at 10 AM"}, config=CONFIG, debug=True)

# Doctor Mentioned - Schedule NOT Match
response = graph_builder.invoke({"messages": "I have High BP. I want to see Dr. Sandeep Singh on Tuesday at 2:30 PM"}, config=CONFIG, debug=True)

for m in response['messages']:
    m.pretty_print()


