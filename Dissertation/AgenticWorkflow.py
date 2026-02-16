from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
import json

## Reducers
from typing import Annotated, List, Dict, Any, Literal, Optional
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from Tools import get_doctor_schedule,find_available_doctors
from Tools import get_current_date,get_patient_details,get_symptom_details,book_appointment,order_medicine

from dotenv import load_dotenv
load_dotenv()

class ClinicalWorkflowState(TypedDict):

    workflow_status: str
    messages: Annotated[list, add_messages]

    # Conversational intent space
    intent: Literal["appointment", "order_medicine", "reminder"]

    # Extracted slots/entities
    extracted_entities: Dict[str, Any]

    # Control flag for orchestration
    ready_for_routing: bool

    # Router output
    route: Literal["appointment", "order_medicine", "reminder"]

    # Intake output
    structured_data: Dict[str, Any]

    # Context Retrieval output
    context: Dict[str, Any]

    # Triage output
    triage_confirmed: bool
    triage: Dict[str, Any]

    # Validation output
    is_valid: bool
    validation_errors: Optional[str]

    # Scheduling output
    appointment_confirmed: bool
    appointment_details: Optional[Dict[str, Any]]

    # Pharmacy
    medication_summary: Optional[Dict[str, Any]]

    # Reminders
    reminders: Optional[List[str]]

llm_conversation=ChatGroq(
    # model="llama-3.3-70b-versatile",
    model="openai/gpt-oss-120b",
    timeout=60
)

llm_reason=ChatGroq(
    model="openai/gpt-oss-120b",
    timeout=60
)

llm_validation=ChatGroq(
    # model="openai/gpt-oss-safeguard-20b",
    model="openai/gpt-oss-120b",
    timeout=60
)

tools=[get_doctor_schedule,find_available_doctors]
llm_tools = llm_reason.bind_tools(tools)

def conversation_ai_agent(state:ClinicalWorkflowState):

    system_prompt = f"""
    You are a clinical conversational AI assistant.

    Your tasks:
    1. You are an expert in healthcare domain, specially in clinical and administrative domain
    2. Continue the conversation naturally
    3. Answer generic medical questions but recommend consulting a doctor
        If the user is booking an appointment, do NOT repeat generic medical disclaimers
    4. Identify ONE intent from:
        - appointment
        - order_medicine
        - reminder
        - general_advise
    5. Extract relevant entities
        - appointment entities:
            - patient_name
            - symptoms (accumulative, never clear)
            - (doctor_name) or speciality, Optional
            - preferred_date, extract in dd-MMM-yy format
            - preferred_time, extract in HH:mm format
        - order_medicine entities:
            - medicine
            - dosage 
            - frequency and duration, optional but should ask
            - quantity
            - shipping address
        - reminder:
            - start_date, time, repeating (true or false), frequency (if repeating), reminder_text
    6. Decide if enough information is extracted for routing, but wait for reconfirm from user before ready for routing
    7. For non generic medicine order, take consent from user that consultation with doctor is done. Reject if consultation not done
    8. Do not mention that appointment is scheduled or order is booked.
    9. Consider today's date as {get_current_date()}

    Respond ONLY in valid JSON:
        "reply": <Str>,
        "intent": <Str>,
        "entities": <Dictionary>,
        "ready_for_routing": true/false
    """

    llm_response = llm_conversation.invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    json_response = json.loads(llm_response.content)

    return {
        "workflow_status": state.get("workflow_status", "WIP"),
        "messages": [{"role": "assistant", "content": json_response["reply"]}],
        "intent": json_response["intent"],
        "extracted_entities": json_response.get("entities", {}),
        "ready_for_routing": json_response["ready_for_routing"],
        "triage_confirmed": False,
        "appointment_confirmed": False
    }

def router_agent(state: ClinicalWorkflowState):

    if state["intent"] == "appointment":
        route = "appointment"
    elif state["intent"] == "order_medicine":
        route = "order_medicine"
    elif state["intent"] == "reminder":
        route = "reminder"
    else:
        route = "none"

    return {"route": route}

def intake_agent(state: ClinicalWorkflowState):

    from datetime import datetime

    # Retrive patient data from extracted_entities
    entities = state["extracted_entities"]

    # Identify day from date
    date = entities.get("preferred_date") # dd-MMM-yy format
    date_obj = datetime.strptime(date, "%d-%b-%y")
    day = date_obj.strftime("%A")

    # Get Patient ID from Patient Table
    patient_name = entities.get("patient_name")
    patient_id = get_patient_details(patient_name)

    structured_data = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "symptoms": entities.get("symptoms"),
        "preferred_doctor_name": entities.get("doctor_name"),
        "preferred_speciality": entities.get("speciality"),
        "preferred_date": date,
        "preferred_time": entities.get("preferred_time"),
        "preferred_day": day
    }

    reply_message = f"We are checking for doctor's availbility on {day}, {date} around {entities.get("preferred_time")}."
    
    return {
            "messages": [{"role": "assistant", "content": reply_message}],
            "structured_data": structured_data
            }

def context_retrieval_agent(state: ClinicalWorkflowState):

    # Retrieve past medical history from DB
    patient_id = state["structured_data"].get("patient_id")
    if not patient_id == 0:
        symptom_details = get_symptom_details(patient_id)

        if len(symptom_details) > 0:

            if len(symptom_details) > 1:
                symptom_text = ", ".join(symptom_details[:-1]) + " and " + symptom_details[-1]
            elif len(symptom_details) == 1:
                symptom_text = symptom_details[0]

            reply_message = f"You have a past medical history of {symptom_text}"

            context = {
                        "previous_conditions": symptom_details
            }

            return {
                "messages": [{"role": "assistant", "content": reply_message}],
                "context": context
                }

    return None

def triage_reasoning_agent(state: ClinicalWorkflowState):

    # Infer speciality from symptom and the find doctor based on the derieved speciality
    system_prompt = f"""
    You are a triage reasoning agent. Consider today's date as {get_current_date()}.

    Your tasks: Execute the following task sequentially
    - If doctor's name is provided in the user message, check if doctor's schedule matches with preferred date and time.
    - If schedule does NOT match, identify the doctor's specility and then find another doctor with the same specility matches best with preferred date and time.
    - If doctor name is NOT provided, but specility is provided in user message, then find a doctor based on the specility who matches best with preferred date and time.
    - If doctor name or specility is NOT provided in user message, Infer speciality based on the symptom. Then find a doctor based on the specility who matches best with preferred date and time.
    - If NO match, suggest the doctor of the same specility based on earliest availble date and time.
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

    return {"messages": llm_response}

def appointment_validation_agent(state: ClinicalWorkflowState):

    structured_data = state["structured_data"]

    system_prompt = f"""
        You are an outpatient appointment validation assistant.

        Instructions: 
        - Identify if the appointment details are confirmed by user
        - Extract the following appointment details post confirmation from user
        - Do not mention that appointment is booked
        Consider today's date as {get_current_date()}

        Respond ONLY in valid JSON with the following elements:
        "reply": "...",
        "doctor_id": "...",
        "doctor_name": "...",
        "date": "...",
        "time": "...",
        "day", "...",
        "confirmed_by_user": true/false,
        """

    llm_response = llm_validation.invoke([{"role": "system", "content": system_prompt}] + state["messages"])

    json_response = json.loads(llm_response.content)

    if json_response.get("confirmed_by_user"):
            state["appointment_details"] = {
                                            "patient_id": structured_data.get("patient_id"),
                                            "patient_name": structured_data.get("patient_name"),
                                            "symptoms": structured_data.get("symptoms"),
                                            "doctor_id": json_response.get("doctor_id"),
                                            "doctor_name": json_response.get("doctor_name"),
                                            "date": json_response.get("date"),
                                            "time": json_response.get("time"),
                                            "day": json_response.get("day")
                                            }
            state["triage_confirmed"] = True

    data = state.get("appointment_details", {})
    required_fields = ["patient_name", "symptoms", "doctor_id", "date", "time"]
    missing = [f for f in required_fields if f not in data]

    if missing:
        return {
                "triage_confirmed": state.get("triage_confirmed"),
                "is_valid": False,
                "validation_errors": f"Missing fields: {missing}",
                "appointment_details": state.get("appointment_details", {})
                }

    return {
            "triage_confirmed": state.get("triage_confirmed"),
            "is_valid": True,
            "validation_errors": None,
            "appointment_details": state.get("appointment_details", {})
            }

def scheduling_agent(state: ClinicalWorkflowState):

    if state.get("is_valid", True):

        # Update DB with appointment details
        appointment = book_appointment(state)
        appointment_id = appointment["appointment_id"]

        reply = f"The appointment is booked. Appointment ID APT-00{appointment_id}"

        workflow_status = "COMPLETED"
        appointment_confirmed = True

    else:
        workflow_status = "WIP"
        appointment_confirmed = False

    return {
            "workflow_status": workflow_status,
            "messages": [{"role": "assistant", "content": reply}],
            "appointment_confirmed": appointment_confirmed,
    }

def medicine_intake_agent(state: ClinicalWorkflowState):

    # Retrive medicine order data from extracted_entities
    entities = state["extracted_entities"]

    structured_data = {
        "medicine": entities.get("medicine"),
        "dosage": entities.get("dosage"),
        "frequency": entities.get("frequency", None),
        "duration": entities.get("duration", None),
        "quantity": entities.get("quantity", None),
        "shipping_address": entities.get("shipping_address")
    }

    return {
        "structured_data": structured_data
    }

def medicine_order_validation_agent(state: ClinicalWorkflowState):
    data = state["extracted_entities"]

    required_fields = ["dosage", "medicine", "quantity", "shipping_address"]
    missing = [f for f in required_fields if f not in data]

    if missing:
        return {
            "is_valid": False,
            "validation_errors": f"Missing fields: {missing}"
        }

    return {"is_valid": True, "validation_errors": None}

def pharmacy_agent(state: ClinicalWorkflowState):
    """
    Takes medicine Order and Store in Medicine Order Table.
    """

    # Update DB with medicine order details
    order = order_medicine(state)
    order_id = order["order_id"]

    entities = state["extracted_entities"]

    medication_summary = {
        "medicine": entities.get("medicine"),
        "dosage": entities.get("dosage"),
        "quantity": entities.get("quantity"),
        "frequency": entities.get("frequency"),
        "duration": entities.get("duration"),
        "shipping_address": entities.get("shipping_address")
    }

    confirmation_message = (
                f"Your order is confirmed. Order ID ORR-000{order_id}."
                )               

    return {
            "messages": [{"role": "assistant", "content": confirmation_message}],
            "medication_summary": medication_summary
            }

def reminder_validation_agent(state: ClinicalWorkflowState):
    data = state["extracted_entities"]

    required_fields = ["start_date", "time", "reminder_text"]
    missing = [f for f in required_fields if f not in data]

    if missing:
        return {
            "is_valid": False,
            "validation_errors": f"Missing fields: {missing}"
        }

    return {"is_valid": True, "validation_errors": None}

def reminder_agent(state: ClinicalWorkflowState):

    entities = state["extracted_entities"]
    reminders = {
        "start_date": entities.get("start_date", None),
        "time": entities.get("time", None),
        "frequency": entities.get("frequency", None),
        "duration": entities.get("duration", None),
        "repeating": entities.get("repeating", False),
        "reminder_text": entities.get("reminder_text", None)
    }

    if state["route"]=="appointment":
        reminder_text = (
                        f"Appointment Reminder: Your appointment is scheduled with {state['appointment_details']['doctor_name']} "
                        f"on {state['appointment_details']['date']} at {state['appointment_details']['time']}. Please come 15 minutes early."
        )

        reminders["reminder_text"] = reminder_text

    if state["route"]=="order_medicine":
        
        reminder_text = (
                        f"Medication Reminder: Take {state['medication_summary']['medicine']} "
                        f"{state['medication_summary']['dosage']}"
        )
                
        if reminders.get("frequency"):
            reminder_text = reminder_text + f" {reminders.get('frequency')}"

        reminder_text = reminder_text + " as per the prescribed schedule."

        reminders["reminder_text"] = reminder_text

    from twilio.rest import Client

    TWILIO_SID = os.getenv("TWILIO_SID")
    TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
    client = Client(TWILIO_SID, TWILIO_TOKEN)

    message = client.messages.create(
                    from_='whatsapp:+14155238886',
                    body=reminders["reminder_text"],
                    to='whatsapp:+919163040468'
                )
    
    print(message.sid)

    return {"reminders": reminders}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ClinicalWorkflowState)

# Register agents
graph.add_node("conversation", conversation_ai_agent)
graph.add_node("router", router_agent)
graph.add_node("appointment_intake", intake_agent)
graph.add_node("context", context_retrieval_agent)
graph.add_node("triage", triage_reasoning_agent)
graph.add_node("appointment_validation", appointment_validation_agent)
graph.add_node("medicine_validation", medicine_order_validation_agent)
graph.add_node("reminder_validation", reminder_validation_agent)
graph.add_node("scheduling", scheduling_agent)
graph.add_node("pharmacy", pharmacy_agent)
graph.add_node("reminder", reminder_agent)

# Register Tools
graph.add_node("tools", ToolNode(tools))

# Conditional Routing Logic
def route_from_start(state: ClinicalWorkflowState):

    workflow_status = state.get("workflow_status", "STARTED")

    if state.get("ready_for_routing") and workflow_status=="WIP":
        return "triage"

    return "conversation"

graph.add_conditional_edges(
    START,
    route_from_start,
    {
        "conversation": "conversation",
        "triage": "triage"
    }
)

# Conditional Routing Logic
def route_from_conversation(state: ClinicalWorkflowState):
    if state["ready_for_routing"]:
        return "router"
    else:
        return END

graph.add_conditional_edges(
    "conversation",
    route_from_conversation,
    {
        "router": "router",
        END: END
    }
)

def route_from_router(state: ClinicalWorkflowState):
    if state["route"] == "appointment":
        if state["triage_confirmed"]:
            return "appointment_validation"
        else:
            return "appointment_intake"
    elif state["route"] == "order_medicine":
        return "medicine_validation"
    elif state["route"] == "reminder":
        return "reminder_validation"
    return END

graph.add_conditional_edges(
    "router",
    route_from_router,
    {
        "appointment_intake": "appointment_intake",
        "appointment_validation": "appointment_validation",
        "medicine_validation": "medicine_validation",
        "reminder_validation": "reminder_validation",
        END: END
    }
)

# Define Sequential Dependencies
graph.add_edge("appointment_intake", "context")
graph.add_edge("context", "triage")

graph.add_conditional_edges(
    "triage",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to appointment_validation
    tools_condition,
    {
        "tools": "tools",                      # assistant produced a tool call
        "__end__": "appointment_validation"    # assistant produced a normal message

    }
)

graph.add_edge("tools","triage")

# Validation Gate (Critical for Clinical Safety)
def validation_gate(state: ClinicalWorkflowState):
    if state["triage_confirmed"] and state["is_valid"]:
        return "scheduling"
    return END

graph.add_conditional_edges(
    "appointment_validation",
    validation_gate,
    {
        "scheduling": "scheduling",
        END: END
    }
)

# Validation Gate (Critical for Clinical Safety)
def validation_gate(state: ClinicalWorkflowState):
    if state["is_valid"]:
        return "pharmacy"
    return END

graph.add_conditional_edges(
    "medicine_validation",
    validation_gate,
    {
        "pharmacy": "pharmacy",
        END: END
    }
)

# Validation Gate (Critical for Clinical Safety)
def validation_gate(state: ClinicalWorkflowState):
    if state["is_valid"]:
        return "reminder"
    return END

graph.add_conditional_edges(
    "reminder_validation",
    validation_gate,
    {
        "reminder": "reminder",
        END: END
    }
)

graph.add_edge("scheduling", "reminder")
graph.add_edge("pharmacy", "reminder")
graph.add_edge("reminder", END)

graph_builder=graph.compile(checkpointer=checkpointer)