import sqlite3
from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict
from datetime import datetime

class State(TypedDict):
    appointment_details: Optional[Dict[str, Any]]

# Tool Function - DB Lookup
def find_available_doctors (speciality: str) -> str:
    """
    Tool: Find doctors by speciality.
    """

    conn = sqlite3.connect("PatientCareDB.db")
    
    cursor = conn.cursor()

    base_query = """
    SELECT
        DOCTOR_ID,
        NAME,
        QUALIFICATION,
        AVAILABLE_FROM,
        AVAILABLE_TO,
        AVAILABLE_DAYS
    FROM DOCTOR
    WHERE
        SPECIALITY = ?
        AND IS_ACTIVE = 1
    ORDER BY AVAILABLE_FROM LIMIT 5
    """

    params = [speciality]
    cursor.execute(base_query, params)
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "doctor_id": r[0],
            "name": r[1],
            "qualification": r[2],
            "available_from": r[3],
            "available_to": r[4],
            "available_days": r[5]
        }
        for r in rows
    ]

def get_doctor_schedule (name: str) -> str:
    """
    Tool: Retrieve doctor's schedule by name.
    """

    conn = sqlite3.connect("PatientCareDB.db")
    
    cursor = conn.cursor()

    base_query = """
    SELECT
        DOCTOR_ID,
        SPECIALITY,
        NAME,
        QUALIFICATION,
        AVAILABLE_FROM,
        AVAILABLE_TO,
        AVAILABLE_DAYS
    FROM DOCTOR
    WHERE
        LOWER(NAME) like ?
        AND IS_ACTIVE = 1
    ORDER BY AVAILABLE_FROM LIMIT 5
    """

    params = [f"%{name.lower()}%"]
    cursor.execute(base_query, params)
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "doctor_id": r[0],
            "speciality": r[1],
            "name": r[2],
            "qualification": r[3],
            "available_from": r[4],
            "available_to": r[5],
            "available_days": r[6]
        }
        for r in rows
    ]

def get_patient_details (name: str) -> str:
    """
    Tool: Get Patient's details by name.
    """

    conn = sqlite3.connect("PatientCareDB.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT PATIENT_ID
    FROM PATIENT
    WHERE LOWER(NAME)=LOWER(?)
    """, (name,))

    result = cursor.fetchone()
    conn.close()

    if result:
        patient_id = result[0]
        return patient_id
    else:
        return 0
    
def get_symptom_details (patient_id: str) -> List[str]:
    """
    Tool: Fetch all symptoms recorded for a given patient_id.
    """

    conn = sqlite3.connect("PatientCareDB.db")
    cursor = conn.cursor()

    cursor.execute("""
            SELECT SYMPTOMS
            FROM SYMPTOMS
            WHERE PATIENT_ID = ?
        """, (patient_id,))

    rows = cursor.fetchall()
    conn.close()

    # Convert rows to simple list
    symptoms_list = [row[0] for row in rows if row[0] is not None]

    return symptoms_list

def book_appointment(state: State) -> str:
    """
    Tool: Insert appointment and symptoms into database
    """
    details = state["appointment_details"]

    patient_id = details["patient_id"]
    patient_name = details["patient_name"]
    doctor_id = details["doctor_id"]
    symptoms = details["symptoms"]
    time = details["time"]

    # Convert '15-Feb-26' -> '2026-02-15' (SQLite DATE format)
    date_obj = datetime.strptime(details["date"], "%d-%b-%y")
    db_date = date_obj.strftime("%Y-%m-%d")

    try:
        conn = sqlite3.connect("PatientCareDB.db")
        cursor = conn.cursor()

        # INSERT PATIENT if not present in Table
        if patient_id == 0:
            cursor.execute("""
                        INSERT INTO PATIENT (NAME)
                        VALUES (?)
                    """, (patient_name,))

            patient_id = cursor.lastrowid

        # INSERT APPOINTMENT
        cursor.execute("""
            INSERT INTO APPOINTMENT (PATIENT_ID, DOCTOR_ID, DATE, TIME)
            VALUES (?, ?, ?, ?)
        """, (patient_id, doctor_id, db_date, time))

        appointment_id = cursor.lastrowid

        # INSERT SYMPTOMS
        cursor.execute("""
            INSERT INTO SYMPTOMS (PATIENT_ID, SYMPTOMS)
            VALUES (?, ?)
        """, (patient_id, symptoms))

        # Commit transaction
        conn.commit()

        return {
            "appointment_id": appointment_id,
            "status": "confirmed"
        }

    except Exception as e:
        conn.rollback()
        print("Database error:", e)
        return {
            "status": "failed"
        }

    finally:
        conn.close()

def order_medicine(state: State) -> str:
    """
    Tool: Insert medicine order details into database
    """

    details = state["extracted_entities"]

    medicine = details["medicine"]
    dosage = details["dosage"]
    quantity = details["quantity"]
    shipping_address = details["shipping_address"]

    try:
        conn = sqlite3.connect("PatientCareDB.db")
        cursor = conn.cursor()

        # INSERT APPOINTMENT
        cursor.execute("""
            INSERT INTO MEDICINE_ORDER (MEDICINE, DOSAGE, QUANTITY, SHIPPING_ADDRESS)
            VALUES (?, ?, ?, ?)
        """, (medicine, dosage, quantity, shipping_address))

        # get generated ORDER_ID
        order_id = cursor.lastrowid

        # Commit transaction
        conn.commit()

        return {
            "order_id": order_id,
            "status": "confirmed"
        }

    except Exception as e:
        conn.rollback()
        print("Database error:", e)
        return {
            "status": "failed"
        }

    finally:
        conn.close()

def get_current_date() -> str:
    """
    Tool: Get current system date
    Example: 15-Feb-26, Sunday
    """

    now = datetime.now()

    formatted_date = now.strftime("%d-%b-%y, %A")

    return formatted_date