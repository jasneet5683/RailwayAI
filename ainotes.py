import json
import os
import requests
# import smtplib
import pandas as pd
# from email.message import EmailMessage
from typing import Optional 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

# OPENAI IMPORTS
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

#added for Email Attachement support
import matplotlib
matplotlib.use('Agg') # Required for Render/Server usage
import matplotlib.pyplot as plt
import io
import base64
# Add these imports for app shcheduler
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

#for Speech Recognition
from fastapi import FastAPI, UploadFile, File, HTTPException
import speech_recognition as sr
from pydub import AudioSegment
import io

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
openai_key = os.getenv("OPENAI_API_KEY")
email_sender = os.getenv("EMAIL_SENDER")
email_password = os.getenv("EMAIL_PASSWORD")
SHEET_NAME = "Task_Manager"  # Make sure this matches your Google Sheet Name exactly

# Global variables to hold data state
excel_text_context = ""
document_loaded = False

# --- DATA MODELS ---
class PromptRequest(BaseModel):
    prompt: str

class TaskRequest(BaseModel):
    task_name: str
    assigned_to: str
    start_date: str
    end_date: str
    status: str
    client: str
    #new optional field for sending email to notify task addition
    notify_email: Optional[str] = None 

# --- 1. HELPER: CONNECT TO GOOGLE SHEETS ---
def get_google_sheet():
    try:
        json_creds = os.getenv("GOOGLE_CREDS")
        if not json_creds:
            print("❌ Error: GOOGLE_CREDS not found in environment.")
            return None
        creds_dict = json.loads(json_creds)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open(SHEET_NAME).sheet1
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return None


# --- 2. HELPER: LOAD DATA (Fixes 'Sheet is Empty') ---
def load_data_global():
    global excel_text_context, document_loaded
    print("🔄 Loading data from Google Sheets...")
    sheet = get_google_sheet()
    if not sheet:
        document_loaded = False
        return

    try:
        data = sheet.get_all_records()
        if not data:
            print("⚠️ Sheet is empty or couldn't read records.")
            excel_text_context = "No data found."
            document_loaded = True
            return

        df = pd.DataFrame(data)
        df.fillna("N/A", inplace=True)
        
        # Convert dates to string to avoid errors
        for col in df.columns:
            if "date" in col.lower():
                df[col] = df[col].astype(str)

        excel_text_context = df.to_csv(index=False)
        document_loaded = True
        print("✅ Data Successfully Loaded into Memory.")
        
    except Exception as e:
        print(f"❌ Error processing data: {str(e)}")
        document_loaded = False

#..............get email for assignees
def get_email_for_assignee(assignee_name):
    """
    Finds the email for a given name using the Team Directory.
    """
    # 1. Fetch the directory
    team_map = get_team_directory()
    
    # 2. Look up the name (case insensitive)
    clean_name = assignee_name.strip().lower()
    
    email = team_map.get(clean_name)
    
    if email:
        return email
    else:
        print(f"❌ No email found for user: {assignee_name}")
        return None


# Helper for Chart Generator function
def generate_chart_base64():
    """
    Generates a chart based on current Google Sheet data.
    No arguments required - fetches data internally.
    """
    try:
        # 1. Fetch fresh data directly
        sheet = get_google_sheet()
        if not sheet:
            print("❌ Chart Error: Could not connect to sheet.")
            return None

        data = sheet.get_all_records()
        if not data:
            print("⚠️ Chart Error: No data in sheet.")
            return None

        # 2. Prepare DataFrame
        df = pd.DataFrame(data)
        
        # Ensure 'Status' column exists (flexible check)
        # If your column is named differently (e.g., 'Project Status'), update it here.
        if 'status' not in df.columns:
            print("⚠️ Chart Error: 'Status' column not found.")
            return None

        # 3. Create the plot
        plt.clf() # Clear previous figures
        plt.figure(figsize=(8, 5))
        
        counts = df['status'].value_counts()
        
        # Plot with some nice colors
        counts.plot(kind='bar', color=['#667eea', '#764ba2', '#28a745'])
        plt.title('Project Status Overview')
        plt.xlabel('Status')
        plt.ylabel('Count')
        plt.xticks(rotation=45) # Rotate labels if they are long
        plt.tight_layout()
        
        # 4. Save to memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # 5. Convert to Base64 String
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # 6. Cleanup
        plt.close()
        
        return img_str

    except Exception as e:
        print(f"❌ Chart generation failed: {e}")
        return None

#---- Table Generator Function
def generate_table_base64():
    """
    Generates a table image (PNG) based on current Google Sheet data.
    """
    try:
        sheet = get_google_sheet()
        if not sheet:
            return None

        data = sheet.get_all_records()
        if not data:
            return None

        df = pd.DataFrame(data)
        
        # Create figure for the table
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6)) # Adjust size as needed
        ax.axis('tight')
        ax.axis('off')
        
        # Create the table
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.15] * len(df.columns)
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Add a nice header color
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#667eea')
                cell.set_text_props(weight='bold', color='white')

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str

    except Exception as e:
        print(f"❌ Table generation failed: {e}")
        return None

#--------- Helper for task creation function
def send_task_creation_email(to_email, task_name, assigned_to, client, due_date):
    """
    Sends a specific email notification when a new task is added.
    """
    api_key = os.getenv("BREVO_API_KEY")
    sender_email = os.getenv("SENDER_EMAIL")
    sender_name = os.getenv("SENDER_NAME", "AI Assistant")

    if not api_key:
        print("❌ Error: Missing BREVO_API_KEY")
        return False

    url = "https://api.brevo.com/v3/smtp/email"
    headers = {
        "accept": "application/json",
        "api-key": api_key,
        "content-type": "application/json"
    }

    # Professional HTML Email Template
    html_content = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px;">
        <h2 style="color: #667eea;">🚀 New Task Assigned</h2>
        <p>Hello,</p>
        <p>A new task has been added to the tracker and assigned to <strong>{assigned_to}</strong>.</p>
        <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Task Name:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{task_name}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Client:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{client}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Due Date:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{due_date}</td>
            </tr>
        </table>
        <p style="margin-top: 20px; color: #888; font-size: 12px;">This is an automated message from your AI Task Manager.</p>
    </div>
    """

    payload = {
        "sender": {"name": sender_name, "email": sender_email},
        "to": [{"email": to_email}],
        "subject": f"New Task Assigned: {task_name}",
        "htmlContent": html_content
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 201:
            print(f"✅ Notification sent to {to_email}")
            return True
        else:
            print(f"❌ Email Failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Email Error: {str(e)}")
        return False

# --- HELPER: FETCH TEAM DIRECTORY ---
def get_team_directory():
    """
    Reads the 'Team' tab from Google Sheets and returns a dictionary.
    Format: {'Rahul': 'rahul@example.com', 'Sarah': 'sarah@test.com'}
    """
    try:
        json_creds = os.getenv("GOOGLE_CREDS")
        if not json_creds:
            return {}
            
        creds_dict = json.loads(json_creds)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Open the specific 'Team' worksheet
        sheet = client.open(SHEET_NAME).worksheet("Team") 
        data = sheet.get_all_records()
        
        # Create a dictionary mapping Name -> Email
        # We use lower() to make it case-insensitive matching later
        team_map = {row['Name'].strip().lower(): row['Email'].strip() for row in data}
        
        print(f"✅ Team Directory Loaded: {len(team_map)} members found.")
        return team_map

    except Exception as e:
        print(f"⚠️ Could not load Team Directory: {e}")
        return {}



# --- 3. HELPER: EMAIL SENDER (UPDATED FOR RENDER) 

def internal_send_email(to_email, subject, body, attachment_base64=None, attachment_type="none"):
    api_key = os.getenv("BREVO_API_KEY")
    sender_email = os.getenv("SENDER_EMAIL")
    sender_name = os.getenv("SENDER_NAME", "AI Assistant")

    if not api_key:
        return {"message": "❌ Missing BREVO_API_KEY", "status": "error"}

    url = "https://api.brevo.com/v3/smtp/email"
    headers = {
        "accept": "application/json",
        "api-key": api_key,
        "content-type": "application/json"
    }
    
    payload = {
        "sender": {"name": sender_name, "email": sender_email},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": f"<p>{body}</p>"
    }
    
    # Logic to handle different attachment names
    if attachment_base64:
        filename = "project_status.png"
        if attachment_type == "table":
            filename = "status_table.png"
        elif attachment_type == "chart":
            filename = "status_chart.png"
            
        payload["attachment"] = [
            {
                "content": attachment_base64,
                "name": filename
            }
        ]

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 201:
            return {"message": f"✅ Email sent to {to_email} successfully!", "status": "success"}
        else:
            return {"message": f"❌ Failed: {response.text}", "status": "error"}
    except Exception as e:
        return {"message": f"❌ Error: {str(e)}", "status": "error"}


# --- 4. HELPER: UPDATE TASK ---
def internal_update_task(task_name, field, value):
    sheet = get_google_sheet()
    if not sheet:
        return {"message": "Connection Error", "status": "error"}

    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)

        # Flexible column matching
        col_map = {c.strip().lower().replace("_", " "): c for c in df.columns}
        
        task_col_actual = col_map.get("task name") or col_map.get("taskname") or col_map.get("task")
        if not task_col_actual:
            return {"message": "Could not find 'Task Name' column", "status": "error"}

        target_col_clean = field.strip().lower().replace("_", " ")
        target_col_actual = col_map.get(target_col_clean)
        if not target_col_actual:
            return {"message": f"Column '{field}' not found.", "status": "error"}

        mask = df[task_col_actual].astype(str).str.strip().str.lower() == task_name.strip().lower()
        if not mask.any():
            return {"message": f"Task '{task_name}' not found.", "status": "error"}

        df.loc[mask, target_col_actual] = value
        
        sheet.clear()
        sheet.update([df.columns.values.tolist()] + df.values.tolist())
        load_data_global() # Refresh memory after update
        return {"message": f"✅ Updated '{task_name}': Set '{target_col_actual}' to '{value}'", "status": "success"}

    except Exception as e:
        return {"message": f"Error updating: {str(e)}", "status": "error"}

# --- AUTOMATED SCHEDULER LOGIC ---

def check_deadlines_and_notify():
    """
    1. Reads all tasks.
    2. Checks if 'End Date' is 2 days from now.
    3. Finds Assignee email from Team Directory.
    4. Sends reminder.
    """
    print("⏰ Scheduler running: Checking for upcoming deadlines...")
    
    sheet = get_google_sheet()
    if not sheet: 
        print("❌ Scheduler Error: Can't connect to sheet.")
        return

    # 1. Fetch Data
    tasks = sheet.get_all_records()
    team_directory = get_team_directory() # Uses the function we made in the previous step
    
    today = datetime.now().date()
    
    for row in tasks:
        task_name = row.get("Task Name") or row.get("task_name")
        assigned_to = row.get("Assigned To") or row.get("assigned_to")
        status = row.get("Status") or row.get("status")
        end_date_str = row.get("End Date") or row.get("end_date")
        
        # Skip if already done or data is missing
        if str(status).lower() in ["completed", "done", "cancelled"] or not end_date_str:
            continue

        try:
            # 2. Parse Date (Assumes format YYYY-MM-DD)
            # If your sheet uses DD-MM-YYYY, change this to "%d-%m-%Y"
            due_date = datetime.strptime(str(end_date_str), "%Y-%m-%d").date()
            
            # Calculate days remaining
            days_left = (due_date - today).days
            
            # 3. TRIGGER: If due in exactly 2 days (or overdue)
            if days_left == 2:
                print(f"⚠️ Task '{task_name}' is due in 2 days!")
                
                # 4. Find Email
                assignee_email = team_directory.get(str(assigned_to).strip().lower())
                
                if assignee_email:
                    # 5. Send Email
                    subject = f"🔔 Reminder: '{task_name}' is due soon"
                    body = f"""
                    <h3>Deadline Approaching</h3>
                    <p>Hi {assigned_to},</p>
                    <p>This is a gentle reminder that the task <strong>{task_name}</strong> is due on <strong>{end_date_str}</strong>.</p>
                    <p>Please update the status if completed.</p>
                    """
                    # Reuse your existing email sender
                    internal_send_email(assignee_email, subject, body)
                    print(f"✅ Reminder sent to {assignee_email}")
                else:
                    print(f"⚠️ No email found for user: {assigned_to}")
                    
        except ValueError:
            # Date format was likely wrong in the sheet
            continue


#StartUp Event
# Create the scheduler instance
scheduler = BackgroundScheduler()

@app.on_event("startup")
async def startup_event():
    # 1. Load Data for AI
    load_data_global()
    
    # 2. Start the Scheduler
    # For TESTING: Use 'interval' and seconds=60 to see it work immediately
    # For PRODUCTION: Use 'cron' to run once a day (e.g., at 9 AM)
    
    # --- UNCOMMENT ONE OF THESE ---
    
    # OPTION A: Testing Mode (Runs every 60 seconds)
    # scheduler.add_job(check_deadlines_and_notify, 'interval', seconds=60)
    
    # OPTION B: Production Mode (Runs every day at 09:00 AM UTC)
    scheduler.add_job(check_deadlines_and_notify, 'cron', hour=9, minute=0)
    
    scheduler.start()
    print("🚀 Background Scheduler Started")


@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
#------ Helper for summary genration
def create_simple_summary(text: str) -> str:
    """
    Creates a basic summary by extracting the first two sentences.
    Useful for quick logic without heavy NLP dependencies.
    """
    if not text:
        return ""

    # Split text roughly by sentences (looking for period + space)
    sentences = text.split('. ')
    
    # If the text is short (less than 3 sentences), just return the whole thing
    if len(sentences) < 3:
        return text

    # Otherwise, join the first two sentences to create a 'preview' summary
    summary = '. '.join(sentences[:2]) + '.'
    return summary


#function to parse tasks from command
def parse_task_from_command(command_text: str):
    """
    Parses natural language text into a dictionary compatible with TaskRequest.
    Example Input: "Add task Buy Milk start date 2023-10-01"
    """
    command_text = command_text.lower()
    
    # Basic logic to check if this is an 'add task' command
    if "add task" in command_text:
        # Remove the trigger phrase and split by comma or key phrases
        # This logic splits by commas for simplicity
        clean_text = command_text.replace("add task", "").strip()
        parts = clean_text.split(",")
        
        task_info = {
            "client": "",      # Defaults
            "notify_email": "" # Defaults
        }
        
        # Simple keyword parsing
        for part in parts:
            part = part.strip()
            if "start date" in part:
                task_info['start_date'] = part.split("start date")[-1].strip()
            elif "end date" in part:
                task_info['end_date'] = part.split("end date")[-1].strip()
            elif "assigned to" in part:
                task_info['assigned_to'] = part.split("assigned to")[-1].strip()
            elif "status" in part:
                task_info['status'] = part.split("status")[-1].strip()
            elif part:
                # Assume the first non-keyword chunk is the task name
                if 'task_name' not in task_info:
                    task_info['task_name'] = part
        
        return task_info
        
    return None

# Helper function to generate summary
def generate_ai_summary(text: str) -> str:
    try:
        # Initialize the model (Adjust temperature for creativity vs precision)
        llm = ChatOpenAI(
            model="gpt-4", 
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.5
        )
        # Define the instructions for the AI
        messages = [
            SystemMessage(content="You are a helpful AI assistant. Summarize the following text clearly and concisely in bullet points."),
            HumanMessage(content=text)
        ]
        # Get response
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Summary could not be generated at this time."


# --- 6. API ENDPOINTS ---

# Fixes 404 Error
@app.get("/")
def read_root():
    return {"status": "active", "message": "Backend is running. Data loaded: " + str(document_loaded)}

#End Point to test Team
# @app.get("/api/test-team")
# def test_team():
#    directory = get_team_directory()
#    return {"message": "Directory loaded", "data": directory}

@app.get("/api/status")
def get_status():
    return {"document_loaded": document_loaded, "data_preview": excel_text_context[:100]}

@app.post("/api/add-task")
def add_task(task: TaskRequest):
    sheet = get_google_sheet()
    if not sheet:
        return {"message": "Database connection failed", "status": "error"}
    try:
        # Append the new row
        # Ensure the order matches your Google Sheet columns!
        new_row = [
            task.task_name, 
            task.start_date, 
            task.end_date,
            task.status,
            task.assigned_to,
            task.client
        ]
        
        sheet.append_row(new_row)
        
        # Refresh the global data cache so the AI knows about the new task
        load_data_global()
        
         # --- NEW: EMAIL LOGIC ---
        email_status = ""
        if task.notify_email:
            # Only try to send if the email field is not empty
            sent = send_task_creation_email(
                to_email=task.notify_email,
                task_name=task.task_name,
                assigned_to=task.assigned_to,
                client=task.client,
                due_date=task.end_date
            )
            if sent:
                email_status = " Email notification sent!"
            else:
                email_status = " (Email notification failed to send)."
        return {
            "message": f"Task '{task.task_name}' added successfully!{email_status}", 
            "status": "success"
        }
    except Exception as e:
        return {"message": f"Failed to add task: {str(e)}", "status": "error"}

# ----- API function for speech recongnition
@app.post("/api/voice")
async def process_audio(audio: UploadFile = File(...)):
   try:
        # 1. Read Audio File
        audio_bytes = await audio.read()
        
        # 2. Convert to WAV using PyDub (Handles various formats like mp3, ogg, etc.)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        # 3. Transcribe using SpeechRecognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_buffer) as source:
            audio_data = recognizer.record(source)
            # Using Open AI Speech API (default key)
            transcribed_text = recognizer.recognize_openai(audio_data)
            print(f"DEBUG - User said: {transcribed_text}")

        # 1. SAVE & TRANSCRIBE AUDIO
        # (Your existing code to save the file and run the transcription goes here)
        # For this example, let's assume the result is stored in 'transcribed_text'
        
        # Example placeholder:
        # transcribed_text = transcribe_function(saved_file_path) 
        
        # If you are testing without real audio logic yet, you can uncomment this:
        #transcribed_text = "This is a test recording. We are ignoring dates for now. We just want a summary."
        # 2. GENERATE SUMMARY
        # Pass the text to our helper function
        #summary_text = create_simple_summary(transcribed_text)
        
        #Ai Summary 
        ai_summary = generate_ai_summary(transcribed_text)
        # 3. Return JSON response
        return {
            "status": "success",
            "transcription": transcribed_text,
            "summary": ai_summary
        }
        print(f"Transcription: {transcribed_text}")
        print(f"Summary: {ai_summary}")
        
        # 3. RETURN RESPONSE
        #return {
        #    "status": "success",
        #    "transcription": transcribed_text,
        #    "summary": summary_text
        #}
   except Exception as e:
        print(f"Error processing audio: {e}")
        return {"status": "error", "message": str(e)}


# --- 7. LANGCHAIN TOOLS ---

@tool
def update_sheet_tool(task_name: str, field: str, value: str):
    """
    Updates a task in the Google Sheet. 
    Use this tool when the user asks to modify, update, change, or set a value in the tracker.
    """
    print(f"🛠 Tool Triggered: Updating {task_name}...")
    result = internal_update_task(task_name, field, value)
    return result["message"]

@tool
def send_email_tool(to_email: str, subject: str, body: str, attachment_type: str = "none"):
    """
    Sends an email.
    IMPORTANT: 'attachment_type' must be one of: 'chart', 'table', or 'none'.
    - Use 'chart' if user asks for a visualization or graph.
    - Use 'table' if user asks for a list, grid, or table in the email.
    - Use 'none' for standard text emails.
    """
    print(f"📧 Tool Triggered: Sending email to {to_email} with {attachment_type}...")
    
    attachment_data = None
    
    # Decide what to generate based on the AI's request
    if attachment_type.lower() == "chart":
        attachment_data = generate_chart_base64()
    elif attachment_type.lower() == "table":
        attachment_data = generate_table_base64()
    
    # Send the email once
    result = internal_send_email(to_email, subject, body, attachment_data, attachment_type)
    return result["message"]


# --- 8. CHAT AGENT (UPDATED) ---

@app.post("/api/chat")
def chat(request: PromptRequest):
    global excel_text_context
    
    try:
        # 1. Reload data context if missing
        if not document_loaded or not excel_text_context:
            load_data_global()

        # 2. Define Tools
        tools = [update_sheet_tool, send_email_tool]
        tool_map = {
            "update_sheet_tool": update_sheet_tool,
            "send_email_tool": send_email_tool
        }

        # 3. Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o", 
            openai_api_key=openai_key,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(tools)

        # 4. System Prompt
        # NOTE: Double braces {{ }} are used here to escape JSON inside the f-string
        system_msg = f"""
        You are an advanced Project Manager Agent.
        
        CURRENT DATA CONTEXT:
        {excel_text_context}
        
        YOUR TOOLS:
        1. 'update_sheet_tool': Modify data.
        2. 'send_email_tool': Send emails. 
           - PARAMETER 'attachment_type': Set this to 'chart', 'table', or 'none' strictly based on user request.
        
        INSTRUCTIONS:
        - If the user says "Add task [task_name]", call the function responsible for adding a task (e.g., via 'add_task' configured for additions).
        - If the user says "Update task [task_name]", call the function responsible for updating a task (e.g., via 'update_sheet_tool' configured for updates).
        - If the user says "Send email with a CHART", call 'send_email_tool' with attachment_type='chart'.
        - If the user says "Send email with a TABLE", call 'send_email_tool' with attachment_type='table'.
        - If the user says "Send email", use attachment_type='none'.
        - Do NOT call the tool twice.
        - Answer general questions normally.
        - Critical if the users asks to "create action item" or "Add Task", NOT just reply with text. Instead, output a JSON block strictly following this format:  

        FORMAT FOR TASK ADDITION (Output this JSON strictly):
        ```json
        {{
            "action": "add",
            "task_name": "Task Name",
            "assigned_to": "Assignee Name",
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD",
            "status": "Not Started",
            "client": "Client Name",
            "notify_email": "email@example.com"
        }}
        ```

        FORMAT FOR CHART (For Chat Display Only):
        ```json
        {{ "is_chart": true, "chart_type": "bar", "title": "Tasks by Status", "data": {{ "labels": ["Done", "Pending"], "values": [5, 2] }}, "summary": "Here is the chart." }}
        ```

        FORMAT FOR TABLE (For Chat Display Only):
        ```json
        {{
            "is_table": true,
            "title": "Task Overview",
            "headers": ["Task Name", "Status", "Due Date"],
            "rows": [
                ["Fix Bug", "Done", "2023-10-01"],
                ["Write Docs", "Pending", "2023-10-05"]
            ],
            "summary": "Here is the table you requested."
        }}
        ```
        """

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=request.prompt)
        ]

        print("🤖 AI Thinking...")
        ai_response = llm_with_tools.invoke(messages)

        # --- CASE A: TOOL CALLS (LangChain Tools) ---
        if ai_response.tool_calls:
            print(f"🔧 AI decided to use tools: {len(ai_response.tool_calls)}")
            results = []
            
            for tool_call in ai_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                if tool_name in tool_map:
                    print(f"   -> Executing {tool_name} with args: {tool_args}")
                    tool_output = tool_map[tool_name].invoke(tool_args)
                    results.append(str(tool_output))
                else:
                    results.append(f"Error: Tool {tool_name} not found.")

            # FIXED: Closed the dictionary properly
            return {
                "response": " | ".join(results),
                "type": "text",
                "status": "success"
            }

        # --- CASE B: JSON ACTIONS (Visuals & Add Task) ---
        # Get content cleanly first
        content = ai_response.content.strip()

        if "```json" in content:
            try:
                # Extract clean JSON string
                # FIXED: split() logic adjusted
                clean_json = content.split("```json")[1].split("```")[0].strip()
                data_obj = json.loads(clean_json)
                
                # 1. Handle Charts
                if data_obj.get("is_chart"):
                    return {
                        "response": data_obj.get("summary", "Here is the chart."), 
                        "chart_data": data_obj, 
                        "type": "chart", 
                        "status": "success"
                    }
                
                # 2. Handle Tables
                if data_obj.get("is_table"):
                    return {
                        "response": data_obj.get("summary", "Here is the table."), 
                        "table_data": data_obj, 
                        "type": "table", 
                        "status": "success"
                    }
                
                # 3. Handle Task Addition
                if data_obj.get("action") == "add":
                    print("📝 AI requesting to ADD a new task...")
                    
                    # Extract task details from AI response
                    task_payload = {
                        "task_name": data_obj.get("task_name"),
                        "assigned_to": data_obj.get("assigned_to", "Unassigned"),
                        "start_date": data_obj.get("start_date", ""),
                        "end_date": data_obj.get("end_date", ""),
                        "status": data_obj.get("status", "Not Started"),
                        "client": data_obj.get("client", ""),
                        "notify_email": data_obj.get("notify_email", None)
                    }

                    # Call internal API endpoint
                    api_url = "https://web-production-b8ca4.up.railway.app/api/add-task"
                    
                    sheet_response = requests.post(api_url, json=task_payload)
                    
                    if sheet_response.status_code == 200:
                        return {
                            "response": f"✅ Task '{task_payload['task_name']}' has been successfully added to the Sheet.",
                            "type": "text",
                            "status": "success"
                        }
                    else:
                        return {
                            "response": f"❌ Failed to add task. Server replied: {sheet_response.text}",
                            "type": "text",
                            "status": "error"
                        }

            except json.JSONDecodeError:
                print("⚠️ Failed to parse JSON from AI response.")
            except Exception as e:
                print(f"⚠️ Error processing JSON action: {e}")

        # --- CASE C: STANDARD TEXT RESPONSE ---
        return {
            "response": content,
            "type": "text",
            "status": "success"
        }

    except Exception as e:
        print(f"❌ Chat Error: {e}")
        return {"response": f"Error: {str(e)}", "status": "error"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
