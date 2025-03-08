from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import json
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import traceback
import requests
import sys
from groq import Groq

app = Flask(__name__)
# Update CORS configuration to allow requests from your Netlify domain
CORS(app, origins=["https://dws-medicare.netlify.app", "http://localhost:3000"])

# Set Groq API key - in production, this should be set as an environment variable
GROQ_API_KEY = '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'
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

print(f"Setting Groq API Key: {GROQ_API_KEY[:5]}...")

# Initialize Groq client
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Groq client initialized successfully")
except Exception as e:
    print(f"Error initializing Groq client: {str(e)}")
    groq_client = None

# Function to call Groq API
def call_groq_api(messages, model="llama-3.3-70b-versatile"):
    """Call Groq API for chat completions"""
    try:
        if not groq_client:
            raise ValueError("Groq client not properly initialized")
            
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=800
        )
        
        # Extract and return the response content
        if chat_completion and chat_completion.choices and len(chat_completion.choices) > 0:
            return {"text": chat_completion.choices[0].message.content}
        else:
            print("Empty or invalid response from Groq API")
            return None
    except Exception as e:
        print(f"Error calling Groq API: {str(e)}")
        print(traceback.format_exc())
        return None

# Test the Groq API
def test_api_connectivity():
    """Test if the Groq API is working properly"""
    try:
        print("\nTesting Groq API connectivity...")
        test_message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        result = call_groq_api(test_message)
        
        print(f"Groq API test result: {result}")
        if result and isinstance(result, dict) and 'text' in result and result['text']:
            print("✅ Groq API is working properly!")
            return True
        else:
            print("❌ Groq API returned an unexpected response")
            return False
    except Exception as e:
        print(f"❌ Groq API test failed with error: {str(e)}")
        print(traceback.format_exc())
        return False

# Run the API connectivity test during initialization
api_working = test_api_connectivity()

# Alternative fallback responses
def call_fallback_responses(messages):
    """Generate rule-based fallback responses"""
    try:
        # Check if we're asking about specific health conditions
        user_message = ""
        for msg in messages:
            if msg["role"] == "user":
                user_message = msg["content"].lower()
                break
        
        # Generate slightly more targeted responses based on keywords
        if "headache" in user_message or "migraine" in user_message:
            return {
                "text": "I understand you're concerned about headaches. Headaches can have various causes including stress, dehydration, eye strain, or more serious conditions. To provide more helpful information, I'd need to know: How long have you been experiencing these headaches? How would you rate the pain on a scale of 1-10? And are there any specific triggers you've noticed?"
            }
        elif "stomach" in user_message or "digest" in user_message or "nausea" in user_message:
            return {
                "text": "I see you have concerns about digestive issues. These can range from temporary discomfort to more persistent conditions. To better understand your situation, could you tell me: When did these symptoms start? Do they occur after meals or at specific times? Have you noticed any foods that seem to trigger or worsen symptoms?"
            }
        elif "acid reflux" in user_message or "heartburn" in user_message:
            return {
                "text": "I see you're asking about acid reflux or heartburn. This occurs when stomach acid flows back into the esophagus, causing irritation. To provide more tailored information, I'd need to know: How often do you experience these symptoms? Do they occur at specific times (like after meals or when lying down)? And have you tried any remedies so far?"
            }
        elif "depression" in user_message or "anxiety" in user_message or "mental health" in user_message:
            return {
                "text": "Thank you for bringing up this important mental health concern. Mental health conditions are common and treatable. To better understand your situation, could you share: How long have you been experiencing these feelings? Are there particular situations that seem to trigger or worsen them? Have you spoken with a healthcare provider about these concerns before?"
            }
        
        # Default response
        return {
            "text": "I understand your health concern. To provide more specific guidance, I'd need more details about your symptoms, how long you've been experiencing them, and any factors that seem to make them better or worse. Could you provide this information?"
        }
    except Exception as e:
        print(f"Error in fallback response generation: {str(e)}")
        return {
            "text": "I understand you have a health question. To provide the most helpful information, I'd need more details about your specific symptoms and concerns. Could you tell me more about what you're experiencing?"
        }

# In a real app, you would use a database
users_db = {}
sessions = {}
conversation_history = {}  # Store conversation history for each session
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Add a default user for testing
users_db = {
    "test": {"password": "test123"}
}

# Authentication functions - shared between both features
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    
    if username in users_db:
        return jsonify({"error": "User already exists"}), 400
    
    users_db[username] = {"password": password}
    return jsonify({"message": "User registered successfully"}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    
    if username not in users_db or users_db[username]["password"] != password:
        return jsonify({"error": "Invalid credentials"}), 401
    
    # Create session
    session_id = str(uuid.uuid4())
    sessions[session_id] = username
    # Initialize conversation history for this session
    conversation_history[session_id] = []
    
    print(f"Login successful for user: {username}, session: {session_id}")
    print(f"Active sessions: {sessions}")
    
    return jsonify({"message": "Login successful", "token": session_id}), 200


#############################################################################
# GENERAL QUERY MODULE - Everything related to the chat functionality
#############################################################################

class GeneralQueryHandler:
    @staticmethod
    def verify_token(token):
        """Verify token is valid"""
        if not token or token not in sessions:
            return False
        return True

    @staticmethod
    def handle_chat(token, user_question):
        """Process a chat message and return response using AI"""
        # Get conversation history or initialize if empty
        session_history = conversation_history.get(token, [])
        
        # Check if this is a structured response from the interactive UI
        is_structured_response = False
        structured_answers = {}
        
        # Look for structured answer pattern: "Question: Answer"
        if "\n\n" in user_question:
            sections = user_question.split("\n\n")
            is_structured_response = True
            
            # Parse each question-answer pair
            for section in sections:
                if ":" in section:
                    q_part, a_part = section.split(":", 1)
                    structured_answers[q_part.strip()] = a_part.strip()
        
        # Format conversation history for the AI model
        formatted_history = []
        
        # Add up to 5 previous exchanges for context (limiting to reduce token usage)
        context_limit = min(len(session_history), 5)
        if context_limit > 0:
            for i in range(context_limit):
                entry = session_history[len(session_history) - context_limit + i]
                role = "user" if entry["type"] == "question" else "assistant"
                formatted_history.append({"role": role, "content": entry["content"]})
            
        # Create comprehensive system prompt
        system_prompt = """You are an intelligent, empathetic medical assistant that helps users understand health concerns and provides educational information. 
        
        Guidelines:
        1. Be informative, clear, and compassionate in your responses
        2. Acknowledge user concerns and reference specific symptoms they mention
        3. Provide relevant educational content about conditions, treatments, and preventive measures
        4. When users provide structured answers to questions, acknowledge each specific answer they've given
        5. Ask focused follow-up questions to gather more information when needed
        6. If enough information is present, provide a thoughtful analysis but avoid definitive diagnosis
        7. Suggest self-care measures or lifestyle adjustments when appropriate
        8. Remind users that you're providing information, not medical advice
        9. Respond in a structured, easy-to-read format with clear sections
        10. NEVER repeat generic questions that have already been asked and answered
        
        IMPORTANT: If the user has provided answers to specific questions you asked, address those answers directly and move the conversation forward.
        """
        
        if is_structured_response:
            # Add additional context for structured responses
            system_prompt += "\nThe user has provided structured answers to your previous questions in the format 'Question: Answer'. Respond specifically to these details rather than asking the same questions again."
            
            # Add the specific answers to the prompt to ensure they're addressed
            system_prompt += "\n\nThe user's answers include: "
            for q, a in structured_answers.items():
                system_prompt += f"\n- {q}: {a}"
        
        # Build conversation for the AI model
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add conversation history for context
        if formatted_history:
            messages.extend(formatted_history)
        
        # Add the current user question
        messages.append({"role": "user", "content": user_question})
        
        # Try different AI options in sequence, falling back as needed
        try:
            # First attempt: Try Groq API call if it's working
            if api_working:
                try:
                    print("Attempting to use Groq API for response generation...")
                    
                    # Make the API call
                    response = call_groq_api(messages)
                    
                    # Verify we got a valid response
                    print(f"Groq API response: {response}")
                    
                    if response and isinstance(response, dict) and 'text' in response and response['text']:
                        answer = response['text']
                        print(f"Groq API response length: {len(answer)}")
                        print(f"Response preview: {answer[:100]}...")
                        
                        # Update conversation history
                        session_history.append({"type": "question", "content": user_question})
                        session_history.append({"type": "answer", "content": answer})
                        conversation_history[token] = session_history
                        
                        return {"message": answer, "is_fallback": False}
                    else:
                        raise ValueError(f"Invalid Groq API response format: {response}")
                        
                except Exception as e:
                    print(f"Groq API call failed: {str(e)}")
                    print(traceback.format_exc())
                    # Continue to fallback option
            else:
                print("Groq API not available or not working, skipping...")
                
            # Fallback to rule-based responses
            print("Using fallback response system...")
            fallback_response = call_fallback_responses(messages)
            
            if fallback_response and 'text' in fallback_response:
                answer = fallback_response['text']
                print(f"Fallback response length: {len(answer)}")
                print(f"Response preview: {answer[:100]}...")
                
                # Update conversation history
                session_history.append({"type": "question", "content": user_question})
                session_history.append({"type": "answer", "content": answer})
                conversation_history[token] = session_history
                
                return {"message": answer, "is_fallback": True}
            else:
                # Structure-specific fallback if we have structured answers from the UI
                if is_structured_response:
                    fallback_answer = GeneralQueryHandler.generate_ai_fallback_for_structured_data(structured_answers, user_question)
                else:
                    fallback_answer = GeneralQueryHandler.generate_ai_fallback_for_freetext(user_question, session_history)
                
                # Update conversation history
                session_history.append({"type": "question", "content": user_question})
                session_history.append({"type": "answer", "content": fallback_answer})
                conversation_history[token] = session_history
                
                return {"message": fallback_answer, "is_fallback": True}
                
        except Exception as e:
            print(f"Unhandled error in handle_chat: {str(e)}")
            print(traceback.format_exc())
            
            # Ultimate fallback response
            generic_response = "I understand your health concern. To provide more specific guidance, I'd need more details about your symptoms, how long you've been experiencing them, and any factors that seem to make them better or worse. Could you provide this information?"
            
            # Still update conversation history
            session_history.append({"type": "question", "content": user_question})
            session_history.append({"type": "answer", "content": generic_response})
            conversation_history[token] = session_history
            
            return {"message": generic_response, "is_fallback": True, "error": True}

    @staticmethod
    def generate_ai_fallback_for_structured_data(structured_answers, user_question):
        """Generate a fallback response when the AI model fails but we have structured data"""
        
        # Extract symptom types from the questions
        symptom_types = set()
        for question in structured_answers.keys():
            question_lower = question.lower()
            
            # Identify common health conditions from the questions
            if any(term in question_lower for term in ['headache', 'migraine', 'head pain']):
                symptom_types.add("headache")
            elif any(term in question_lower for term in ['stomach', 'digest', 'nausea', 'vomit']):
                symptom_types.add("digestive")
            elif any(term in question.lower() for term in ['cough', 'breath', 'lung']):
                symptom_types.add("respiratory")
            elif any(term in question.lower() for term in ['skin', 'rash', 'itch']):
                symptom_types.add("skin")
            elif any(term in question.lower() for term in ['eye', 'vision', 'sight']):
                symptom_types.add("eye")
            elif any(term in question.lower() for term in ['fever', 'temperature']):
                symptom_types.add("fever")
            elif any(term in question.lower() for term in ['anxiety', 'depress', 'mental']):
                symptom_types.add("mental health")
            elif any(term in question.lower() for term in ['joint', 'muscle', 'pain']):
                symptom_types.add("musculoskeletal")
            elif any(term in question.lower() for term in ['heart', 'chest', 'cardiac']):
                symptom_types.add("cardiac")
        
        # Create a personalized response based on the information provided
        response = "Thank you for providing those details about your health concerns. "
        
        if symptom_types:
            symptom_list = ", ".join(symptom_types)
            response += f"Based on the information you've shared about your {symptom_list} symptoms, "
        else:
            response += "Based on what you've shared, "
            
        response += "I can provide some general information, though consulting with a healthcare provider would be best for specific advice tailored to your situation.\n\n"
        
        # Add duration-based information if available
        duration_info = None
        for question, answer in structured_answers.items():
            if any(term in question.lower() for term in ['long', 'duration', 'start', 'began', 'when']):
                duration_info = answer
                break
                
        if duration_info:
            if any(term in duration_info.lower() for term in ['day', 'today', 'recent', 'just']):
                response += "Since your symptoms are recent, monitoring them and seeing if they resolve with basic self-care would be reasonable. However, if they worsen or persist, please seek medical attention.\n\n"
            elif any(term in duration_info.lower() for term in ['week', 'month', 'year', 'chronic']):
                response += "Since you've been experiencing these symptoms for some time, it would be advisable to consult with a healthcare provider for proper evaluation and treatment.\n\n"
                
        # Add symptom severity information if available
        severity_info = None
        for question, answer in structured_answers.items():
            if any(term in question.lower() for term in ['severe', 'intensity', 'pain level', 'rate']):
                severity_info = answer
                break
                
        if severity_info and any(term in severity_info.lower() for term in ['severe', 'high', '8', '9', '10']):
            response += "The severity you've described is concerning. Please consider seeking prompt medical attention, especially if accompanied by other concerning symptoms.\n\n"
            
        # Conclude with general advice
        response += "Would you like more specific information about managing your symptoms, or do you have other health concerns you'd like to discuss?"
        
        return response

    @staticmethod
    def generate_ai_fallback_for_freetext(user_question, session_history):
        """Generate a fallback response for free text questions when AI model fails"""
        user_question_lower = user_question.lower()
        
        # Extract potentially mentioned health concerns
        health_concerns = []
        
        health_topics = {
            "headache": ["headache", "migraine", "head pain", "head ache"],
            "stomach": ["stomach", "abdomen", "digest", "nausea", "vomit", "diarrhea"],
            "heart": ["heart", "chest pain", "palpitation", "cardiac"],
            "respiratory": ["cough", "breathing", "short of breath", "lung", "respiratory"],
            "skin": ["skin", "rash", "itch", "hives", "eczema"],
            "eye": ["eye", "vision", "sight", "blind", "blurry"],
            "mental health": ["anxiety", "depression", "stress", "mental health", "mood"],
            "fever": ["fever", "temperature", "chills"],
            "pain": ["pain", "ache", "sore", "hurt"],
            "sleep": ["sleep", "insomnia", "tired", "fatigue"],
            "injury": ["injury", "sprain", "strain", "broken", "fracture"]
        }
        
        # Check which health topics are mentioned
        for topic, keywords in health_topics.items():
            if any(keyword in user_question_lower for keyword in keywords):
                health_concerns.append(topic)
                
        # Create base response
        if health_concerns:
            response = f"I understand you're asking about {', '.join(health_concerns)}. "
            
            # Check if it seems to be a question about causes
            if any(term in user_question_lower for term in ["cause", "why", "what makes", "reason"]):
                response += "To properly address questions about causes, I'd need more specific information about your symptoms. "
                
            # Check if it seems to be a question about treatment
            elif any(term in user_question_lower for term in ["treat", "cure", "help", "relieve", "remedy"]):
                response += "Regarding treatment options, it's important to first understand more about your specific situation. "
                
            # Check if it seems to be a question about prevention
            elif any(term in user_question_lower for term in ["prevent", "avoid", "stop", "reduce risk"]):
                response += "Prevention strategies are important, but they work best when tailored to your specific circumstances. "
                
            response += "To provide you with more specific and helpful information, could you tell me:\n\n"
            response += "1. How long have you been experiencing these symptoms?\n"
            response += "2. Do you notice any particular patterns or triggers?\n"
            response += "3. Have you tried any remedies or treatments so far?\n\n"
            response += "This information will help me provide more relevant guidance for your situation."
                
        else:
            # Generic response for unclear health queries
            response = "Thank you for your health question. To provide you with the most helpful information, I need to understand more about your situation. Could you please share:\n\n"
            response += "1. What specific symptoms or health concerns are you experiencing?\n"
            response += "2. How long have these been occurring?\n"
            response += "3. Is there anything that seems to trigger or worsen them?\n\n"
            response += "With more details, I can offer more relevant information for your situation."
            
        return response

@app.route('/api/chat', methods=['POST'])
def chat():
    token = request.headers.get('Authorization')
    print(f"Received chat request with token: {token}")
    
    if not GeneralQueryHandler.verify_token(token):
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    user_question = data.get('message')
    
    if not user_question or not user_question.strip():
        return jsonify({"error": "Empty message"}), 400
    
    try:
        # First check if the token exists in the conversation history dictionary
        # If it doesn't, initialize it with an empty list
        if token not in conversation_history:
            print(f"Initializing new conversation history for token: {token}")
            conversation_history[token] = []
            
        # Check for duplicate questions to prevent conversation loops
        if conversation_history[token] and len(conversation_history[token]) >= 2:
            last_question = None
            for entry in reversed(conversation_history[token]):
                if entry["type"] == "question":
                    last_question = entry["content"]
                    break
                    
            if last_question and last_question == user_question:
                print("Detected duplicate question. Generating variation of previous response")
                # Find the last bot answer
                for entry in reversed(conversation_history[token]):
                    if entry["type"] == "answer":
                        alternative_response = "I've already addressed this question. Is there something specific about my response that you'd like me to clarify further?"
                        return jsonify({"message": alternative_response, "is_fallback": False})
        
        result = GeneralQueryHandler.handle_chat(token, user_question)
        
        # Double-check that we have a valid response
        if not result or not isinstance(result, dict) or 'message' not in result:
            raise ValueError("Invalid response structure")
        
        # Ensure we have a non-empty message
        if not result['message'] or len(result['message'].strip()) < 10:
            result['message'] = "I understand your question. Could you provide more details about your symptoms so I can better assist you?"
        
        # Log the conversation to help debugging
        print(f"Conversation history for token {token} now has {len(conversation_history.get(token, []))} entries")
        
        return jsonify(result)
    except Exception as e:
        print(f"Unhandled error in chat endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "message": "I understand your question but am having trouble processing it right now. Could you try rephrasing or asking a different health question?",
            "is_fallback": True,
            "error": True  # Flag to indicate error state to the frontend
        }), 200  # Still return 200 to allow frontend to handle gracefully


#############################################################################
# MEDICAL REPORT MODULE - Everything related to report analysis
#############################################################################

class MedicalReportHandler:
    @staticmethod
    def verify_token(token):
        """Verify token is valid"""
        if not token or token not in sessions:
            return False
        return True
        
    @staticmethod
    def extract_text_from_file(file):
        """Extract text content from various file formats"""
        filename = file.filename
        file_extension = filename.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        elif file_extension in ['doc', 'docx']:
            doc = docx.Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        elif file_extension in ['txt']:
            return file.read().decode('utf-8')
        else:
            return None

    @staticmethod
    def generate_report_analysis(extracted_text, filename):
        """Generate a structured analysis of a medical report when ML services fail"""
        # Convert text to lowercase for easier matching
        text_lower = extracted_text.lower()
        
        # Prepare a response structure
        analysis = f"## Medical Report Analysis: {filename}\n\n"
        
        # Try to detect the type of report
        report_type = "Unknown"
        if any(term in text_lower for term in ["cbc", "complete blood count", "wbc", "rbc", "hemoglobin", "hematocrit"]):
            report_type = "Blood Test"
        elif any(term in text_lower for term in ["urine", "urinalysis"]):
            report_type = "Urinalysis"
        elif any(term in text_lower for term in ["glucose", "hba1c", "blood sugar"]):
            report_type = "Glucose/Diabetes Test"
        elif any(term in text_lower for term in ["cholesterol", "hdl", "ldl", "triglycerides", "lipid"]):
            report_type = "Lipid Panel"
        elif any(term in text_lower for term in ["x-ray", "xray", "radiograph"]):
            report_type = "X-Ray Report"
        elif any(term in text_lower for term in ["mri", "magnetic resonance"]):
            report_type = "MRI Report"
        elif any(term in text_lower for term in ["ct scan", "cat scan"]):
            report_type = "CT Scan"
        elif any(term in text_lower for term in ["ultrasound", "sonogram", "doppler"]):
            report_type = "Ultrasound"
        elif any(term in text_lower for term in ["ecg", "ekg", "electrocardiogram"]):
            report_type = "ECG/EKG"
        
        analysis += f"**Report Type**: {report_type}\n\n"
        
        # Look for common patterns in medical reports
        # 1. Find values with units and compare to normal ranges if possible
        values_found = []
        # Check for common test patterns like "Test: Value Unit (Range)"
        import re
        
        # Look for these common tests and extract values
        common_tests = {
            # Blood tests
            "hemoglobin": {"unit": "g/dL", "normal": "12-16 g/dL (females), 13.5-17.5 g/dL (males)"},
            "hematocrit": {"unit": "%", "normal": "36-48% (females), 41-50% (males)"},
            "rbc": {"unit": "million/μL", "normal": "4.2-5.4 million/μL (females), 4.7-6.1 million/μL (males)"},
            "wbc": {"unit": "cells/μL", "normal": "4,500-11,000 cells/μL"},
            "platelets": {"unit": "/μL", "normal": "150,000-450,000/μL"},
            
            # Lipid panel
            "cholesterol": {"unit": "mg/dL", "normal": "<200 mg/dL"},
            "ldl": {"unit": "mg/dL", "normal": "<100 mg/dL"},
            "hdl": {"unit": "mg/dL", "normal": ">40 mg/dL (males), >50 mg/dL (females)"},
            "triglycerides": {"unit": "mg/dL", "normal": "<150 mg/dL"},
            
            # Liver function
            "alt": {"unit": "U/L", "normal": "7-56 U/L"},
            "ast": {"unit": "U/L", "normal": "5-40 U/L"},
            
            # Kidney function
            "creatinine": {"unit": "mg/dL", "normal": "0.6-1.2 mg/dL (males), 0.5-1.1 mg/dL (females)"},
            "bun": {"unit": "mg/dL", "normal": "7-20 mg/dL"},
            "egfr": {"unit": "mL/min", "normal": ">60 mL/min"},
            
            # Glucose
            "glucose": {"unit": "mg/dL", "normal": "70-99 mg/dL (fasting)"},
            "hba1c": {"unit": "%", "normal": "< 5.7%"},
            
            # Thyroid
            "tsh": {"unit": "μIU/mL", "normal": "0.4-4.0 μIU/mL"},
            "t4": {"unit": "μg/dL", "normal": "4.5-12 μg/dL"},
            "t3": {"unit": "ng/dL", "normal": "80-200 ng/dL"}
        }
        
        # Try to extract test results for common tests
        for test, info in common_tests.items():
            # Look for test name followed by numbers
            pattern = rf'{test}\D*(\d+\.?\d*)'
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                values_found.append(f"**{test.upper()}**: {matches[0]} {info['unit']} (Normal range: {info['normal']})")
        
        if values_found:
            analysis += "### Detected Values:\n"
            for value in values_found:
                analysis += f"- {value}\n"
            analysis += "\n"
        
        # Look for terms indicating abnormality
        abnormal_terms = ["abnormal", "high", "low", "elevated", "decreased", "positive", "negative", 
                           "out of range", "reference range", "critical"]
        
        abnormal_findings = []
        for term in abnormal_terms:
            # Find instances where abnormal terms appear
            pattern = rf'(\w+\s*\w*)\s*(?:is|was|were|appears?|shows?)\s*{term}'
            matches = re.findall(pattern, text_lower)
            abnormal_findings.extend(matches)
        
        if abnormal_findings:
            analysis += "### Potential Abnormal Findings:\n"
            for finding in set(abnormal_findings):  # Using set to remove duplicates
                analysis += f"- {finding.strip().capitalize()}\n"
            analysis += "\n"
        
        # Add recommendations
        analysis += "### Recommendations:\n"
        analysis += "1. **Consult with your healthcare provider**: This automated analysis is not a replacement for professional medical interpretation.\n"
        analysis += "2. **Review with a specialist**: Have these results reviewed by an appropriate medical specialist.\n"
        analysis += "3. **Follow-up testing**: Your doctor may recommend additional tests based on these results.\n\n"
        
        # Add disclaimer
        analysis += "### Disclaimer:\n"
        analysis += "This analysis is generated by an automated system with limited capabilities. It may miss important findings or incorrectly identify normal results as abnormal. Always consult with a qualified healthcare professional for accurate interpretation of medical reports."
        
        return analysis

    @staticmethod
    def analyze_report(token, file):
        """Process a report analysis request and return response"""
        if not file or file.filename == '':
            return {"error": "No file selected"}, 400
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text from the file
        file.seek(0)  # Reset file pointer to beginning
        try:
            extracted_text = MedicalReportHandler.extract_text_from_file(file)
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            print(traceback.format_exc())
            return {"error": f"Error extracting text from file: {str(e)}"}, 400
        
        if not extracted_text:
            return {"error": "Could not extract text from file"}, 400
        
        # System prompt for report analysis
        system_content = """You are a medical assistant analyzing medical reports. Follow these guidelines:
        1. Identify key metrics and test results
        2. Compare results with normal ranges when available
        3. Highlight any abnormal findings
        4. Organize information in a structured way
        5. Avoid making definitive diagnostic statements
        6. Always remind that this is not a substitute for professional medical interpretation
        7. Be factual and objective in your analysis
        """
        
        # Build the prompt for the model
        conversation_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Here's a medical report to analyze: {extracted_text}\n\nPlease analyze this report and provide a summary of key findings."}
        ]
        
        # Get response from Groq or use rule-based fallback
        try:
            # First try Groq
            try:
                print("Sending report analysis request to Groq...")
                response = call_groq_api(conversation_messages)
                
                print(f"Groq report analysis response: {response}")
                
                # Check if we got a valid response
                if response and isinstance(response, dict) and 'text' in response and response['text']:
                    analysis = response['text']
                    return {"analysis": analysis, "is_fallback": False}, 200
                else:
                    # Empty or invalid response
                    raise ValueError("Empty or invalid Groq response")
                    
            except Exception as e:
                print(f"Groq API failed, using rule-based analysis: {str(e)}")
                # Use rule-based analysis as fallback
                analysis = MedicalReportHandler.generate_report_analysis(extracted_text, filename)
                return {"analysis": analysis, "is_fallback": True}, 200
                
        except Exception as e:
            print(f"Error in report analysis: {str(e)}")
            print(traceback.format_exc())
            
            # Ultimate fallback
            analysis = MedicalReportHandler.generate_report_analysis(extracted_text, filename)
            return {"analysis": analysis, "is_fallback": True}, 200


# Update endpoints to use the handler classes

@app.route('/api/analyze-report', methods=['POST'])
def analyze_report():
    token = request.headers.get('Authorization')
    print(f"Received report analysis request with token: {token}")
    
    if not MedicalReportHandler.verify_token(token):
        return jsonify({"error": "Unauthorized"}), 401
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    result, status_code = MedicalReportHandler.analyze_report(token, file)
    
    if status_code == 200:
        return jsonify(result)
    else:
        return jsonify(result), status_code


if __name__ == '__main__':
    app.run(debug=True, port=5000)
