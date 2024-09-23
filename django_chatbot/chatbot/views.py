from django.shortcuts import render
from .models import Patient, Conversation
from datetime import datetime
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain, StuffDocumentsChain
from langchain.memory import ConversationSummaryBufferMemory
import dateparser

# Load the patient data
def get_patient():
    return Patient.objects.first()

# LangChain setup
llm = OpenAI(temperature=0.7)

# Update the prompt template to explicitly define all input variables
prompt_template = """
You are an AI chatbot for healthcare. Respond to patient's health queries, medication regimen, and appointment requests. 
Ignore unrelated or sensitive topics. Here's the context:
Patient: {message}
"""

# Define PromptTemplate
prompt = PromptTemplate.from_template(template=prompt_template)
# Memory for conversation history
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)

# LLM Chain for conversation
conversation_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Setup for retrieving and combining documents
# combine_docs_chain = StuffDocumentsChain(" ")
# vectorstore = " "
# retriever = vectorstore.as_retriever()

# Define the question generator for the conversational retrieval chain
template = (
    "Combine the chat history and follow up question into "
    "a standalone question. Chat History: {chat_history}"
    "Follow up question: {question}"
)
question_prompt = PromptTemplate.from_template(template)
question_generator_chain = LLMChain(llm=llm, prompt=question_prompt)

chain = ConversationalRetrievalChain(
    question_generator=question_generator_chain,
)

# Home view
def home(request):
    patient = get_patient()
    conversation_history = Conversation.objects.filter(patient=patient).order_by('timestamp')

    if request.method == 'POST':
        user_message = request.POST['message']

        if user_message:
            # Collect the necessary context details
            input_variables = {
                'message': user_message,
                # 'doctor_name': patient.doctor_name if patient.doctor_name else 'Unknown Doctor', 
                # 'next_appointment': patient.next_appointment.strftime("%Y-%m-%d %H:%M") if patient.next_appointment else 'Not Scheduled',
                # 'medical_condition': patient.medical_condition if patient.medical_condition else 'No medical condition'
            }

            # Generate response from the LLM Chain
            response = conversation_chain.run(input_variables)

            # Filter response for sensitive topics
            filtered_response = filter_response(response)

            # Save conversation
            Conversation.objects.create(
                patient=patient,
                message=user_message,
                response=filtered_response
            )

            # Check for appointment requests
            check_for_appointment_requests(user_message, patient)

    return render(request, 'chatbot.html', {'conversations': conversation_history, 'patient': patient})

# Filter unrelated/sensitive topics
def filter_response(response):
    sensitive_keywords = ['politics', 'religion', 'finance', 'violence', 'abuse']
    for word in sensitive_keywords:
        if word in response.lower():
            return "Sorry, I can't discuss this topic."
    return response

# Check for appointment requests
def check_for_appointment_requests(message, patient):
    if 'reschedule' in message.lower() and 'appointment' in message.lower():
        requested_date_time = extract_datetime_from_message(message)
        if requested_date_time:
            print(f"Patient {patient.first_name} {patient.last_name} requests an appointment change "
                  f"from {patient.next_appointment} to {requested_date_time}.")

# Extract datetime from the message
def extract_datetime_from_message(message):
    return dateparser.parse(message)

# Entity extraction (medications, symptoms)
def extract_entities_from_message(message):
    entity_prompt = PromptTemplate(template="Extract health-related entities (e.g., medications, symptoms) from: {message}", input_variables=['message'])
    entity_chain = LLMChain(llm=llm, prompt=entity_prompt)
    entities = entity_chain.run({'message': message})
    return entities
