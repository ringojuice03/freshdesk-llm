import os
from dotenv import load_dotenv
load_dotenv()
import requests
from django.shortcuts import render
from django.http import HttpResponse

from google import genai
gemini_api_key = os.environ.get('GEMINI_API_KEY')
gemini_client = genai.Client(api_key=gemini_api_key)


def hello_world(request):
    return render(request, 'hello.html')


def latest_tickets(request):
    headers = {
        'Content-Type': 'application/json'
    }
    freshdesk_api_key = os.environ.get('FRESHDESK_API_KEY')
    auth = (freshdesk_api_key, 'X')
    domain = 'nueca'
    tickets_url = f"https://{domain}.freshdesk.com/api/v2/tickets?order_by=created_at&order_type=desc&per_page=5"
    ticket_response = requests.get(tickets_url, headers=headers, auth=auth)
    tickets = ticket_response.json() if ticket_response.status_code == 200 else []
    ticket_list = []

    for ticket in tickets:
        ticket_id = ticket.get('id')
        description_url = f"https://{domain}.freshdesk.com/api/v2/tickets/{ticket_id}"
        ticket_description = requests.get(description_url, headers=headers, auth=auth).json()

        requestor_id = ticket_description['requester_id']
        contact_url = f"https://{domain}.freshdesk.com/api/v2/contacts/{requestor_id}"
        contact_response = requests.get(contact_url, headers=headers, auth=auth)
        contact = contact_response.json()
        contact_name = contact.get('name')

        print(contact_name)

        raw_conversations = get_all_conversations(ticket_id, domain, headers, auth)
        parsed_conversations = parse_freshdesk_conversations(raw_conversations, ticket_description)

        is_customer_last_msg = False
        last_msg = parsed_conversations['ordered_conversation_details'][-1]
        if last_msg['role'] == 'Customer':
            is_customer_last_msg = True
            last_customer_msg = last_msg['body_text']

        print(f"Total conversations: {len(parsed_conversations['full_conversation'])}")
        for i, conv in enumerate(parsed_conversations['full_conversation']):
            print(f"{conv}")

        ticket_list.append({
            'id': ticket_id,
            'name': contact_name,
            'conversation': parsed_conversations['ordered_conversation_details']
        })

        translated_conversation = google_translate(parsed_conversations['full_text_conversation'])
        print(f'Translated conversation:\n{translated_conversation}')

        generated_response = "Only respond when the customer messaged last."
        if is_customer_last_msg:
            print('Generating response...')
            generated_response = generate_response(last_customer_msg, translated_conversation)
            print(f'Response:\n{generated_response}')

    return render(request, 'response.html', {'tickets': ticket_list, 'generated_response': generated_response})


def get_all_conversations(ticket_id, domain, headers, auth):
    conversations = []
    url = f"https://{domain}.freshdesk.com/api/v2/tickets/{ticket_id}/conversations"
    while url:
        response = requests.get(url, headers=headers, auth=auth)
        if response.status_code != 200:
            break
        data = response.json()
        conversations.extend(data)
        link = response.headers.get('link')
        if link and 'rel="next"' in link:
            import re
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = match.group(1) if match else None
            print("has pagination")
        else:
            url = None
    return conversations


def parse_freshdesk_conversations(conversations, ticket_description):
    sorted_conversations = sorted(conversations, key=lambda x: x['created_at'])
    
    full_text_conversation = ""
    full_conversation = []
    customer_conversation = []
    agent_conversation = []
    ordered_conversation_details = []

    message_data = {
        'id': ticket_description['id'],
        'sequence': 1,
        'body_text': ticket_description['description_text'],
        'timestamp': ticket_description['created_at'],
        'user_id': ticket_description['requester_id'],
        'role': 'Customer'
    }

    message = f'Customer: {message_data['body_text']}'
    full_text_conversation += message
    full_conversation.append(message)
    customer_conversation.append(message_data['body_text'])
    ordered_conversation_details.append(message_data)

    for i, conv in enumerate(sorted_conversations, 2):
        message_data = {
            'id': conv['id'],
            'sequence': i,
            'body_text': conv['body_text'],
            'timestamp': conv['created_at'],
            'user_id': conv['user_id'],
        }

        if message_data['body_text'] == '':
            continue
        if conv['incoming']:
            message = f'Customer: {message_data['body_text']}'
            full_text_conversation += message
            full_conversation.append(message)
            message_data['role'] = 'Customer'
            customer_conversation.append(message_data['body_text'])
        else:
            message = f'Agent: {message_data['body_text']}'
            full_text_conversation += message
            full_conversation.append(message)
            message_data['role'] = 'Agent'
            agent_conversation.append(message_data['body_text'])

        ordered_conversation_details.append(message_data)
        
    return {
        'full_text_conversation': full_text_conversation,
        'full_conversation': full_conversation,
        'customer_conversation': customer_conversation,
        'agent_conversation': agent_conversation,
        'ordered_conversation_details': ordered_conversation_details,
    }


def google_translate(conversations) -> str:
    print('\nTranslating...')
    prompt_template = """
You are an expert linguist and translator specializing in preserving semantic meaning and key terminology. Your task is to translate a given conversation into clear, natural-sounding English.

**Context:** The conversation is relevant to Tindahang Tapat's operations, which is a company of Nueca. Understanding this business context will help ensure accurate translation of company-specific terms, operational procedures, and industry-related vocabulary.

**Key Requirements for Translation:**
1.  **Semantic Fidelity:** The translated English conversation must accurately convey the original meaning, intent, and nuances of the Bikol, Tagalog, or English (or combination thereof) conversation.
2.  **Keyword Preservation:** Identify and retain critical keywords, technical terms, proper nouns, and domain-specific vocabulary. If a direct English equivalent for a keyword is not perfectly semantically aligned, prioritize the most common or closest English counterpart while ensuring the original context is maintained.
3.  **Contextual Accuracy:** Understand the conversational flow and ensure that pronouns, references, and implied meanings are correctly translated based on the full context of the conversation.
4.  **Natural English:** The output should read as fluent and grammatically correct English, avoiding overly literal translations that sound unnatural.
5.  **Business Context Awareness:** Consider Tindahang Tapat's business operations and Nueca's corporate context when translating company-specific terminology, processes, and references.

**Important Note for Embedding:** The translated English output will be used for vector embedding (sparse and dense techniques). Therefore, the clarity, accuracy, and presence of key terms in the English translation are paramount for effective information retrieval and semantic search.

**Conversation to Translate:**
{conversation}

**Translated English Conversation:**
"""

    prompt = prompt_template.format(conversation=conversations)
    response = gemini_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )

    return response.candidates[0].content.parts[0].text


def generate_response(last_customer_msg: str, translated_conversation: str) -> str:
    prompt_template = """
You are a dedicated Customer Service Representative for Nueca's Tindahang Tapat, an online platform designed to empower 'sari-sari' stores across the Philippines by facilitating their grocery orders. Your primary goal is to provide clear, helpful, and empathetic assistance to our valued store owners.

Carefully review the entire conversation history to understand the full context of the customer's interaction. Your response should be **comprehensive and thorough**, providing a detailed answer that addresses all aspects of the customer's concern. While being detailed, ensure your response remains **direct and focused on providing a solution or clear information.**

It is crucial that your tone is **warm, empathetic, and reassuring**, acknowledging the customer's feelings and concerns. Beyond just answering, your response should be **engaging and address their concerns thoroughly**, leaving no stone unturned and proactively anticipating follow-up questions to provide a complete resolution.

**Conversation History:**
{translated_conversation}

**Customer's Specific Concern/Last Message:**
{last_customer_msg}

**Your Detailed, Empathetic, and Thorough Response (as a Customer Service Representative):**
"""

    prompt = prompt_template.format(translated_conversation=translated_conversation, last_customer_msg=last_customer_msg)
    response = gemini_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )

    return response.candidates[0].content.parts[0].text
