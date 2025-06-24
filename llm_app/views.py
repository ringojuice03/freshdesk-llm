import os
from dotenv import load_dotenv
load_dotenv()
import requests
import re
from django.shortcuts import render
from django.http import HttpResponse
import json

from groq import Groq
groq_api_key = os.environ.get('GROQ_API_KEY')
groq_client = Groq(api_key=groq_api_key)

from google import genai
gemini_api_key = os.environ.get('GEMINI_API_KEY')
gemini_client = genai.Client(api_key=gemini_api_key)

from qdrant_client import models, QdrantClient

headers = {
    'Content-Type': 'application/json'
}
freshdesk_api_key = os.environ.get('FRESHDESK_API_KEY')
auth = (freshdesk_api_key, 'X')
domain = 'nueca'

PRIORITY_MAP = {
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Urgent"
}

STATUS_MAP = {
    2: "Open",
    3: "Pending",
    4: "Resolved",
    5: "Closed",
    6: "Waiting on Customer",
    7: "Waiting on Warehouse Confirmation",
    8: "Follow-up on Warehouse Confirmation",
}


def hello_world(request):
    return render(request, 'hello.html')


def embed_tickets(request):
    tickets_payload = []
    with open("documents.json", "r", encoding="utf-8") as f:
        tickets_payload = json.load(f)

    print('Initializing qdrant client')
    qd_client = QdrantClient('http://localhost:6333')

    collection_name = 'freshdesk-tickets-rag'

    qd_client.delete_collection(collection_name=collection_name)

    qd_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "jina-small": models.VectorParams(
                size = 512,
                distance = models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier = models.Modifier.IDF
            )
        },
    )

    print('Uploading data points...')
    qd_client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=ticket['id'],
                vector={
                    "jina-small": models.Document(
                        text=str(ticket['problem']),
                        model="jinaai/jina-embeddings-v2-small-en"
                    ),
                    "bm25": models.Document(
                        text=str(ticket['problem']),
                        model="Qdrant/bm25"
                    )
                },
                payload=ticket
            )

            for ticket in tickets_payload
        ]
    )

    print('Creating payload index...')
    qd_client.create_payload_index(
        collection_name=collection_name,
        field_name="priority",
        field_schema="keyword"
    )

    qd_client.create_payload_index(
        collection_name=collection_name,
        field_name="status",
        field_schema="keyword"
    )

    qd_client.create_payload_index(
        collection_name=collection_name,
        field_name="query_type",
        field_schema="keyword"
    )

    qd_client.create_payload_index(
        collection_name=collection_name,
        field_name="specific_issue",
        field_schema="keyword"
    )

    print('Done')    
    return render(request, 'embed.html')

def get_payload_to_json(request):
    tickets_payload = []

    print('Fetching tickets')
    # 21901
    for ticket_id in range(21400, 21501):
        print(ticket_id)
        description_url = f"https://{domain}.freshdesk.com/api/v2/tickets/{ticket_id}"
        ticket_description = requests.get(description_url, headers=headers, auth=auth).json()

        payload = get_payload(ticket_description)

        if payload['status'] == 2: continue

        raw_conversations = get_all_conversations(ticket_id, domain, headers, auth)
        parsed_conversations = parse_freshdesk_conversations(raw_conversations, ticket_description)
        full_text_conversation = parsed_conversations['full_text_conversation']

        problem_and_resolution = get_problem_and_resolution(full_text_conversation)
        problem, resolution = split_problem_and_resolution(problem_and_resolution)
        payload['problem'] = problem
        payload['resolution'] = resolution

        tickets_payload.append(payload)

    with open("documents.json", "w", encoding="utf-8") as f:
        json.dump(tickets_payload, f, ensure_ascii=False, indent=4)

    return render(request, 'json.html')


def latest_tickets(request):
    tickets_url = f"https://{domain}.freshdesk.com/api/v2/tickets?order_by=created_at&order_type=desc&per_page=5"
    ticket_response = requests.get(tickets_url, headers=headers, auth=auth)
    tickets = ticket_response.json() if ticket_response.status_code == 200 else []
    ticket_list = []

    for ticket_id in range(20969, 20975):
    # for ticket in tickets:
    #     ticket_id = ticket.get('id')
        description_url = f"https://{domain}.freshdesk.com/api/v2/tickets/{ticket_id}"
        ticket_description = requests.get(description_url, headers=headers, auth=auth).json()

        requestor_id = ticket_description['requester_id']
        contact = get_contact(requestor_id, domain, headers, auth)

        contact_name = contact.get('name')

        payload = get_payload(ticket_description)

        print(f"Name: {contact_name}")
        # print(f"Priority: {payload['priority']}")
        # print(f"Status: {payload['status']}")
        # print(f"Query type: {payload['query_type']}")
        # print(f"Specific Issue: {payload['specific_issue']}")

        raw_conversations = get_all_conversations(ticket_id, domain, headers, auth)
        parsed_conversations = parse_freshdesk_conversations(raw_conversations, ticket_description)
        full_text_conversation = parsed_conversations['full_text_conversation']

        problem_and_resolution = get_problem_and_resolution(full_text_conversation)
        problem, resolution = split_problem_and_resolution(problem_and_resolution)
        payload['problem'] = problem
        payload['resolution'] = resolution

        # print(f"Total conversations: {len(parsed_conversations['full_conversation'])}")
        # for i, conv in enumerate(parsed_conversations['full_conversation']):
        #     print(f"{conv}")

        # print(f"Problem: {problem}")
        # print(f"Solution: {resolution}")

        is_customer_last_msg = False
        last_msg = parsed_conversations['ordered_conversation_details'][-1]
        if last_msg['role'] == 'Customer':
            is_customer_last_msg = True
            last_customer_msg = last_msg['body_text']

        # translated_conversation = google_translate(parsed_conversations['full_text_conversation'])
        # print(f'Translated conversation:\n{translated_conversation}')

        generated_response = "Only respond when the customer messaged last."
        # if is_customer_last_msg:
        #     print('Generating response...')
        #     generated_response = generate_response(last_customer_msg, translated_conversation)
        #     print(f'Response:\n{generated_response}')

        ticket_list.append({
            'id': ticket_id,
            'name': contact_name,
            'conversation': parsed_conversations['ordered_conversation_details'],
            'generated_response': generated_response,
            'payload': payload
        })
        
        print('----------')

    return render(request, 'response.html', {'tickets': ticket_list,})


def get_payload(ticket_description):
    priority_integer = ticket_description.get('priority')
    priority = PRIORITY_MAP.get(priority_integer, "Unknown")

    status_integer = ticket_description.get('status')
    status = STATUS_MAP.get(status_integer, "Unknown")

    query_type = ticket_description.get('type')
    specific_issue = ticket_description.get('custom_fields')['cf_specific_issues_and_inquiries']

    return {
        'id': ticket_description['id'],
        'priority': priority,
        'status': status,
        'query_type': query_type,
        'specific_issue': specific_issue 
    }


def split_problem_and_resolution(text):
    match = re.search(
        r"\*\*CORE PROBLEM STATEMENT:\*\*\s*(.*?)\s*\*\*RESOLUTION:\*\*\s*(.*)",
        text,
        re.DOTALL
    )
    if match:
        problem = match.group(1).strip()
        resolution = match.group(2).strip()
        return problem, resolution
    else:
        return None, None


def get_problem_and_resolution(full_text_conversation):
    prompt_template = """
You are analyzing customer support conversations for Tindahang Tapat, a digital platform that enables sari-sari stores to order groceries via mobile phone.

## TASK
Extract and structure the key information from the conversation below for storage in a vector database used by a RAG system.

## INSTRUCTIONS
1. **Problem Statement**: Identify the customer's core issue in 2-3 clear, concise sentences
   - Focus on the functional problem, not emotional expressions
   - Use generic product categories instead of specific brand names
   - Remove customer-specific details (names, phone numbers, order IDs)
   - Prioritize searchable, reusable problem patterns

2. **Resolution**: Summarize the final solution provided in 2-3 sentences
   - Include specific steps taken by the agent
   - Mention any preventive measures or follow-up actions
   - If unresolved, state "UNRESOLVED" and note next steps

3. **Cleanup Requirements**:
   - Remove: timestamps, agent names, email signatures, greetings, pleasantries, repetitions
   - Normalize: technical terms, feature names
   - **Generalize products**: Replace specific brand names with generic categories (e.g., "Nescafe stick" → "instant coffee product", "Tide detergent" → "laundry detergent")
   - Generalize: customer-specific details to make the problem statement reusable

4. **Format**: Use clear, professional language suitable for knowledge base search

---

**CONVERSATION:**
{conversation}

---

Output CORE [CORE PROBLEM STATEMENT] and [RESOLUTION] ONLY.
If there is no problem and resolution, output No Problem and No Resolution respectively.

**CORE PROBLEM STATEMENT:**
[Customer's main issue in standardized, searchable format]

**RESOLUTION:**
[Final solution provided or current status if unresolved]
"""
    prompt = prompt_template.format(conversation=full_text_conversation)

    # response = gemini_client.models.generate_content(
    #         model='gemini-2.0-flash',
    #         contents=prompt
    #     )
    # answer = response.candidates[0].content.parts[0].text
    
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
        {
            "role": "user",
            "content": prompt,
        }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    answer = ""
    for chunk in response:
        content = chunk.choices[0].delta.content or ""
        answer += content

    return answer


def get_contact(requestor_id, domain, headers, auth):
    contact_url = f"https://{domain}.freshdesk.com/api/v2/contacts/{requestor_id}"
    contact_response = requests.get(contact_url, headers=headers, auth=auth)
    contact = contact_response.json()

    return contact


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

**Make it 3 sentences long at maximum. Your Detailed, Empathetic, and Thorough Response (as a Customer Service Representative):**
"""

    prompt = prompt_template.format(translated_conversation=translated_conversation, last_customer_msg=last_customer_msg)
    response = gemini_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )

    return response.candidates[0].content.parts[0].text
