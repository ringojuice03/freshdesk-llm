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
qd_client = QdrantClient('http://localhost:6333')
collection_name = 'freshdesk-tickets-rag'

with open('fields.json', "r", encoding="utf-8") as f:
    fields = json.load(f)
qt_choices = fields['query_type_choices']
si_choices = fields['specific_issue_choices']
query_type_choices = ""
for choice in qt_choices:
    query_type_choices = query_type_choices + choice + "\n"
specific_issue_choices = ""
for choice in si_choices:
    specific_issue_choices = specific_issue_choices + choice + "\n"

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
    print(query_type_choices)
    print('-----')
    print(specific_issue_choices)
    return render(request, 'hello.html')


def embed_tickets(request):
    tickets_payload = []
    with open("documents.json", "r", encoding="utf-8") as f:
        tickets_payload = json.load(f)

    print('Initializing qdrant client')
    
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
    for ticket_id in range(0, 101):
        print(ticket_id)
        description_url = f"https://{domain}.freshdesk.com/api/v2/tickets/{ticket_id}"
        response = requests.get(description_url, headers=headers, auth=auth)
    
        if response.status_code == 200:
            ticket_description = response.json()
        else:
            print(f"Ticket ID {ticket_id} not found (Status: {response.status_code}). Skipping...")
            continue

        payload = get_payload(ticket_description)

        if payload['status'] == 2: continue
        if payload['query_type'] == None: continue 
        if payload['specific_issue'] == None: continue

        raw_conversations = get_all_conversations(ticket_id, domain, headers, auth)
        parsed_conversations = parse_freshdesk_conversations(raw_conversations, ticket_description)
        full_text_conversation = parsed_conversations['full_text_conversation']

        problem_and_resolution = get_problem_and_resolution(full_text_conversation)
        problem, resolution = split_problem_and_resolution(problem_and_resolution)
        payload['problem'] = problem
        payload['resolution'] = resolution

        tickets_payload.append(payload) 

    file_path = "documents.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.extend(tickets_payload)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return render(request, 'json.html')


def latest_tickets(request):
    tickets_url = f"https://{domain}.freshdesk.com/api/v2/tickets?order_by=created_at&order_type=desc&per_page=5"
    ticket_response = requests.get(tickets_url, headers=headers, auth=auth)
    tickets = ticket_response.json() if ticket_response.status_code == 200 else []
    ticket_list = []

    # for ticket in range(0, 1):
    for ticket in tickets:
        ticket_id = ticket.get('id')
        description_url = f"https://{domain}.freshdesk.com/api/v2/tickets/{ticket_id}"
        ticket_description = requests.get(description_url, headers=headers, auth=auth).json()

        requestor_id = ticket_description['requester_id']
        contact = get_contact(requestor_id, domain, headers, auth)

        contact_name = contact.get('name')
        print(f"Name: {contact_name}")

        payload = get_payload(ticket_description)

        raw_conversations = get_all_conversations(ticket_id, domain, headers, auth)
        parsed_conversations = parse_freshdesk_conversations(raw_conversations, ticket_description)
        full_text_conversation = parsed_conversations['full_text_conversation']

        problem = get_problem(full_text_conversation)
        payload['problem'] = problem
        print(f'Core Problem: {problem}')

        # to False later
        is_customer_last_msg = True
        last_msg = parsed_conversations['ordered_conversation_details'][-1]
        if last_msg['role'] == 'Customer':
            is_customer_last_msg = True
            last_customer_msg = last_msg['body_text']

        generated_response = "Only respond when the customer messaged last."
        if is_customer_last_msg:
            classification_details = rag_classification(problem)
            query_type, specific_issue_type = split_query_and_issue_type(classification_details)
            generated_response = rag(problem, query_type)
            print(f'Query type: {query_type}')
            print(f'Specific issue type: {specific_issue_type}')
            
            # generated_response = generate_response(last_customer_msg, full_text_conversation)

        ticket_list.append({
            'id': ticket_id,
            'name': contact_name,
            'conversation': parsed_conversations['ordered_conversation_details'],
            'generated_response': generated_response,
            'payload': payload,
            'query_type': query_type,
            'specific_issue_type': specific_issue_type,
        })
        
        print('----------')

    return render(request, 'response.html', {'tickets': ticket_list,})


def rag_classification(problem: str) -> str:
    search_response = rrf_search(problem)
    prompt = classification_prompt(problem, search_response)
    response = llm_response(prompt)

    return response


def rag(problem: str, query_type: str) -> str:
    print(f'Query type at rag: {query_type}')
    search_response = rrf_filter_search(problem, query_type)
    prompt = answer_ticket_prompt(problem, search_response)
    response = llm_response(prompt)

    return response


def rrf_search(problem: str, limit: int = 5) -> str:
    search_results = qd_client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=problem,
                    model="jinaai/jina-embeddings-v2-small-en"
                ),
                using="jina-small",
                limit=5*limit,
            ),
            models.Prefetch(
                query=models.Document(
                    text=problem,
                    model="Qdrant/bm25"
                ),
                using="bm25",
                limit=5*limit,
            ),
        ],
        limit=limit,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
    )

    context = ""
    for result in search_results.points:
        problem = result.payload['problem']
        resolution = result.payload['resolution'] 
        context += f'Problem: {problem}\nResolution: {resolution}\n---------------\n'

    return context

def rrf_filter_search(problem: str, query_type: str, limit: int = 5) -> str:
    search_results = qd_client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="query_type",
                            match=models.MatchValue(value=query_type),
                        ),
                    ]
                ),
                query=models.Document(
                    text=problem,
                    model="jinaai/jina-embeddings-v2-small-en"
                ),
                using="jina-small",
                limit=5*limit,
            ),
            models.Prefetch(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="query_type",
                            match=models.MatchValue(value=query_type),
                        ),
                    ]
                ),
                query=models.Document(
                    text=problem,
                    model="Qdrant/bm25"
                ),
                using="bm25",
                limit=5*limit,
            ),
        ],
        limit=limit,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
    )

    context = ""
    for result in search_results.points:
        problem = result.payload['problem']
        resolution = result.payload['resolution'] 
        context += f'Problem: {problem}\nResolution: {resolution}\n\n'

    return context

def classification_prompt(problem: str, context: str) -> str:
    prompt_template = """
You are a customer support representative for Tindahang Tapat, a digital platform that enables sari-sari stores to order groceries via mobile phone.

## OBJECTIVE
Classify the customer's problem into the appropriate query type and specific issue category using the provided context as reference.

## CLASSIFICATION FRAMEWORK

### Available Query Types
```
{query_type_choices}
```

### Available Specific Issue Types
```
{specific_issue_choices}
```

## CLASSIFICATION GUIDELINES
- Use the **CONTEXT** as your primary reference for accurate classification
- Select the **most specific** category that matches the problem
- Choose only **one** query type and **one** specific issue type
- If uncertain between categories, prioritize the one that best captures the core functionality issue
- Output **NONE** if no category adequately fits the problem

## REFERENCE CONTEXT
```
{context}
```

## CUSTOMER PROBLEM
```
{problem}
```

## OUTPUT REQUIREMENTS
- Provide **only** the classifications below
- No additional text, explanations, or formatting
- Use **NONE** if no suitable category exists

## OUTPUT FORMAT

QUERY TYPE:
[Selected category or NONE]

SPECIFIC ISSUE TYPE:
[Selected category or NONE]
"""
    prompt = prompt_template.format(query_type_choices=query_type_choices, 
                                    specific_issue_choices=specific_issue_choices,
                                    context=context,
                                    problem=problem
                                )
    
    return prompt

def answer_ticket_prompt(problem: str, context: str) -> str:
    prompt_template = """
You are a customer support representative for Tindahang Tapat, a digital platform that enables sari-sari stores to order groceries via mobile phone.

## OBJECTIVE
Provide a helpful answer to the customer's problem using only the information available in the provided context.

## RESPONSE GUIDELINES
- Base your answer **exclusively** on the provided context
- Provide clear, actionable solutions when available
- Use professional, friendly customer service language
- If the context doesn't contain relevant information, respond with: "I don't have information about this issue in my current resources. Please contact our support team for further assistance."

## REFERENCE CONTEXT
```
{context}
```

## CUSTOMER PROBLEM
```
{problem}
```

## OUTPUT REQUIREMENTS
- Provide a direct, helpful response
- No meta-commentary or explanations about the process
- Stay within the bounds of the provided context
- The output should ONLY contain the ANSWER. DO NOT INCLUDE ANYTHING ELSE.

## OUTPUT FORMAT

ANSWER:
[Your complete, 2-3 sentence, response to the customer]
"""
    prompt = prompt_template.format(context=context, problem=problem)
    
    return prompt

def llm_response(prompt: str) -> str:
    # response = gemini_client.models.generate_content(
    #         model='gemini-2.0-flash',
    #         contents=prompt
    #     )
    # answer = response.candidates[0].content.parts[0].text
    
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
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
    pattern = r"""
        [#\*\s]*CORE\s+PROBLEM\s+STATEMENT[:\s]*[#\*\s]*
        (.*?)                                  
        [#\*\s]*RESOLUTION[:\s]*[#\*\s]*        
        (.*)                          
    """

    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE | re.VERBOSE)
    if match:
        problem = match.group(1).strip()
        resolution = match.group(2).strip()
        return problem, resolution
    else:
        return None, None


def split_query_and_issue_type(text: str):
    pattern = r"""
        [#\*\s]*QUERY\s+TYPE[:\s]*
        (.*?)                                  
        [#\*\s]*SPECIFIC\s+ISSUE\s+TYPE[:\s]*        
        (.*)              
    """

    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE | re.VERBOSE)
    if match:
        query_type = match.group(1).strip()
        specific_issue_type = match.group(2).strip()
        return query_type, specific_issue_type
    else:
        return None, None


def get_problem_and_resolution(full_text_conversation):
    prompt_template = """
You are analyzing customer support conversations for Tindahang Tapat, a digital platform that enables sari-sari stores to order groceries via mobile phone.

## OBJECTIVE
Extract and structure key problem-resolution pairs from the conversation below for storage in a vector database used by a RAG system.

## OUTPUT REQUIREMENTS

### Core Problem Statement (2-3 sentences)
- **Functional focus**: Describe the actual issue, not emotional reactions
- **Generic terminology**: Use product categories instead of brand names
- **Reusable format**: Remove customer-specific identifiers
- **Searchable language**: Use standard platform terminology

### Resolution Summary (2-3 sentences)
- **Action-oriented**: Include specific steps taken by the agent
- **Complete process**: Mention preventive measures or follow-up actions
- **Clear status**: If unresolved, state "UNRESOLVED" and note next steps

### Content Processing Rules

**Remove:**
- Timestamps, agent names, signatures, greetings
- Pleasantries, repetitive statements
- Customer-specific identifiers (names, phone numbers, order IDs)

**Generalize:**
- Brand names: "Nescafe stick" → "instant coffee product"
- Product references: "Tide detergent" → "laundry detergent"
- Customer details: Make scenarios broadly applicable
- Technical terms: Use standardized platform language

## CONVERSATION DATA
{conversation}

## OUTPUT FORMAT (The output should be optimized for hybrid retrieval (dense + BM25) in RAG systems. Output ONLY the CORE PROBLEM STATEMENT and RESOLUTION. DO NOT INCLUDE ANYTHING ELSE.)
CORE PROBLEM STATEMENT:
[Standardized problem description for knowledge base search]

RESOLUTION:
[Solution steps and outcome, or "UNRESOLVED" with next steps]
"""
    prompt = prompt_template.format(conversation=full_text_conversation)
    answer = llm_response(prompt)

    return answer


def get_problem(full_text_conversation):
    prompt_template = """
You are a customer support representative for Tindahang Tapat, a digital platform that enables sari-sari stores to order groceries via mobile phone.

## OBJECTIVE
Extract and structure the customer's core problem from the conversation below for storage in a vector database used by a RAG system.

## OUTPUT REQUIREMENTS

### Core Problem Statement
Create a 2-3 sentence problem statement that is:
- **Functional**: Focus on the actual issue, not emotional expressions
- **Generic**: Use product categories instead of brand names
- **Reusable**: Remove customer-specific details (names, phone numbers, order IDs)
- **Searchable**: Use standard terminology for knowledge base retrieval

### Content Processing Rules

**Remove:**
- Timestamps, agent names, email signatures
- Greetings, pleasantries, repetitive statements
- Customer-specific identifiers

**Generalize:**
- Product names: "Nescafe stick" → "instant coffee product"
- Brand names: "Tide detergent" → "laundry detergent"
- Technical terms: Use standardized platform terminology
- Customer details: Make scenarios broadly applicable

**Maintain:**
- Core functionality issues
- Process flow problems
- System behavior descriptions

## CONVERSATION DATA
```
{conversation}
```

# OUTPUT FORMAT
[Deliver ONLY the standardized problem statement, DO NOT include anything else in the output. If no clear problem exists, output "No identifiable problem."]
"""
    prompt = prompt_template.format(conversation=full_text_conversation)
    answer = llm_response(prompt)

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
    full_text_conversation += message + "\n"
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
            full_text_conversation += message + "\n" 
            full_conversation.append(message)
            message_data['role'] = 'Customer'
            customer_conversation.append(message_data['body_text'])
        else:
            message = f'Agent: {message_data['body_text']}'
            full_text_conversation += message + "\n" 
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


def generate_response(last_customer_msg: str, conversation: str) -> str:
    prompt_template = """
You are a dedicated Customer Service Representative for Nueca's Tindahang Tapat, an online platform designed to empower 'sari-sari' stores across the Philippines by facilitating their grocery orders. Your primary goal is to provide clear, helpful, and empathetic assistance to our valued store owners.

Carefully review the entire conversation history to understand the full context of the customer's interaction. Your response should be **comprehensive and thorough**, providing a detailed answer that addresses all aspects of the customer's concern. While being detailed, ensure your response remains **direct and focused on providing a solution or clear information.**

It is crucial that your tone is **warm, empathetic, and reassuring**, acknowledging the customer's feelings and concerns. Beyond just answering, your response should be **engaging and address their concerns thoroughly**, leaving no stone unturned and proactively anticipating follow-up questions to provide a complete resolution.

**Conversation History:**
{conversation}

**Customer's Specific Concern/Last Message:**
{last_customer_msg}

**Make it 3 sentences long at maximum. Your Detailed, Empathetic, and Thorough Response (as a Customer Service Representative):**
"""

    prompt = prompt_template.format(conversation=conversation, last_customer_msg=last_customer_msg)
    response = gemini_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )

    return response.candidates[0].content.parts[0].text


def rag_response(last_customer_msg: str, conversation: str) -> str:
    prompt_template = """
You are a dedicated Customer Service Representative for Nueca's Tindahang Tapat, an online platform designed to empower 'sari-sari' stores across the Philippines by facilitating their grocery orders. Your primary goal is to provide clear, helpful, and empathetic assistance to our valued store owners.

Carefully review the entire conversation history to understand the full context of the customer's interaction. Your response should be **comprehensive and thorough**, providing a detailed answer that addresses all aspects of the customer's concern. While being detailed, ensure your response remains **direct and focused on providing a solution or clear information.**

It is crucial that your tone is **warm, empathetic, and reassuring**, acknowledging the customer's feelings and concerns. Beyond just answering, your response should be **engaging and address their concerns thoroughly**, leaving no stone unturned and proactively anticipating follow-up questions to provide a complete resolution.

Most importantly, use the CONTEXT as the basis for catering the customer's concerns/queries.

**CONTEXT:**
{context}

**Conversation History:**
{conversation}

**Customer's Specific Concern/Last Message:**
{last_customer_msg}

**Make it 3 sentences long at maximum. Your Detailed, Empathetic, and Thorough Response (as a Customer Service Representative):**
"""
    prompt = prompt_template.format(conversation=conversation, last_customer_msg=last_customer_msg)
    response = gemini_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )

    return response.candidates[0].content.parts[0].text
