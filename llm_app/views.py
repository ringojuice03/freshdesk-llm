import os
from dotenv import load_dotenv
load_dotenv()
import requests
import re
from django.shortcuts import render
from django.http import HttpResponse
import json

from groq import Groq
# groq_api_key = os.environ.get('GROQ_API_KEY')
groq_api_key = "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"
groq_client = Groq(api_key=groq_api_key)

from google import genai
gemini_api_key = os.environ.get('GEMINI_API_KEY')
gemini_client = genai.Client(api_key=gemini_api_key)

from qdrant_client import models, QdrantClient
qd_client = QdrantClient('http://localhost:6333')
collection_name = 'freshdesk-tickets-rag'

def extract_query_and_issue_types():
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
    return query_type_choices,specific_issue_choices

query_type_choices, specific_issue_choices = extract_query_and_issue_types()

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
    fixed_tickets = []

    file_path = "documents.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    for doc in data:
        if doc['problem'] == None or doc['resolution'] == None:
            print(doc['id'])
            description_url = f"https://{domain}.freshdesk.com/api/v2/tickets/{doc['id']}"
            response = requests.get(description_url, headers=headers, auth=auth)
        
            if response.status_code == 200:
                ticket_description = response.json()
            else:
                print(f"Ticket ID {doc['id']} not found (Status: {response.status_code}). Skipping...")
                continue

            doc = get_payload(ticket_description)

            raw_conversations = get_all_conversations(doc['id'], domain, headers, auth)
            parsed_conversations = parse_freshdesk_conversations(raw_conversations, ticket_description)
            full_text_conversation = parsed_conversations['full_text_conversation']

            problem_and_resolution = get_problem_and_resolution(full_text_conversation)
            problem, resolution = split_problem_and_resolution(problem_and_resolution)
            doc['problem'] = problem
            doc['resolution'] = resolution

            print(problem_and_resolution)
            print('-----')

        fixed_tickets.append(doc)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(fixed_tickets, f, ensure_ascii=False, indent=4)

    print('Tickets okay')
    return render(request, 'hello.html')

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

        answer = "Only respond when the customer messaged last."
        if is_customer_last_msg:
            classification_details = rag_classification(problem)
            query_type, specific_issue_type = split_query_and_issue_type(classification_details)
            answer, priority = rag(problem, query_type)
            print(f'Query type: {query_type}')
            print(f'Specific issue type: {specific_issue_type}')
            print(f'Priority: {priority}')
            print(f'Answer: {answer}')
            
        ticket_list.append({
            'id': ticket_id,
            'name': contact_name,
            'conversation': parsed_conversations['ordered_conversation_details'],
            'generated_response': answer,
            'payload': payload,
            'query_type': query_type,
            'specific_issue_type': specific_issue_type,
            'priority': priority,
        })
        
        print('----------')

    return render(request, 'response.html', {'tickets': ticket_list,})

def rag_classification(problem: str) -> str:
    search_response = rrf_search(problem)
    prompt = classification_prompt(problem, search_response)
    response = llm_response(prompt)

    return response

def rag(problem: str, query_type: str) -> str:
    search_response = rrf_filter_search(problem, query_type)
    prompt = answer_ticket_prompt(problem, search_response)
    response = llm_response(prompt)

    answer, priority = split_answer_and_priority(response)

    return answer, priority

def rrf_search(problem: str, limit: int = 5) -> str:
    search_results = qd_client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=problem,
                    model="BAAI/bge-large-en-v1.5"
                ),
                using="baai-large",
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
                    model="BAAI/bge-large-en-v1.5"
                ),
                using="baai-large",
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
        priority = result.payload['priority']
        context += f'Problem: {problem}\nResolution: {resolution}\nPriority: {priority}\n\n'

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
## ROLE & CONTEXT
You are a customer support representative for Tindahang Tapat, a digital platform enabling sari-sari stores to order groceries via mobile phone. Generate appropriate responses using only the provided knowledge base context.

## PRIMARY OBJECTIVE
Deliver accurate, helpful solutions to customer problems based strictly on available context information, while maintaining professional service standards.

## RESPONSE METHODOLOGY

### Content Requirements
- **Context fidelity**: Use ONLY information from the provided context
- **Solution focus**: Prioritize actionable steps and clear guidance
- **Completeness**: Address all aspects of the customer's problem when context allows
- **Accuracy**: Never infer or assume details not explicitly stated in context

### Communication Standards
- **Professional tone**: Courteous, confident, and solution-oriented
- **Filipino market awareness**: Use terminology familiar to sari-sari store owners
- **Clarity**: Avoid technical jargon; use simple, direct language
- **Brevity**: Concise responses (2-4 sentences) that fully address the issue

### Fallback Protocol
When context is insufficient, use exactly: *"I don't have specific information about this issue in my current resources. Please contact our support team at [support contact] for immediate assistance with your concern."*

## INPUT DATA

### Knowledge Base Context
```
{context}
```

### Customer Problem
```
{problem}
```

## RESPONSE GENERATION RULES

### Content Validation
1. **Context check**: Ensure solution exists in provided context
2. **Completeness check**: Verify all problem aspects are addressed
3. **Accuracy check**: Confirm no assumptions beyond context are made
4. **Tone check**: Maintain professional, helpful customer service voice

### Quality Standards
- **Actionable**: Include specific steps when solutions are available
- **Comprehensive**: Address root cause when context provides sufficient detail
- **Preventive**: Mention prevention tips if included in context
- **Follow-up ready**: Set clear expectations for next steps if needed

## OUTPUT REQUIREMENTS
Provide exactly two sections with no additional commentary, explanations, or formatting:

ANSWER
[Complete customer response based on context - 2-4 sentences addressing their problem with specific, actionable guidance]

PRIORITY  
[Single priority level: Low, Medium, High, or Urgent]
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
        temperature=0.4, # for more determinisitc output, default 1
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

def embed_tickets(request):
    tickets_payload = []
    with open("documents.json", "r", encoding="utf-8") as f:
        tickets_payload = json.load(f)

    print('Initializing qdrant client')
    
    # qd_client.create_collection(
    #     collection_name=collection_name,
    #     vectors_config={
    #         "baai-large": models.VectorParams(
    #             size = 1024,
    #             distance = models.Distance.COSINE
    #         )
    #     },
    #     sparse_vectors_config={
    #         "bm25": models.SparseVectorParams(
    #             modifier = models.Modifier.IDF
    #         )
    #     },
    # )

    print('Uploading data points...')
    qd_client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=ticket['id'],
                vector={
                    "baai-large": models.Document(
                        text=str(ticket['problem']),
                        model="BAAI/bge-large-en-v1.5"
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

    print('Indexing payload...')
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

def get_payload_to_json(request):
    tickets_payload = []

    print('Fetching tickets')
    for ticket_id in range(201, 501):
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

        print(problem_and_resolution)
        print('-----')

        tickets_payload.append(payload) 

        file_path = "documents.json"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        # data.extend(tickets_payload)
        data.append(payload)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    return render(request, 'json.html')

def split_problem_and_resolution(text):
    pattern = r"""
        [#\*\s]*CORE\s+PROBLEM\s+STATEMENT[:\s\#\*]*
        (.*?)                                  
        [#\*\s]*RESOLUTION[:\s\#\*]*    
        (.*)                          
    """

    match = re.search(pattern, text, re.DOTALL | re.VERBOSE)
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

def split_answer_and_priority(text: str):
    pattern = r"""
        [#\*\s]*ANSWER[:\s\*\#]*
        (.*?)                                  
        [#\*\s]*PRIORITY[:\s\*\#]*        
        (.*)              
    """

    match = re.search(pattern, text, re.DOTALL | re.VERBOSE)
    if match:
        answer = match.group(1).strip()
        priority = match.group(2).strip()
        return answer, priority
    else:
        return None, None

def get_problem(full_text_conversation):
    prompt_template = """
## CONTEXT
You are extracting customer problems from Tindahang Tapat support conversations. Tindahang Tapat is a digital platform enabling sari-sari stores to order groceries via mobile phone. Your output will be stored in a vector database for RAG-based support automation.

## ANALYSIS OBJECTIVE
Extract the **primary functional problem** that prompted the customer to contact support, formatted for optimal knowledge base retrieval.

## PROBLEM STATEMENT REQUIREMENTS

### Structure (2-3 sentences maximum)
1. **Issue identification**: What specific functionality failed or caused confusion
2. **Context**: Under what circumstances the problem occurs
3. **Impact**: How it affects the customer's workflow (optional, if relevant)

### Quality Standards
- **Functional focus**: Describe system behavior, not customer emotions
- **Actionable language**: Use verbs that describe what went wrong
- **Consistent terminology**: Apply standardized platform vocabulary
- **Generic applicability**: Remove unique identifiers while preserving problem essence
- **Search optimization**: Include keywords support agents would use to find solutions

## STANDARDIZATION GUIDELINES

### Content Normalization
**Products & Brands**
- "Nescafe 3-in-1 sachets" → "instant coffee products"
- "Tide powder 1kg" → "laundry detergent"
- "Lucky Me noodles" → "instant noodle products"

**Customer References**
- "Store owner Maria from Bataan" → "store owner"
- "My sari-sari store in Quezon City" → "customer's store"
- "Order #TT240156" → "customer order"

**Technical Terms**
- "App crashed" → "mobile application stopped responding"
- "Payment failed" → "payment processing error occurred"
- "Can't login" → "authentication failure during login process"

**Process References**
- "Ordering process" → "product ordering workflow"
- "Checkout" → "order finalization process"
- "Delivery tracking" → "order status monitoring"

### Content Exclusions
**Remove completely:**
- Greetings, closings, courtesies
- Agent responses and solutions
- Timestamps, reference numbers
- Emotional expressions ("frustrated", "disappointed")
- Repetitive explanations of the same issue

**Preserve:**
- Technical symptoms and error conditions
- User actions that triggered the problem
- System responses or lack thereof
- Workflow step where issue occurred

## CONVERSATION DATA
```
{conversation}
```

## OUTPUT INSTRUCTIONS
Extract and output ONLY the standardized problem statement. Follow these rules:
- Use exactly 2-3 sentences
- Start directly with the problem (no preamble)
- End with a period
- If multiple problems exist, focus on the primary/most severe one
- If no clear problem is identifiable, output exactly: "No identifiable problem."

## EXPECTED OUTPUT FORMAT
[Standardized problem statement optimized for vector similarity search and keyword matching]
"""
    prompt = prompt_template.format(conversation=full_text_conversation)
    answer = llm_response(prompt)

    return answer

def get_problem_and_resolution(full_text_conversation):
    prompt_template = """
## CONTEXT
You are analyzing customer support conversations for Tindahang Tapat, a digital platform enabling sari-sari stores to order groceries via mobile phone. Your task is to extract standardized problem-resolution pairs for a RAG knowledge base.

## ANALYSIS REQUIREMENTS

### Problem Statement Extraction
Create a **searchable, generic problem description** that:
- **Focuses on function over emotion**: Describe what failed/broke, not how customers felt
- **Uses standard terminology**: Platform features, order processes, payment methods, product categories
- **Removes specificity**: No customer names, order numbers, specific brands, or timestamps
- **Enables similarity matching**: Use consistent vocabulary for similar issues
- **Length**: It should be 2-3 sentences.

### Resolution Documentation  
Provide an **actionable solution summary** that:
- **Details specific steps**: What the agent did to resolve the issue
- **Includes verification**: How resolution was confirmed
- **Notes preventive measures**: Steps to avoid recurrence
- **States clear outcome**: "RESOLVED" with method, or "UNRESOLVED" with next steps
- **Length**: It should be 2-3 sentences.

## STANDARDIZATION RULES

### Content Normalization
- **Products**: "Nescafe 3-in-1" → "instant coffee product"
- **Customers**: "Mrs. Santos from Quezon City" → "store owner"  
- **Orders**: "Order #TT2024001" → "customer order"
- **Amounts**: "₱1,250.00" → "order amount"
- **Dates/Times**: Remove all temporal references unless process-critical

### Language Standardization
- **Payment issues**: "payment processing", "transaction failure", "payment method"
- **Delivery problems**: "delivery scheduling", "logistics coordination", "fulfillment"
- **Product concerns**: "product availability", "inventory discrepancy", "catalog issue"
- **Account access**: "login authentication", "account verification", "profile management"
- **App functionality**: "mobile app", "platform feature", "system functionality"

### Excluded Content
Remove entirely:
- Greetings, sign-offs, pleasantries
- Agent names, department references  
- Conversation metadata (timestamps, channel info)
- Emotional expressions and subjective language
- Repetitive confirmations or status updates

## CONVERSATION DATA
{conversation}

## OUTPUT REQUIREMENTS
Provide ONLY the CORE PROBLEM STATEMENT and RESOLUTION below. No explanations, commentary, or additional text.

CORE PROBLEM STATEMENT:
[Generic, searchable problem description optimized for vector similarity and keyword matching]

RESOLUTION:
[Actionable solution steps and final status: RESOLVED or UNRESOLVED with next steps]
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