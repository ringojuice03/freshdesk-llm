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
from google.genai import types
gemini_api_key = os.environ.get('GEMINI_API_KEY')
gemini_client = genai.Client(api_key=gemini_api_key)

import ollama
ollama.pull("mistral")

from qdrant_client import models, QdrantClient
qd_client = QdrantClient('http://localhost:6333')
collection_name = 'freshdesk-tickets-rag'


with open('fields.json', "r", encoding="utf-8") as f:
    fields = json.load(f)

qt_choices = fields['query_type_choices']
si_choices = fields['specific_issue_choices']

query_type_choices = ""
specific_issue_choices = ""
query_type_list = []
specific_issue_list = []

for choice in qt_choices:
    query_type_choices = query_type_choices + choice + "\n"
    query_type_list.append(choice)
for choice in si_choices:
    specific_issue_choices = specific_issue_choices + choice + "\n"
    specific_issue_list.append(choice)


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
    for ticket_id in range(22296, 22297):
        print(ticket_id)
        description_url = f"https://{domain}.freshdesk.com/api/v2/tickets/{ticket_id}"
        response = requests.get(description_url, headers=headers, auth=auth)
    
        if response.status_code == 200:
            ticket_description = response.json()
        else:
            print(f"Ticket ID {ticket_id} not found (Status: {response.status_code}). Skipping...")
            continue

        doc = get_payload(ticket_description)

        parsed_conversations = get_all_conversations(ticket_id, domain, headers, auth, ticket_description)
        full_text_conversation = parsed_conversations['full_text_conversation']

        problem, resolution = get_problem_and_resolution(full_text_conversation)
        doc['problem'] = problem
        doc['resolution'] = resolution

        print(full_text_conversation)
        print('-----')

    
    return render(request, 'hello.html')

def latest_tickets(request):
    tickets_url = f"https://{domain}.freshdesk.com/api/v2/tickets?order_by=created_at&order_type=desc&per_page=2"
    ticket_response = requests.get(tickets_url, headers=headers, auth=auth)
    tickets = ticket_response.json() if ticket_response.status_code == 200 else []

    ticket_list = []

    # for ticket in range(0, 1):
    #     ticket_id = 22384
    for ticket in tickets:
        ticket_id = ticket.get('id')
        description_url = f"https://{domain}.freshdesk.com/api/v2/tickets/{ticket_id}"
        ticket_description = requests.get(description_url, headers=headers, auth=auth).json()

        requestor_id = ticket_description['requester_id']
        contact = get_contact(requestor_id, domain, headers, auth)

        contact_name = contact.get('name')
        print(f"Name: {contact_name}")

        payload = get_payload(ticket_description)

        parsed_conversations = get_all_conversations(ticket_id, domain, headers, auth, ticket_description)
        full_text_conversation = parsed_conversations['full_text_conversation']

        full_text_conversation = translate_conversation(full_text_conversation)

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
            query_type, specific_issue_type = rag_classification(problem)
            query_type_no_context, specific_issue_type_no_context = rag_classification(problem, use_context = False)
            
            answer, priority = rag(problem, query_type)
            answer_no_context, priority_no_context = rag(problem, query_type, use_context = False)

            print(f'Query type: {query_type}')
            print(f'Specific issue type: {specific_issue_type}')
            print(f'Priority: {priority}')
            print(f'Answer: {answer}')
            
        ticket_list.append({
            'id': ticket_id,
            'name': contact_name,
            'conversation': parsed_conversations['ordered_conversation_details'],
            'payload': payload,

            # rag
            'query_type': query_type,
            'specific_issue_type': specific_issue_type,
            'generated_response': answer,
            'priority': priority,

            # non-rag
            'query_type_no_context': query_type_no_context, 
            'specific_issue_type_no_context': specific_issue_type_no_context,
            'answer_no_context': answer_no_context,
            'priority_no_context': priority_no_context,
        })
        
        print('----------')

    return render(request, 'response.html', {'tickets': ticket_list,})

def rag_classification(problem: str, use_context: bool = True) -> str:
    search_response = rrf_search(problem)
    prompt = classification_prompt(problem, search_response, use_context)

    query_type = ""
    specific_issue_type = ""

    while query_type not in query_type_list or specific_issue_type not in specific_issue_choices:
        response = llm_response(prompt)
        query_type, specific_issue_type = split_query_and_issue_type(response)

        query_type = str(query_type)
        specific_issue_type = str(specific_issue_type)

    return query_type, specific_issue_type

def rag(problem: str, query_type: str, use_context: bool = True) -> str:
    search_response = rrf_filter_search(problem, query_type)
    prompt = answer_ticket_prompt(problem, search_response, use_context)

    priority = ""
    
    while priority not in ["Low", "Medium", "High", "Urgent"]:
        response = llm_response(prompt)
        answer, priority = split_answer_and_priority(response)

        if priority is None:
            priority = ""

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

def classification_prompt(problem: str, context: str, use_context: bool = True) -> str:
    prompt_template = """
You are a customer support representative for Tindahang Tapat, a digital platform that enables sari-sari stores to order groceries via mobile phone.

## OBJECTIVE
Classify the customer's problem into the appropriate query type and specific issue category using the provided context as reference.

## CLASSIFICATION FRAMEWORK

### AVAILABLE QUERY TYPES
```
{query_type_choices}
```

### AVAILABLE SPECIFIC ISSUE TYPES
```
{specific_issue_choices}
```

## CLASSIFICATION GUIDELINES
- Use the **CONTEXT** as your primary reference for accurate classification
- Select the **most specific** category that matches the problem
- Choose only **one** QUERY TYPE and **one** SPECIFIC ISSUE TYPE
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
- Provide **ONLY** the classifications below
- STRICTLY no additional text, explanations, or formatting
- Use **NONE** if no suitable category exists

## OUTPUT FORMAT
Provide exactly two sections with no additional commentary, explanations, or formatting:

QUERY TYPE:
[ONLY ONE selected category from AVAILABLE QUERY TYPES or NONE]

SPECIFIC ISSUE TYPE:
[ONLY ONE selected category from AVAILABLE SPECIFIC ISSUE TYPES or NONE]
"""

    prompt_template_no_context = """
You are a customer support representative for Tindahang Tapat, a digital platform that enables sari-sari stores to order groceries via mobile phone.

## OBJECTIVE
Classify the customer's problem into the appropriate query type and specific issue category.

## CLASSIFICATION FRAMEWORK

### AVAILABLE QUERY TYPES
```
{query_type_choices}
```

### AVAILABLE SPECIFIC ISSUE TYPES
```
{specific_issue_choices}
```

## CLASSIFICATION GUIDELINES
- Select the **most specific** category that matches the problem
- Choose only **one** QUERY TYPE and **one** SPECIFIC ISSUE TYPE
- If uncertain between categories, prioritize the one that best captures the core functionality issue
- Output **NONE** if no category adequately fits the problem


## CUSTOMER PROBLEM
```
{problem}
```

## OUTPUT REQUIREMENTS
- Provide **ONLY** the classifications below
- STRICTLY no additional text, explanations, or formatting
- Use **NONE** if no suitable category exists

## OUTPUT FORMAT
Provide exactly two sections with no additional commentary, explanations, or formatting:

QUERY TYPE:
[ONLY ONE selected category from AVAILABLE QUERY TYPES or NONE]

SPECIFIC ISSUE TYPE:
[ONLY ONE selected category from AVAILABLE SPECIFIC ISSUE TYPES or NONE]
"""

    if use_context is True:
        prompt = prompt_template.format(query_type_choices=query_type_choices, 
                                        specific_issue_choices=specific_issue_choices,
                                        context=context,
                                        problem=problem
                                    )
    else:
        prompt = prompt_template_no_context.format(query_type_choices=query_type_choices, 
                                        specific_issue_choices=specific_issue_choices,
                                        problem=problem
                                    )

    return prompt

def answer_ticket_prompt(problem: str, context: str, use_context: bool = True) -> str:
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

    prompt_template_no_context = """
## ROLE & CONTEXT
You are a customer support representative for Tindahang Tapat, a digital platform enabling sari-sari stores to order groceries via mobile phone. Generate appropriate responses using only the provided knowledge base context.

## PRIMARY OBJECTIVE
Deliver accurate, helpful solutions to customer problems, maintaining professional service standards.

## RESPONSE METHODOLOGY

### Content Requirements
- **Solution focus**: Prioritize actionable steps and clear guidance
- **Completeness**: Address all aspects of the customer's problem

### Communication Standards
- **Professional tone**: Courteous, confident, and solution-oriented
- **Filipino market awareness**: Use terminology familiar to sari-sari store owners
- **Clarity**: Avoid technical jargon; use simple, direct language
- **Brevity**: Concise responses (2-4 sentences) that fully address the issue

### Customer Problem
```
{problem}
```

## RESPONSE GENERATION RULES

### Content Validation
1. **Completeness check**: Verify all problem aspects are addressed
2. **Tone check**: Maintain professional, helpful customer service voice

### Quality Standards
- **Actionable**: Include specific steps
- **Comprehensive**: Address root cause in sufficient detail
- **Preventive**: Mention prevention tips
- **Follow-up ready**: Set clear expectations for next steps if needed
- ANSWER should be in English

## OUTPUT REQUIREMENTS
Provide exactly two sections with no additional commentary, explanations, or formatting:

ANSWER
[Complete customer response - 2-4 sentences addressing their problem with specific, actionable guidance]

PRIORITY  
[Single priority level: Low, Medium, High, or Urgent]
"""

    if use_context is True:
        prompt = prompt_template.format(context=context, problem=problem)
    else:
        prompt = prompt_template_no_context.format(problem=problem)
    
    return prompt

def llm_response(prompt: str, temperature: int = 1) -> str:

    response = gemini_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            maxOutputTokens=1024
        )
    )
    answer = response.candidates[0].content.parts[0].text
    
    # response = groq_client.chat.completions.create(
    #     model="meta-llama/llama-4-maverick-17b-128e-instruct",
    #     messages=[
    #     {
    #         "role": "user",
    #         "content": prompt,
    #     }
    #     ],
    #     temperature=temperature,
    #     max_completion_tokens=1024,
    #     top_p=1,
    #     stream=True,
    #     stop=None,
    # )

    # answer = ""
    # for chunk in response:
    #     content = chunk.choices[0].delta.content or ""
    #     answer += content

    # response = ollama.generate(
    #     model="mistral", 
    #     prompt=prompt, 
    #     options={
    #         "temperature": temperature,
    #         # "top_p": 0.9,
    #         # "stop": ["\n"],
    #         # "num_predict": 100
    #     }
    # )
    # answer = response.response.strip()

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

def get_payload(ticket_description) -> dict:
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
    for ticket_id in range(523, 526):
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

        parsed_conversations = get_all_conversations(ticket_id, domain, headers, auth, ticket_description)
        full_text_conversation = parsed_conversations['full_text_conversation']

        problem = None
        resolution = None

        while problem is None or resolution is None:
            problem, resolution = get_problem_and_resolution(full_text_conversation)

        payload['problem'] = problem
        payload['resolution'] = resolution

        print(f'Problem: {problem}\n\nResolution: {resolution}')
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

def split_problem_and_resolution(text) -> tuple[str, str]:
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

def split_query_and_issue_type(text: str) -> tuple[str, str]:
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

def split_answer_and_priority(text: str) -> tuple[str, str]:
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

def get_problem(full_text_conversation) -> str:
    prompt_template = """
```
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
- **Generic applicability**: Remove ALL unique identifiers while preserving problem essence
- **Search optimization**: Include keywords support agents would use to find solutions

## MANDATORY STANDARDIZATION RULES

### CRITICAL: Product Normalization
**ALWAYS replace ALL specific product names with "items"**
- "Nescafe 3-in-1 sachets" → "items"
- "Tide powder 1kg" → "items"
- "Lucky Me noodles" → "items"
- "Coca-Cola 1.5L" → "items"
- "Pantene shampoo" → "items"
- "Kopiko coffee" → "items"
- "Maggi seasoning" → "items"
- ANY brand name or specific product → "items"

### Other Standardizations
**Customer References**
- Any customer name → "customer"
- Any store name/location → "customer's store"
- Any order number → "customer order"
- Any account details → "customer account"

**Technical Terms**
- "App crashed/froze/stopped" → "mobile application became unresponsive"
- "Payment failed/declined/error" → "payment processing failed"
- "Can't login/access" → "authentication failed"
- "Won't load/loading" → "content failed to load"

**Process References**
- "Ordering/checkout process" → "order placement process"
- "Delivery tracking" → "order status tracking"
- "Cart/basket" → "shopping cart"

### Absolute Exclusions
**NEVER include:**
- Any brand names, product names, or specific item descriptions
- Customer names, store names, locations
- Order numbers, account IDs, reference codes
- Agent names or responses
- Timestamps or dates
- Emotional language ("frustrated", "angry", "disappointed")
- Greetings or pleasantries
- Multiple explanations of the same issue

## CONVERSATION DATA
```
{conversation}
```

## CRITICAL INSTRUCTIONS
1. Read the conversation carefully
2. Identify the PRIMARY technical/functional problem
3. Apply ALL standardization rules - especially product normalization
4. Write EXACTLY 2-3 sentences describing the problem
5. Double-check that NO specific product names remain
6. Ensure the problem statement is generic and searchable

## OUTPUT REQUIREMENTS
- Start immediately with the problem (no introduction)
- Use only standardized terminology
- End with a period
- If no clear problem exists, output: "No identifiable problem."
- If multiple problems exist, focus on the most critical one

## EXPECTED OUTPUT FORMAT
Provide ONLY the section below. No explanations, commentary, or additional text.
[Standardized problem statement with ALL products normalized as "items"]
"""
    prompt = prompt_template.format(conversation=full_text_conversation)
    answer = llm_response(prompt, temperature=0.4)

    return answer

def get_problem_and_resolution(full_text_conversation) -> tuple[str, str]:
    prompt_template = """
## CONTEXT
You are analyzing customer support conversations for Tindahang Tapat, a digital platform enabling sari-sari stores to order groceries via mobile phone. Your task is to extract standardized problem-resolution pairs for a RAG knowledge base.

## ANALYSIS REQUIREMENTS

### Problem Statement Extraction
Create a **searchable, generic problem description** that:
- **Focuses on function over emotion**: Describe what failed/broke, not how customers felt
- **Uses standard terminology**: Platform features, order processes, payment methods, product categories
- **Removes ALL specificity**: No customer names, order numbers, specific brands, or timestamps
- **Enables similarity matching**: Use consistent vocabulary for similar issues
- **Length**: Exactly 2-3 sentences

### Resolution Documentation  
Provide an **actionable solution summary** that:
- **Details specific steps**: What the agent did to resolve the issue
- **Includes verification**: How resolution was confirmed
- **Notes preventive measures**: Steps to avoid recurrence (if applicable)
- **States clear outcome**: "RESOLVED" with method, or "UNRESOLVED" with next steps
- **Length**: Exactly 2-3 sentences

## MANDATORY STANDARDIZATION RULES

### CRITICAL: Product Normalization
**ALWAYS replace ALL specific product names, brands, and descriptions with "items"**
- "Nescafe 3-in-1 sachets" → "items"
- "Tide powder 1kg" → "items"
- "Lucky Me instant noodles" → "items"
- "Coca-Cola 1.5L bottles" → "items"
- "Pantene shampoo 200ml" → "items"
- "instant coffee product" → "items"
- "laundry detergent" → "items"
- "beverage products" → "items"
- ANY specific product reference → "items"

### Other Critical Normalizations
**Customer References**
- Any customer name → "customer"
- Any store name/location → "customer's store"
- Any order number/ID → "customer order"
- Any account details → "customer account"
- Specific amounts → "order amount" or "payment amount"

**Technical Terms**
- "App crashed/froze/stopped/won't work" → "mobile application became unresponsive"
- "Payment failed/declined/error/won't process" → "payment processing failed"
- "Can't login/access/sign in" → "authentication failed"
- "Won't load/loading/stuck loading" → "content failed to load"
- "System error/bug/glitch" → "system malfunction occurred"

**Process References**
- "Ordering/checkout/purchasing process" → "order placement process"
- "Delivery tracking/monitoring" → "order status tracking"
- "Cart/basket/shopping list" → "shopping cart"
- "Account setup/registration" → "account creation process"

### Language Standardization
- **Payment issues**: "payment processing", "transaction failure", "payment method error"
- **Delivery problems**: "delivery scheduling issue", "logistics coordination failure", "order fulfillment problem"
- **Product concerns**: "item availability issue", "inventory discrepancy", "catalog display problem"
- **Account access**: "login authentication failure", "account verification issue", "profile access problem"
- **App functionality**: "mobile app malfunction", "platform feature failure", "system functionality error"

### Absolute Exclusions
**NEVER include:**
- Any brand names, specific product names, or detailed item descriptions
- Customer names, store names, specific locations
- Order numbers, account IDs, reference codes, transaction IDs
- Agent names, department names, company structure references
- Timestamps, dates, or time-specific information
- Monetary amounts (use "order amount" instead)
- Emotional language ("frustrated", "angry", "happy", "satisfied")
- Greetings, sign-offs, pleasantries, or social conversation
- Repetitive confirmations or multiple explanations of same issue

## CONVERSATION DATA
{conversation}

## CRITICAL INSTRUCTIONS
1. Read the conversation completely
2. Identify the PRIMARY functional problem and its resolution
3. Apply ALL standardization rules - especially product normalization to "items"
4. Write exactly 2-3 sentences for each section
5. Verify NO specific product names, brands, or identifiers remain
6. Ensure language is generic and searchable

## OUTPUT REQUIREMENTS
Provide ONLY the sections below. No explanations, commentary, or additional text.

CORE PROBLEM STATEMENT:
[Generic, searchable problem description with ALL products normalized as "items"]

RESOLUTION:
[Actionable solution steps and final status: RESOLVED or UNRESOLVED with next steps]
"""
    prompt = prompt_template.format(conversation=full_text_conversation)

    problem = None
    resolution = None

    while problem is None or resolution is None:
        response = llm_response(prompt, temperature=0.4)
        problem, resolution = split_problem_and_resolution(response)

    return problem, resolution

def get_contact(requestor_id, domain, headers, auth):
    contact_url = f"https://{domain}.freshdesk.com/api/v2/contacts/{requestor_id}"
    contact_response = requests.get(contact_url, headers=headers, auth=auth)
    contact = contact_response.json()

    return contact

def get_all_conversations(ticket_id, domain, headers, auth, ticket_description):
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

    parsed_conversations = parse_freshdesk_conversations(conversations, ticket_description)

    return parsed_conversations

def parse_freshdesk_conversations(conversations, ticket_description) -> dict:
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

def translate_conversation(conversation: str) -> str:
    prompt_template = """
INSTRUCTIONS:
You are a professional translator. Translate the following CONVERSATION from languages (English, Tagalog, Bikol, or a mix) into clear, natural English.

REQUIREMENTS:
- Preserve all "Customer:" and "Agent:" labels exactly as they appear
- Maintain the original meaning, tone, and context of each message
- Use natural, conversational English that sounds authentic
- Keep the same dialogue structure and flow
- Do not add explanations, summaries, or commentary
- If text is already in English, keep it unchanged unless it needs clarity improvements

CONVERSATION:
{conversation}

OUTPUT:
"""
    prompt = prompt_template.format(conversation=conversation)
    translated_conversation = llm_response(prompt)

    print(f'Translated: {translated_conversation}')

    return translated_conversation