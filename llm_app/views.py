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

openai_api_key = os.environ.get('OPENAI_API_KEY')

model_names = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "chatgpt-4o-latest"]
# model_names = ["gpt-4o-mini"]

import ollama
ollama.pull("mistral")

from qdrant_client import models, QdrantClient
qd_client = QdrantClient('http://localhost:8333')
# collection_name = "freshdesk-tickets-rag"
collection_name = "strict-resolution-rag"



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

        parsed_conversations = get_all_conversations(ticket_id, ticket_description)
        full_text_conversation = parsed_conversations['full_text_conversation']

        problem, resolution = get_problem_and_resolution(full_text_conversation)
        doc['problem'] = problem
        doc['resolution'] = resolution

        print(full_text_conversation)
        print('-----')

    
    return render(request, 'hello.html')

def latest_tickets(request):
    tickets_url = f"https://{domain}.freshdesk.com/api/v2/tickets?order_by=created_at&order_type=desc&per_page=3"
    ticket_response = requests.get(tickets_url, headers=headers, auth=auth)
    tickets = ticket_response.json() if ticket_response.status_code == 200 else []

    ticket_list = []

    for ticket_id in range(23446, 23450):
    # for ticket_id in range(23446, 23447):
    # for ticket in tickets:
    #     ticket_id = ticket.get('id')
        description_url = f"https://{domain}.freshdesk.com/api/v2/tickets/{ticket_id}"
        ticket_description = requests.get(description_url, headers=headers, auth=auth).json()

        requestor_id = ticket_description['requester_id']
        contact = get_contact(requestor_id, domain, headers, auth)

        contact_name = contact.get('name')
        print(f"Name: {contact_name}")

        payload = get_payload(ticket_description)

        parsed_conversations = get_all_conversations(ticket_id, ticket_description)
        full_text_conversation = parsed_conversations['full_text_conversation']
        customer_conversation_text = parsed_conversations['customer_conversation_text']
        agent_conversation_text = parsed_conversations['agent_conversation_text']

        # can change to full text convo

        model_answers = []
        for model in model_names:
            print(f"Model: {model}")

            problem = get_problem(customer_conversation_text, model=model)
            payload['problem'] = problem
            print(f'Core Problem: {problem}')
            rag_classification
            query_type, specific_issue_type = rag_classification(problem, model=model)

            answer, priority = rag(customer_conversation_text, problem, query_type, model=model)
            model_answers.append({
                "model": model,
                "problem": problem,
                "query_type": query_type,
                "specific_issue_type": specific_issue_type,
                "answer": answer,
                "priority": priority,
            })

        # query_type_no_context, specific_issue_type_no_context = rag_classification(problem, use_context = False)
        # answer_no_context, priority_no_context = rag(problem, query_type, use_context = False)

        # print(f'Query type: {query_type}')
        # print(f'Specific issue type: {specific_issue_type}')
        # print(f'Priority: {priority}')
        # print(f'Answer: {answer}')
        # print('**********')
        
        ticket_list.append({
            'id': ticket_id,
            'name': contact_name,
            'conversation': parsed_conversations['ordered_conversation_details'],
            'payload': payload,

            # rag
            # 'query_type': query_type,
            # 'specific_issue_type': specific_issue_type,
            # 'priority': priority,
            # 'generated_response': answer,

            'model_answers': model_answers

            # non-rag
            # 'query_type_no_context': query_type_no_context, 
            # 'specific_issue_type_no_context': specific_issue_type_no_context,
            # 'answer_no_context': answer_no_context,
            # 'priority_no_context': priority_no_context,
        })
        
        print('----------')

    return render(request, 'response.html', {'tickets': ticket_list,})

def rag_classification(problem: str, use_context: bool = True, model: str = "gpt-4o-mini") -> str:
    print("RAG CLASSIFICATION")
    print("Searching relevant documents...")

    search_response = rrf_search(problem)
    prompt = classification_prompt(problem, search_response, use_context)

    print(f'Search results:\n{search_response}\n')

    query_type = ""
    specific_issue_type = ""

    while query_type not in query_type_list or specific_issue_type not in specific_issue_choices:
        print("Generating CLASSIFICATION...")
        response = llm_response(prompt, model=model)
        query_type, specific_issue_type = split_query_and_issue_type(response)

        print(f'{query_type} and {specific_issue_type}\n\n')
        query_type = str(query_type)
        specific_issue_type = str(specific_issue_type)

    return query_type, specific_issue_type

def rag(customer_chat: str, problem: str, query_type: str, use_context: bool = True, model: str = "gpt-4o-mini") -> str:
    print("RAG for ANSWER")
    print("Searching relevant documents...")

    search_response = rrf_filter_search(problem, query_type)
    prompt = answer_ticket_prompt(customer_chat, problem, search_response, use_context)

    print(f'Search results:\n{search_response}\n')

    priority = ""
    while priority not in ["Low", "Medium", "High", "Urgent", "NONE"]:
        print("Generating ANSWER...")
        response = llm_response(prompt, model=model)
        answer, priority = split_answer_and_priority(response)

        print(f'Answer: {answer}\n and \nPriority: {priority}\n')
        print(f'Response: {response}')
        if priority is None:
            print("Priority is none.")
            priority = ""

    return answer, priority

def rrf_search(problem: str, operation: str = "normal", limit: int = 5) -> str:
    search_results = qd_client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                # operation filter
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="operation",
                            match=models.MatchValue(value=operation),
                        ),
                    ]
                ),
                query=models.Document(
                    text=problem,
                    model="BAAI/bge-large-en-v1.5"
                ),
                using="baai-large",
                limit=limit,
            ),
            models.Prefetch(
                # operation filter
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="operation",
                            match=models.MatchValue(value=operation),
                        ),
                    ]
                ),
                query=models.Document(
                    text=problem,
                    model="Qdrant/bm25"
                ),
                using="bm25",
                limit=limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    context = ""
    for result in search_results.points:
        problem = result.payload['problem']
        resolution = result.payload['resolution'] 
        context += f'Problem: {problem}\nResolution: {resolution}\n---------------\n'

    if context == "":
        context = "No relevant data found."

    return context

def rrf_filter_search(problem: str, query_type: str, operation: str = "normal", limit: int = 5) -> str:
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
                        # operation filter
                        models.FieldCondition(
                            key="operation",
                            match=models.MatchValue(value=operation),
                        ),
                    ]
                ),
                query=models.Document(
                    text=problem,
                    model="BAAI/bge-large-en-v1.5"
                ),
                using="baai-large",
                limit=limit,
            ),
            models.Prefetch(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="query_type",
                            match=models.MatchValue(value=query_type),
                        ),
                        # operation filter
                        models.FieldCondition(
                            key="operation",
                            match=models.MatchValue(value=operation),
                        ),
                    ]
                ),
                query=models.Document(
                    text=problem,
                    model="Qdrant/bm25"
                ),
                using="bm25",
                limit=limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    context = ""
    for result in search_results.points:
        problem = result.payload['problem']
        resolution = result.payload['resolution']
        priority = result.payload['priority']
        context += f'Problem: {problem}\nResolution: {resolution}\nPriority: {priority}\n\n'

    if context == "":
        context = "No relevant data found."

    # context_template = """
    # """
    # context = context_template.format()

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
**CRITICAL**: Use the exact keyword given in the available QUERY TYPES and SPECIFIC ISSUE TYPES. Always return output in exactly the following format with no additional commentary, explanations, or formatting:

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

def answer_ticket_prompt(customer_chat: str, problem: str, context: str, use_context: bool = True) -> str:
    prompt_template = """
## ROLE & CONTEXT
You are a customer support representative for Tindahang Tapat, a digital platform enabling sari-sari stores to order groceries via mobile phone. Generate appropriate responses using STRICTLY AND ONLY the provided knowledge base context.

## PRIMARY OBJECTIVE
Deliver accurate, helpful solutions to customer problems based EXCLUSIVELY on available CONTEXT information, while maintaining professional service standards.

## CRITICAL CONTEXT ADHERENCE RULES
**ABSOLUTE REQUIREMENT**: Your response must be based ONLY on information explicitly provided in the CONTEXT section below. The customer problem and chat data provide situational understanding, but solutions must come from CONTEXT.

### Context Usage Protocol
- **ONLY use information from CONTEXT**: No external knowledge, assumptions, or inferences
- **Direct correlation required**: Solution must directly address the customer problem using context information
- **No creative interpretation**: If context doesn't explicitly cover the problem, use fallback response
- **Complete context dependency**: Even common sense solutions must be supported by context content

### Content Requirements
- **Context fidelity**: Use ONLY information from the provided CONTEXT
- **Solution focus**: Prioritize actionable steps and clear guidance FROM CONTEXT
- **Completeness**: Address all aspects of the customer's problem when context allows
- **Accuracy**: Never infer or assume details not explicitly stated in context

### Communication Standards
- **Professional tone**: Courteous, confident, and solution-oriented
- **Filipino market awareness**: Use terminology familiar to sari-sari store owners
- **Clarity**: Avoid technical jargon; use simple, direct language
- **Brevity**: Concise responses (2-4 sentences) that fully address the issue

### Mandatory Fallback Protocol
**STRICT RULE**: If the CONTEXT does not contain information that directly addresses the customer problem, you MUST use the fallback response. Do not attempt to provide general advice or solutions not found in the context.

**Fallback Response**: *"I don't have specific information about this issue in my current resources. Please contact our support team at [support contact] for immediate assistance with your concern."*

## INPUT DATA

### Knowledge Base CONTEXT
```
{context}
```

### Customer Problem (Primary Issue)
```
{problem}
```

### Customer Conversation Data (Additional Context)
```
{customer_chat}
```

## RESPONSE GENERATION PROCESS

### Step 1: Context Analysis
1. **Read the customer problem** to understand the core issue
2. **Search the CONTEXT** for information that directly addresses this problem
3. **Verify relevance**: Ensure context information specifically applies to the customer's situation
4. **Assess completeness**: Determine if context provides sufficient information for a complete solution

### Step 2: Solution Validation
1. **Context check**: Confirm solution exists in provided context - DO NOT INVENT ANY SOLUTION
2. **Direct mapping**: Ensure each part of your response references specific context information
3. **Completeness check**: Verify all problem aspects can be addressed with available context
4. **Fallback trigger**: If context is insufficient or irrelevant, use mandatory fallback response

### Step 3: Response Construction
1. **Context-only content**: Build response using exclusively context information
2. **Problem alignment**: Ensure response directly addresses the customer's stated problem
3. **Actionable guidance**: Include specific steps when available in context
4. **Professional tone**: Maintain helpful, customer service voice

## QUALITY STANDARDS
- **Actionable**: Include specific steps when solutions are available in context
- **Comprehensive**: Address root cause when context provides sufficient detail
- **Preventive**: Mention prevention tips if included in context
- **Follow-up ready**: Set clear expectations for next steps if provided in context

## VALIDATION CHECKLIST
Before providing your response, verify:
- [ ] Does the CONTEXT contain information directly relevant to the customer problem?
- [ ] Can I provide a complete solution using ONLY the context information?
- [ ] Have I avoided adding any information not found in the context?
- [ ] If context is insufficient, have I used the mandatory fallback response?
- [ ] Does my response directly address the customer's primary concern?

## OUTPUT REQUIREMENTS
**CRITICAL**: Always return output in exactly the following format with no additional commentary, explanations, or formatting:

ANSWER:  
[Provide a complete customer response based ONLY on context - 2-4 sentences addressing their problem with specific, actionable guidance in Taglish. If the context does not contain relevant information that directly addresses the customer problem, use exactly: "ANSWER: I don't have specific information about this issue in my current resources. Please contact our support team at [support contact] for immediate assistance with your concern."]

PRIORITY:  
[Provide one of the following values only: Low, Medium, High, Urgent, or NONE. If the fallback answer was used, return PRIORITY as NONE.]
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
With no additional commentary, explanations, or formatting, always return output in the following two sections:

ANSWER:  
[Provide a complete customer response based on context - 2-4 sentences in TAGLISH addressing their problem with specific, actionable guidance in Taglish. If the context does not include relevant information, use exactly: "ANSWER: I don't have specific information about this issue in my current resources. Please contact our support team at [support contact] for immediate assistance with your concern."]

PRIORITY:  
[Provide one of the following values only: Low, Medium, High, Urgent, or NONE. If the fallback answer was used, return PRIORITY as NONE.]
"""

    if use_context is True:
        prompt = prompt_template.format(context=context, customer_chat=customer_chat, problem=problem)
    else:
        prompt = prompt_template_no_context.format(problem=problem)
    
    return prompt

def llm_response(prompt: str, temperature: int = 1, model: str = "gpt-4o-mini", llm_provider: str = "openai") -> str:

    # response = gemini_client.models.generate_content(
    #     model='gemini-2.0-flash',
    #     contents=prompt,
    #     config=types.GenerateContentConfig(
    #         temperature=temperature,
    #         maxOutputTokens=1024
    #     )
    # )
    # answer = response.candidates[0].content.parts[0].text
    
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

    if llm_provider == "groq":
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
            {
                "role": "user",
                "content": prompt,
            }
            ],
            temperature=temperature,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        answer = ""
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            answer += content

    if llm_provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 250,
            "temperature": temperature
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        answer = response.json()
        answer = answer["choices"][0]["message"]["content"]

    return answer

def embed_tickets(request):
    tickets_payload = []

    with open("data_openai.json", "r", encoding="utf-8") as f:
        tickets_payload = json.load(f)

    print('Initializing qdrant client')  
    
    qd_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "baai-large": models.VectorParams(
                size = 1024,
                distance = models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier = models.Modifier.IDF
            )
        },
    )

    points = []
    for ticket in tickets_payload:
        if "unresolved" not in ticket['resolution'].lower():
            point = models.PointStruct(
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
            points.append(point)
            print(ticket['id'])

    print('Uploading data points...')
    qd_client.upsert(
        collection_name=collection_name,
        points=points,
    )

    print('Indexing payload...')

    qd_client.create_payload_index(
        collection_name=collection_name,
        field_name="operation",
        field_schema="keyword"
    )

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
    # for ticket_id in range(0, 101):
    for ticket_id in range(23446, 23447):
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

        parsed_conversations = get_all_conversations(ticket_id, ticket_description)
        full_text_conversation = parsed_conversations['full_text_conversation']

        payload['conversation'] = full_text_conversation
        print(full_text_conversation)

        problem, resolution, operation = get_problem_and_resolution(payload, llm_provider="openai")

        payload['problem'] = problem
        payload['resolution'] = resolution
        payload['operation'] = operation

        tickets_payload.append(payload) 

        # file_path = "documents.json"
        file_path = "data_openai.json"
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
        (.*?)                      
        [#\*\s]*OPERATION\s+TYPE[:\s\#\*]*
        (.*)                          
    """

    match = re.search(pattern, text, re.DOTALL | re.VERBOSE)
    if match:
        problem = match.group(1).strip()
        resolution = match.group(2).strip()
        operation = match.group(3).strip()
        return problem, resolution, operation
    else:
        return None, None, None

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

def get_problem(customer_conversation_text: str, model: str = "gpt-4o-mini") -> str:
#     prompt_template = """
# ```
# ## CONTEXT
# You are extracting customer problems from Tindahang Tapat support conversations. Tindahang Tapat is a digital platform enabling sari-sari stores to order groceries via mobile phone. Your output will be stored in a vector database for RAG-based support automation.

# ## ANALYSIS OBJECTIVE
# Extract the **primary functional problem** that prompted the customer to contact support, formatted for optimal knowledge base retrieval.

# ## PROBLEM STATEMENT REQUIREMENTS

# ### Structure (2-3 sentences maximum)
# 1. **Issue identification**: What specific functionality failed or caused confusion
# 2. **Context**: Under what circumstances the problem occurs
# 3. **Impact**: How it affects the customer's workflow (optional, if relevant)

# ### Quality Standards
# - **Functional focus**: Describe system behavior, not customer emotions
# - **Actionable language**: Use verbs that describe what went wrong
# - **Consistent terminology**: Apply standardized platform vocabulary
# - **Generic applicability**: Remove ALL unique identifiers while preserving problem essence
# - **Search optimization**: Include keywords support agents would use to find solutions

# ## MANDATORY STANDARDIZATION RULES

# ### CRITICAL: Product Normalization
# **ALWAYS replace ALL specific product names with "items"**
# - "Nescafe 3-in-1 sachets" → "items"
# - "Tide powder 1kg" → "items"
# - "Lucky Me noodles" → "items"
# - "Coca-Cola 1.5L" → "items"
# - "Pantene shampoo" → "items"
# - "Kopiko coffee" → "items"
# - "Maggi seasoning" → "items"
# - ANY brand name or specific product → "items"

# ### Other Standardizations
# **Customer References**
# - Any customer name → "customer"
# - Any store name/location → "customer's store"
# - Any order number → "customer order"
# - Any account details → "customer account"

# **Technical Terms**
# - "App crashed/froze/stopped" → "mobile application became unresponsive"
# - "Payment failed/declined/error" → "payment processing failed"
# - "Can't login/access" → "authentication failed"
# - "Won't load/loading" → "content failed to load"

# **Process References**
# - "Ordering/checkout process" → "order placement process"
# - "Delivery tracking" → "order status tracking"
# - "Cart/basket" → "shopping cart"

# ### Absolute Exclusions
# **NEVER include:**
# - Any brand names, product names, or specific item descriptions
# - Customer names, store names, locations
# - Order numbers, account IDs, reference codes
# - Agent names or responses
# - Timestamps or dates
# - Emotional language ("frustrated", "angry", "disappointed")
# - Greetings or pleasantries
# - Multiple explanations of the same issue

# ## CONVERSATION DATA
# ```
# {conversation}
# ```

# ## CRITICAL INSTRUCTIONS
# 1. Read the conversation carefully
# 2. Identify the PRIMARY technical/functional problem
# 3. Apply ALL standardization rules - especially product normalization
# 4. Write EXACTLY 2-3 sentences describing the problem
# 5. Double-check that NO specific product names remain
# 6. Ensure the problem statement is generic and searchable

# ## OUTPUT REQUIREMENTS
# - Start immediately with the problem (no introduction)
# - Use only standardized terminology
# - End with a period
# - If no clear problem exists, output: "No identifiable problem."
# - If multiple problems exist, focus on the most critical one

# ## EXPECTED OUTPUT FORMAT
# Provide ONLY the section below. No explanations, commentary, or additional text.
# [Standardized problem statement with ALL products normalized as "items"]
# """

    prompt_template = """
```
## CONTEXT
You are extracting customer problems from Tindahang Tapat support conversations. Tindahang Tapat is a digital platform enabling sari-sari stores to order groceries via mobile phone. Your output will be stored in a vector database for RAG-based support automation.

## ANALYSIS OBJECTIVE
Extract the **primary functional problem** that prompted the customer to contact support, formatted for optimal knowledge base retrieval.

## CUSTOMER PERSPECTIVE FOCUS
**CRITICAL**: Focus exclusively on the customer's perspective when analyzing the conversation. The conversation data may include messages from both customers and customer service agents. When generating the problem statement:
- **Extract only customer queries/concerns** - ignore agent responses and solutions
- **Identify what the customer is asking about** - not what the agent is explaining
- **Focus on the customer's original issue** - not the troubleshooting process
- **Capture the customer's experience of the problem** - not the technical resolution
- **STRICT RULE**: Only include information explicitly stated in customer messages - NO assumptions or inferences about impact or consequences

## PROBLEM STATEMENT REQUIREMENTS
### Structure (2-3 sentences maximum)
1. **Issue identification**: What specific functionality failed or caused confusion
2. **Context**: Under what circumstances the problem occurs
3. **Impact**: How it affects the customer's workflow (ONLY if explicitly stated in conversation)

### Quality Standards
- **Functional focus**: Describe system behavior, not customer emotions
- **Actionable language**: Use verbs that describe what went wrong
- **Consistent terminology**: Apply standardized platform vocabulary
- **Generic applicability**: Remove ALL unique identifiers while preserving problem essence
- **Search optimization**: Include keywords support agents would use to find solutions
- **NO ASSUMPTIONS**: Only include information explicitly stated in the customer's messages

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
- Any specific store name → "customer's store"
- Any order number → "customer order"
- Any account details → "customer account"

**Location Preservation**
- **PRESERVE location keywords** such as city names, regions, areas, barangays, municipalities
- Keep location-related terms that provide context for delivery or service areas
- Example: "Quezon City," "Manila," "Cebu," "Davao," "Makati," etc. should be retained

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
- Customer names, specific store names
- Order numbers, account IDs, reference codes
- Agent names or responses
- Timestamps or dates
- Emotional language ("frustrated", "angry", "disappointed")
- Greetings or pleasantries
- Multiple explanations of the same issue
- **ASSUMPTIONS OR INFERENCES** about impact, consequences, or effects not explicitly stated by the customer
- Statements about what the customer "needs," "wants," or "expects" unless directly quoted
- Impact statements that are not explicitly mentioned in customer messages

**PRESERVE (do not exclude):**
- Geographic locations (cities, regions, areas, barangays, municipalities) when relevant to the problem

## CONVERSATION DATA
```
{conversation}
```

## CRITICAL INSTRUCTIONS
1. Read the conversation carefully
2. **Focus ONLY on customer messages** - ignore agent responses
3. Identify the PRIMARY technical/functional problem from the customer's perspective
4. Apply ALL standardization rules - especially product normalization
5. Write EXACTLY 2-3 sentences describing the problem
6. Double-check that NO specific product names remain
7. Ensure the problem statement is generic and searchable
8. **VERIFY**: Remove all assumptions, inferences, or impact statements not explicitly stated by the customer

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

    prompt = prompt_template.format(conversation=customer_conversation_text)
    answer = llm_response(prompt, temperature=0.4)

    return answer

def get_problem_and_resolution(payload, llm_provider: str = "groq") -> tuple[str, str]:
#     prompt_template = """
# ## ROLE
# You are analyzing customer support conversations for Tindahang Tapat, a digital platform enabling sari-sari stores to order groceries via mobile phone. Your task is to extract standardized problem-resolution pairs for a RAG knowledge base.

# The CONVERSATION DATA contains messages from the customer and the agent (customer service representative).

# ## CONVERSATION DATA
# {conversation}

# ## QUERY TYPE - This the general classification of the customer's concern, identified by the agent.
# {query_type}

# ## SPECIFIC ISSUE - This is the specific classification of the customer's concern, identified by the agent.
# {specific_issue}

# ## TASK
# Extract the CORE PROBLEM STATEMENT and RESOLUTION based on the given CONVERSATION DATA, QUERY TYPE, and SPECIFIC ISSUE.

# ## OUTPUT REQUIREMENTS

# ### CORE PROBLEM STATEMENT
# - Must be based on the main concern of the customer
# - If multiple concerns exist, focus on the main one. ONLY ONE customer problem should be included in the CORE PROBLEM STATEMENT.
# - Do NOT include assumptions about possible impact to the customer
# - Focus only on problems explicitly stated by the customer
# - If no problem exists, output: "No identifiable problem"
# - Length: Exactly 2-3 sentences

# ### RESOLUTION
# - Must be based on concrete actions taken by the agent to resolve the customer's main issue
# - Promises, reassurances, explanations, or status updates do NOT count as resolution
# - If no concrete resolution exists, output: "Unresolved"
# - Length: Exactly 2-3 sentences for resolved cases, single word "Unresolved" for unresolved cases

# ## WHAT CONSTITUTES CONCRETE RESOLUTION
# **RESOLVED status requires concrete action taken by the agent:**
# - Refund processed and confirmed
# - Replacement item dispatched
# - Technical issue fixed (account unlocked, payment processed, etc.)
# - Correct information provided with verification
# - Problem directly addressed with measurable outcome
# - Compensation or credits applied to account

# ## WHAT DOES NOT CONSTITUTE RESOLUTION
# **The following are NOT resolutions and should be marked "Unresolved":**
# - Promises or reassurances ("we will try", "team will attempt", "we will follow up")
# - Explanations without solutions (explaining why problem occurred)
# - Information gathering only (asking for details without solving)
# - Escalations without final resolution
# - Apologies without corrective action
# - Status updates without problem resolution
# - Acknowledgments of concerns without action

# ## OBJECTIVITY REQUIREMENTS
# **ELIMINATE ALL EMOTIONAL AND SUBJECTIVE LANGUAGE:**

# ### Prohibited Emotional Language
# - Customer emotions: "frustrated", "angry", "upset", "disappointed", "happy", "satisfied", "annoyed", "worried", "concerned", "stressed"
# - Agent emotions: "apologetic", "sympathetic", "understanding", "helpful", "caring"
# - Emotional reactions: "complained", "expressed dissatisfaction", "was pleased", "felt relieved"
# - Subjective descriptions: "terrible experience", "great service", "poor quality", "excellent response"

# ### Required Objective Language
# - Use factual, neutral verbs: "reported", "stated", "indicated", "requested", "received", "experienced"
# - Focus on actions and events, not feelings about them
# - Describe what happened, not how anyone felt about it
# - Use technical, operational language

# ## NORMALIZATION RULES
# Replace all specific details with standardized terms:

# ### Customer References
# - Any customer name → "customer"
# - Any store name → "customer's store"
# - Any order number/ID → "customer order"
# - Any account details → "customer account"
# - Specific amounts → "order amount" or "payment amount"

# ### CRITICAL: Product Normalization
# **ALWAYS replace ALL specific product names with items**
# - "Nescafe 3-in-1 sachets" → items
# - "Tide powder 1kg" → items
# - "Lucky Me noodles" → items
# - "Coca-Cola 1.5L" → items
# - "Pantene shampoo" → items
# - "Kopiko coffee" → items
# - "Maggi seasoning" → items
# - ANY brand name or specific product → items

# ### Time References
# - Specific dates → "scheduled delivery date"
# - Specific times → "delivery time"
# - Time periods → "delivery window"

# ### Location References
# - Specific addresses → "delivery location"
# - Cities/regions → "delivery area"
# - Landmarks → "delivery location"

# ### External Factors
# - Weather events (storms, typhoons, floods) → "weather conditions"
# - Traffic issues → "traffic conditions"
# - System problems → "technical issues"

# ### Technical Terms
# - "App crashed/froze/stopped/won't work" → "mobile application became unresponsive"
# - "Payment failed/declined/error/won't process" → "payment processing failed"
# - "Can't login/access/sign in" → "authentication failed"
# - "Won't load/loading/stuck loading" → "content failed to load"

# ## CRITICAL INSTRUCTIONS
# 1. Read the conversation completely to identify the main customer concern
# 2. Apply ALL normalization rules - especially product normalization to "items". DO NOT EVER NAME A SPECIFIC PRODUCT BRAND OR THE TYPE OF PRODUCT (e.g Alaska, corned beef, etc.)
# 3. Mark as "RESOLVED" ONLY if concrete action was taken to fix the problem
# 4. If only promises, explanations, or reassurances were given, mark as "Unresolved"
# 5. Remove all specific dates, times, locations, and external factors
# 6. Focus on the functional problem, not emotional context
# 7. Ensure language is generic and searchable for RAG retrieval

# ## OUTPUT FORMAT
# **CRITICAL**: Provide ONLY the section below. No explanations, commentary, or additional text.

# CORE PROBLEM STATEMENT:
# [2-3 sentences describing the main customer concern, normalized]

# RESOLUTION:
# [2-3 sentences describing concrete actions taken by agent, normalized] [Status: RESOLVED] OR [Unresolved]
# """

    prompt_template = """
## ROLE
Analyze customer support conversations for Tindahang Tapat to extract standardized problem-resolution pairs for RAG knowledge base.

## INPUT DATA
CONVERSATION: {conversation}
QUERY TYPE: {query_type}
SPECIFIC ISSUE: {specific_issue}

## OUTPUT REQUIREMENTS

### CORE PROBLEM STATEMENT (2-3 sentences)
- Main customer concern only (if multiple, pick primary)
- Based on explicit customer statements, not assumptions
- Use objective, factual language

### RESOLUTION (2-3 sentences or "Unresolved")
**RESOLVED**: Concrete action taken (refund processed, item dispatched, issue fixed, credits applied)
**Unresolved**: Promises, explanations, escalations, or reassurances only

### OPERATION TYPE
- **typhoon**: Weather-related disruptions, storm mentions, emergency protocols
- **christmas**: Holiday season references, December timeframes, seasonal delays
- **normal**: Default for all other cases
**IMPORTANT**: If weather/storm is mentioned ANYWHERE in the conversation as a cause of service disruption, mark as "typhoon" regardless of current status.

## NORMALIZATION RULES
**Replace the following specifics with generic terms:**
- Names/stores/IDs → "customer", "customer's store", "customer order"
- ALL products → "items" (regardless of brand/type)
- Dates/times → "scheduled delivery date", "delivery time"
- Tech issues → "mobile application became unresponsive", "payment processing failed", "authentication failed"
**Critical - Excluding the aforementioned, keywords should be preserved.**

## PROHIBITED LANGUAGE
Remove all emotional/subjective terms: frustrated, angry, disappointed, apologetic, terrible, great, complained, pleased, etc.
Use neutral verbs: reported, stated, indicated, requested, received, experienced.

## CRITICAL INSTRUCTIONS
1. **Read the ENTIRE conversation** - analyze all messages from both customer and agent
2. **Operation Type Priority**: If ANY weather-related disruption is mentioned (even if resolved), classify as "typhoon"
3. Apply normalization rules but preserve operation type context
4. Mark as "RESOLVED" ONLY if concrete action was taken to fix the problem
5. If only promises, explanations, or reassurances were given, mark as "Unresolved"
6. Remove all specific dates, times, locations, and external factors
7. Focus on the functional problem, not emotional context
8. Ensure language is generic and searchable for RAG retrieval
9. Use objective, neutral language throughout

## OUTPUT FORMAT
# **CRITICAL**: Provide ONLY the section below. No explanations, commentary, or additional text.

CORE PROBLEM STATEMENT:
[normalized 2-3 sentences]

RESOLUTION:
[normalized 2-3 sentences with [Status: RESOLVED]] OR [Unresolved]

OPERATION TYPE:
[normal/typhoon/christmas]
"""

    prompt = prompt_template.format(
        conversation=payload['conversation'], 
        query_type=payload['query_type'], 
        specific_issue=payload['specific_issue'],
    )

    problem = None
    resolution = None
    operation = None

    while (problem is None or resolution is None or operation is None) or ("CORE PROBLEM STATEMENT" in resolution) or (operation not in ["normal", "typhoon", "christmas"]):
        response = llm_response(prompt, temperature=0.3, llm_provider=llm_provider)
        print(response)
        problem, resolution, operation = split_problem_and_resolution(response)
        print(f'\nProblem: {problem}\n\nResolution: {resolution}\n\nOperation: {operation}')
        print('-----')

    return problem, resolution, operation

def get_contact(requestor_id, domain, headers, auth):
    contact_url = f"https://{domain}.freshdesk.com/api/v2/contacts/{requestor_id}"
    contact_response = requests.get(contact_url, headers=headers, auth=auth)
    contact = contact_response.json()

    return contact

def get_all_conversations(ticket_id, ticket_description):
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
    customer_conversation_text = ""
    agent_conversation_text = ""
    ordered_conversation_details = []

    message_data = {
        'id': ticket_description['id'],
        'sequence': 1,
        'body_text': ticket_description['description_text'],
        'timestamp': ticket_description['created_at'],
        'user_id': ticket_description['requester_id'],
        'role': 'Customer'
    }

    message = f'{ticket_description['created_at']} Customer: {message_data['body_text']}'
    full_text_conversation += message + "\n"
    customer_conversation_text += message + "\n"
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
            message = f'{conv['created_at']} Customer: {message_data['body_text']}'
            full_text_conversation += message + "\n" 
            customer_conversation_text += message + "\n"
            message_data['role'] = 'Customer'
        else:
            message = f'{conv['created_at']} Agent: {message_data['body_text']}'
            full_text_conversation += message + "\n" 
            agent_conversation_text += message + "\n"
            message_data['role'] = 'Agent'

        ordered_conversation_details.append(message_data)

    return {
        'full_text_conversation': full_text_conversation,
        'customer_conversation_text': customer_conversation_text,
        'agent_conversation_text': agent_conversation_text,
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