import re
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, field_validator
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import uvicorn
import os
import shutil

# --- Pydantic Models ---
class ParsedQuery(BaseModel):
    age: Optional[int] = Field(None, description="The age of the claimant in years.")
    procedure: Optional[str] = Field(None, description="The medical procedure mentioned in the query (e.g., knee surgery).")
    location: Optional[str] = Field(None, description="The geographical location where the procedure occurred (e.g., Pune).")
    policy_duration_months: Optional[int] = Field(None, description="The duration of the insurance policy in months.")

    @field_validator('age', 'policy_duration_months', mode='before')
    def parse_numeric(cls, v):
        if isinstance(v, str):
            v = v.strip().lower().replace('years old', '').replace('months', '').replace('old', '')
            if v.isdigit():
                return int(v)
        return v

class FinalResponse(BaseModel):
    decision: str = Field(..., description="The approval status of the claim. Must be 'approved', 'rejected', or 'further review'.")
    amount: str = Field(..., description="The coverage amount, if applicable. Use 'N/A' if not specified.")
    justification: str = Field(..., description="A detailed explanation for the decision, referencing the specific clause identifiers from the policy documents.")

# --- LLM and Prompts ---
OLLAMA_PARSING_MODEL = "mistral:7b-instruct-v0.2-q4_K_M"
OLLAMA_REASONING_MODEL = "mistral:7b-instruct-v0.2-q4_K_M"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

llm_parser = OllamaLLM(model=OLLAMA_PARSING_MODEL)
llm_reasoning = OllamaLLM(model=OLLAMA_REASONING_MODEL)
final_parser = JsonOutputParser(pydantic_object=FinalResponse)

parsing_prompt_template = PromptTemplate(
    template="You are a data extraction expert. Your goal is to extract structured information from a user's query about an insurance claim. Follow these steps precisely to ensure accuracy:\n\n"
             "1. First, read the User Query and identify all key entities: age, procedure, location, and policy duration.\n"
             "2. For each identified entity, consider its context in the query.\n"
             "3. If policy duration is in years, convert it to months (e.g., '2 years' becomes 24).\n"
             "4. If a field is not present in the query, mark it as null.\n"
             "5. Based on your reasoning, output a single JSON object with the extracted data. Ensure the JSON format is perfect, with no extra text or explanations outside of the JSON block.\n\n"
             "User Query: {query}\n"
             "Format Instructions: {format_instructions}\n"
             "Reasoning and Final JSON:",
    input_variables=["query"],
    partial_variables={"format_instructions": JsonOutputParser(pydantic_object=ParsedQuery).get_format_instructions()},
)

QA_PROMPT_TEMPLATE = PromptTemplate(
    template="""You are a helpful Q&A assistant. Your task is to provide a concise, one- or two-sentence answer to the user's question based strictly on the provided context.
    Do not provide a decision, amount, or justification. Just provide the answer.

    Context:
    {retrieved_context}

    Question: {query}

    Concise Answer:
    """,
    input_variables=["retrieved_context", "query"]
)

# --- Core Functions ---
def parse_query(user_query: str) -> Optional[ParsedQuery]:
    age = None
    age_match = re.search(r'(\d+)\s*[-_]*\s*year[s]?', user_query, re.IGNORECASE)
    if age_match:
        age = int(age_match.group(1))
    policy_duration_months = None
    months_match = re.search(r'(\d+)\s*[-_]*\s*month[s]?', user_query, re.IGNORECASE)
    if months_match:
        policy_duration_months = int(months_match.group(1))
    else:
        years_match = re.search(r'(\d+)\s*[-_]*\s*year[s]?', user_query, re.IGNORECASE)
        if years_match:
            policy_duration_months = int(years_match.group(1)) * 12
    try:
        parsing_chain = parsing_prompt_template | llm_parser | JsonOutputParser(pydantic_object=ParsedQuery)
        llm_extracted_data = parsing_chain.invoke({"query": user_query})
        if age is not None:
            llm_extracted_data['age'] = age
        if policy_duration_months is not None:
            llm_extracted_data['policy_duration_months'] = policy_duration_months
        return ParsedQuery.model_validate(llm_extracted_data)
    except Exception as e:
        print(f"Error during LLM parsing: {e}")
        return None

def load_and_chunk_file(file_path: str) -> List:
    documents = []
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file_extension == '.docx' or file_extension == '.doc':
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
        elif file_extension == '.eml' or file_extension == '.msg':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                documents.append(f.read())
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading document: {str(e)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = text_splitter.split_documents(documents)
    refined_chunks = []
    for i, chunk in enumerate(chunks):
        content = re.sub(r'\n{2,}', '\n', chunk.page_content).strip()
        content = re.sub(r'Page \d+', '', content)
        chunk.page_content = content
        chunk.metadata['chunk_id'] = f"chunk_{i}_{str(hash(content))[:8]}"
        clause_match = re.search(r'(Clause|Section)\s+[\d\.]+', content, re.IGNORECASE)
        if clause_match:
            chunk.metadata['clause_identifier'] = clause_match.group(0)
        else:
            source = chunk.metadata.get('source_file', 'unknown')
            page = chunk.metadata.get('page', 'unknown')
            chunk.metadata['clause_identifier'] = f"{source} - Page {page}"
        chunk.metadata['document_source'] = chunk.metadata.get('source_file', 'unknown')
        chunk.metadata['page_number'] = chunk.metadata.get('page', 'unknown')
        refined_chunks.append(chunk)
    return refined_chunks

def run_main_qa_pipeline(user_query: str, chunks: List) -> str:
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_db = Chroma.from_documents(chunks, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(user_query)
        retrieved_context = "\n\n".join([f"--- Clause from {doc.metadata.get('document_source')} - {doc.metadata.get('clause_identifier')} ---\n{doc.page_content}" for doc in retrieved_docs])
    except Exception as e:
        return f"Error in retrieving policy clauses: {str(e)}"

    try:
        final_prompt = QA_PROMPT_TEMPLATE.format(
            query=user_query,
            retrieved_context=retrieved_context,
        )
        raw_output = llm_reasoning.invoke(final_prompt)
        return raw_output.strip()
    except Exception as e:
        return f"An error occurred during LLM reasoning: {str(e)}"

# --- FastAPI App and Endpoint ---
app = FastAPI(
    title="LLM-Powered Insurance Query System",
    description="A near-production API for processing a single document and query.",
    version="1.0.0",
)

@app.post("/hackrx/run")
async def run_qa_hackrx(
    questions: List[str],
    file: UploadFile = File(...)
):
    TEMP_DIR = "temp_uploads"
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_file_path = os.path.join(TEMP_DIR, file.filename)
    
    all_answers_list = []
    
    try:
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chunks = load_and_chunk_file(temp_file_path)
        
        for q in questions:
            answer_string = run_main_qa_pipeline(q, chunks)
            all_answers_list.append(answer_string)
            
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(TEMP_DIR) and not os.listdir(TEMP_DIR):
             os.rmdir(TEMP_DIR)

    return {
        "status": "success",
        "answers": all_answers_list,
        "document": file.filename
    }

if __name__ == "__main__":
    uvicorn.run("app_with_qa:app", host="0.0.0.0", port=8000, reload=True)