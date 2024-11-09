from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxLLM


from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# origins = [
#     "http://185.111.159.81:9191",
#     "http://62.146.238.53:9191",
#     "http://35.247.190.31:*"
#     "allam.tryanaditya.com",
# ]
origins=["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_path = "document/arabic.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=embeddings_model
)
retriever = vectorstore.as_retriever(search_kwargs={'k':2})

params = {
    "decoding_method": "sample",
    "max_new_tokens": 100,
    "min_new_tokens": 1,
    "temperature": 0,
    "top_k": 50,
    "top_p": 1,
}
llm =WatsonxLLM(
            model_id="sdaia/allam-1-13b-instruct",
            url="https://eu-de.ml.cloud.ibm.com",
            project_id="4162cbfc-92d2-4fe9-b3e7-c26dac539d6c",
            params = params
        )

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "if the question is in arabic, you will also give an answer in arabic."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input} \n AI:"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

class Message(BaseModel):
    role : str
    content : str

class ChatRequest(BaseModel):
    message : str
    history : list[Message]

class ChatResponse(BaseModel):
    content : str
    context : list[str]

@app.post("/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    chat_history = []
    for msg in req.history:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))
    response = rag_chain.invoke({"input": req.message, "chat_history": chat_history})
    context = [r.page_content for r in response['context']]
    # print(context)
    return ChatResponse(content=response["answer"].split("Human:")[0], context=context)

