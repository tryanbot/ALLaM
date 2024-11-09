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

file_path = "document/154.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/sentence-t5-base")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=embeddings_model
)
retriever = vectorstore.as_retriever(search_kwargs={'k':2})

llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-70B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            huggingfacehub_api_token=os.environ['HF_TOKEN']
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
    return ChatResponse(content=response["answer"], context=context)

