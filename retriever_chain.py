import asyncio
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
import vectorstore as vs

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])
# 建立RAG Chain 選擇llm model, embedding model, vector database
def chain(llm_model='llama3', load_path=None):
    llm = ChatOllama(model=llm_model, temperature=0.3)
    embeddings = vs.initialize_embeddings()
    # db_load = Chroma(persist_directory=load_path, embedding_function=embeddings)
    # retriever = db_load.as_retriever()
    
    retrievers = []
    for path in load_path:
        db_load = Chroma(persist_directory=path, embedding_function=embeddings)
        retrievers.append(db_load.as_retriever(search_kwargs={"k": 30}))

    template = """
    你是一個專注在回答化學領域相關問題的中文專家。
    你的任務是根據上下文的內容來回答使用者提出的問題。
    請注意：

    1. 你只能回答與化學物質相關的問題，對於非化學物質的問題，請回答「此問題無法回答，請試著詢問其他化學物質相關問題」。
    2. 只回答使用者目前的問題，不要重複之前已經回答過的問題。
    3. 如果不知道答案，請明確回答「此問題無法回答，請試著詢問其他化學物質相關問題」，不要生成無關的答案。
    4. 只能使用繁體中文回答問題。
    5. 如果使用只想要總結過去問過的答案就必須提供給使用者。
    6. 盡可能使用敘述的方式回答問題。

    上下文內容：{context}
    問題：{question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    config = RailsConfig.from_path("./config")
    
    # Function to retrieve documents from multiple retrievers and print them
    def multi_db_retrieve(query):
        all_results = []
        for retriever in retrievers:
            results = retriever.get_relevant_documents(query)
            print(f"Retrieved {len(results)} documents from a retriever:")
            for i, doc in enumerate(results):
                print(f"Document {i+1}: {doc.page_content[:200]}...")  # Print first 200 characters of each document
            all_results.extend(results)
        return all_results
        
    # Format the combined documents from all vector databases
    def format_combined_docs(query):
        docs = multi_db_retrieve(query)  # Retrieve all relevant documents
        formatted_docs = format_docs(docs)  # Format the documents into a single string
        # print(f"Final context returned to LLM:\n{formatted_docs}\n")  # Print the final context
        return formatted_docs  # Return the formatted context to be used by the LLM

    chain = (
        {"context": format_combined_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    guardrails = RunnableRails(config)
    chain_with_guardrails = guardrails | chain
    
    return chain_with_guardrails

if __name__ == "__main__":
    my_chain = chain()
    
