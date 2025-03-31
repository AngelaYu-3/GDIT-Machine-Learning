import pandas as pd
import numpy as np
import os
import ast
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"]= "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

df = pd.read_csv('data/best_chunking_strategy.csv')
faiss_index = "index_all"
new_dataframe = True
test_questions = [
    "What does randy need to send a schedule of?",
    "What are some of randy's action items?",
    "What is Philip's proposal focused on, and can you provided details about the proposal?",
    "Can you provide me more detail about the microturbine power generation deal?"
]

"""
Creating Vector Database
"""
def create_vector_database(df, save_path=faiss_index):
    model_kwargs = {'device': 'cpu'}
    encode_kwars = {'normalize_embeddings': True}
    embeddings_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwars,
    )

    all_chunks = []
    all_metadata = []

    for index, row in df.iterrows():
        raw_chunks = row['ner_chunks']
        chunks_array = ast.literal_eval(raw_chunks)

        for i, chunk in enumerate(chunks_array):
            all_chunks.append(chunk)
            all_metadata.append({
                "source_row": index,
                "chunk_index": i
            })

    vector_database = FAISS.from_texts(
        texts=all_chunks,
        embedding=embeddings_model,
        metadatas=all_metadata
    )

    vector_database.save_local(save_path)
    print(f"Vector database created and saved to {save_path}")
    return vector_database, embeddings_model

"""
Loading Vector Database
"""
def load_vector_database(save_path=faiss_index):
    model_kwargs = {'device': 'cpu'}
    encode_kwars = {'normalize_embeddings': True}
    embeddings_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwars,
    )

    if os.path.exists(save_path) and os.path.isdir(save_path):
        vector_database = FAISS.load_local(save_path, embeddings_model, allow_dangerous_deserialization=True)
        return vector_database, embeddings_model
    else:
        raise FileNotFoundError(f"No vector database found at {save_path}")


"""
Creating LLM Model: Llama3 8B
"""
llama_llm = ChatGroq(model_name="llama3-8b-8192")


"""
Creating Prompt Template for LLM
"""
# 
prompt = ChatPromptTemplate.from_template(
        """
            Answer question only provided the context. Give a detailed answer IN minimum 5 sentences!
            SAY I DONT KNOW IF CONTEXT IS NOT ENOUGH. DONT MAKE UP ANSWERS. BUT YOU ARE FREE TO INFER/SUGGEST. THERE IS NO CONFIDENTIAL
            INFORMATION, YOU CAN USE ALL INFORMATION THAT IS INPUTTED.
            {context}

            Here is question:
            {input}
        """
)


"""
Running Queries Through Created RAG LLM + Inspecting cosine similarity score
"""
def query_system(question, new_df):
    database_path = "faiss_index"
    if not os.path.exists(database_path) or not os.path.isdir(database_path) or new_df:
        print("Creating vector database for the first time")
        vector_database, embedding_model = create_vector_database(df, database_path)
    else:
        print("Loading existing vector database")
        vector_database, embedding_model = load_vector_database(database_path)

    # displaying top 15 number of cosine similarity matches (raw retrieval)
    print(f"\nInspecting cosine similarity scores for: \"{question}\"\n")
    query_embedding = embedding_model.embed_query(question)
    results = vector_database.similarity_search_with_score_by_vector(
        embedding=query_embedding,
        k=15
    )

    for i, (doc, score) in enumerate(results):
        print(f"Top-{i+1} Chunk Cosine Score = {score:.4f}")
        print(f"Content: {doc.page_content[:100]}...\n")

    # using MMR-based retrieval to pick 10 chunks for the LLM
    retriever = vector_database.as_retriever(
        search_kwargs={'k': 20, 'search_type': 'mmr', 'lambda_mult': 0.5}
    )
    document_chain = create_stuff_documents_chain(llama_llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("Running MMR-based retrieval and querying the LLM...\n")
    result = retrieval_chain.invoke({"input": question})
    print('========================================')
    print(f"Query: {question}")
    print(f"Answer: {result['answer']}")
    print('========================================\n')
    # recalculating cosine similarity for each retrieved chunks (actual input to LLM)
    print("Retrieved Chunks with Cosine Similarity:\n")
    retrieved_docs = result.get('context', [])
    for i, doc in enumerate(retrieved_docs):
        doc_embedding = embedding_model.embed_documents([doc.page_content])[0]
        cosine_score = float(np.dot(query_embedding, doc_embedding)) 
        print(f"Chunk {i+1} Cosine Score = {cosine_score:.4f}")
        print(f"Content: {doc.page_content[:100]}...\n")

    return result


for question in test_questions:
    result = query_system(question, new_dataframe)

