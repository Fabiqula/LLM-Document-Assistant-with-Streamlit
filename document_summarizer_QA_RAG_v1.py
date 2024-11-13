import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    """
    Load the document based on its file extension and return the content.
    Supported formats: PDF, DOCX, TXT.

    Args:
        file (str): Path to the file to be loaded.

    Returns:
        list: Loaded document content as LangChain Document.
    """
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# Function to chunk loaded data for LLM
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    """
    Chunk the loaded document into smaller pieces for processing by LLM.

    Args:
        data (list): List of LangChain documents to be chunked.
        chunk_size (int): The size of each chunk (default 256).
        chunk_overlap (int): The number of overlapping characters between chunks (default 20).

    Returns:
        list: List of document chunks.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# Function to chunk loaded data for Summarization
def chunk_data_summarization(data, chunk_size=10000, chunk_overlap=50):
    """
    Chunk the loaded document into smaller pieces specifically for summarization.

    Args:
        data (list): List of LangChain documents to be chunked.
        chunk_size (int): The size of each chunk (default 10000).
        chunk_overlap (int): The number of overlapping characters between chunks (default 50).

    Returns:
        list: List of document chunks.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk_s = text_splitter.split_documents(data)
    return chunk_s


def create_embeddings(file_name, chunks, dimensions=1536):
    """
    Create embeddings for the document chunks using Pinecone as a vector store.

    Args:
        file_name (str): The name of the file being processed.
        chunks (list): List of document chunks to create embeddings for.
        dimensions (int): The dimensionality of the embeddings (default 1536).

    Returns:
        vector_store: The vector store containing the embeddings.
    """
    import pinecone
    from pinecone import PodSpec
    from langchain_community.vectorstores import Pinecone
    import os

    file_name = os.path.splitext(file_name)[0].split('/')[1].lower().replace("_", "-").replace(" ", "-")

    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=dimensions)

    st.warning(f'Your file_name: {file_name}.\
    Please make sure that file name consist only of lower case alphanumeric characters or "-" .')

    if file_name in pc.list_indexes().names():
        st.write(f"Index: {file_name} already exists. Loading embeddings...")
        vector_store = Pinecone.from_existing_index(file_name, embeddings)
    else:
        st.write("Deleting any existing indexes before creating a new one...")
        indexes = pc.list_indexes().names()
        for index_name in indexes:
            st.write(f"Deleting index: {index_name}")
            pc.delete_index(index_name)

        pc.create_index(
            name=file_name,
            dimension=dimensions,
            metric='cosine',
            spec=PodSpec(
                environment='gcp-starter'
            )
        )
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=file_name)
    return vector_store


def ask_and_get_answer(vector_store, q, chat_history=None, k=3):
    """
    Ask a question and retrieve an answer based on document embeddings and chat history.

    Args:
        vector_store: The vector store containing document embeddings.
        q (str): The user's question.
        chat_history (list): List of previous Q&A to provide context (default None).
        k (int): The number of relevant document chunks to retrieve (default 3).

    Returns:
        dict: The answer to the question, including any additional relevant information.
    """
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

    general_system_template = f""" 
        You are examining a document. Use only the heading and piece of context to answer the questions at the end.
         If you don't know the answer, use general knowledge but don't try to make up an answer.
          Do not add any observations or comments. Answer only in English.
        ----
        CONTEXT: {{context}}
        ----
        """
    general_user_template = "Here is the next question, remember to only answer if you can from the provided context." \
                            "Only respond in English. QUESTION:```{question}```"

    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs={'prompt': qa_prompt}
    )
    answer = crc.invoke({'question': q, 'chat_history': chat_history})
    return answer


def calculate_embedding_cost(texts):
    """
    Calculate the embedding cost based on the number of tokens in the input text.

    Args:
        texts (list): List of LangChain documents to calculate the embedding cost.

    Returns:
        tuple: The total number of tokens and the estimated cost in USD.
    """
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.00002


def calculate_call_costs(texts, call_cost=None):
    """
    Calculate the total cost of making API calls based on the number of tokens in the input text.

    Args:
        texts (list): List of LangChain documents to calculate the call cost.
        call_cost (float): The current call cost (default None).

    Returns:
        tuple: The total number of tokens and the estimated call cost in USD.
    """
    import tiktoken

    enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    total_tokens = sum([len(enc.encode(chunk.page_content)) for chunk in texts])
    final_summary = total_tokens // len(texts)
    total_tokens += final_summary
    call_cost = (total_tokens / 1000) * 0.0005
    return total_tokens, call_cost


def clear_history():
    """
    Clears the chat history stored in the Streamlit session state.
    """
    if 'history' in st.session_state:
        del st.session_state['history']


def summarize_map_reduce(chunks):
    """
    Summarize the given document chunks using the map-reduce approach.

    Args:
        chunks (list): List of document chunks to summarize.

    Returns:
        dict: The generated summary.
    """
    from langchain.chains.summarize import load_summarize_chain
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

    map_prompt = '''
                Write a short and concise summary of the following:
                Text: `{text}`
                CONCISE SUMMARY:
                '''
    map_prompt_template = PromptTemplate(
        input_variables=['text'],
        template=map_prompt
    )
    combine_prompt = '''
                Write a concise summary of the following text that covers the key points.
                Add a title to the summary.
                Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
                by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.
                Text: `{text}`
                '''

    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=['text'])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=False
    )
    output = summary_chain.invoke({'input_documents': chunks})
    return output


if __name__ == "__main__":

    import os
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(), override=True)

    # Initialize memory_state containers
    if 'output' not in st.session_state:
        st.session_state['output'] = {'output_text': None}
    if "vector_stores" not in st.session_state:
        st.session_state['vector_stores'] = {}
    if 'call_costs' not in st.session_state:
        st.session_state['call_costs'] = 0
    if 'embedding_cost' not in st.session_state:
        st.session_state['embedding_cost'] = 0
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'memory' not in st.session_state:
        st.session_state['memory'] = []

    st.image('img1.jpg')
    st.subheader('Summarize documents and ask LLM about it\'s contents.')

    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        api_key2 = st.text_input('PINECONE_API_KEY', type='password')
        if api_key and api_key2:
            os.environ['OPENAI_API_KEY'] = api_key
            os.environ['PINECONE_API_KEY'] = api_key2
        else:
            st.warning("Make sure to provide keys for both services")

        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size_summarization = st.number_input('Chunk Size Used for Summarization:', min_value=100,
                                                   max_value=15000, value=10000, on_change=clear_history)
        add_data_summarization = st.button('Add Data Used for Summarization', on_click=clear_history)
        chunk_size = st.number_input('Chunk_size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data for LLM Questioning', on_click=clear_history)
        remove_data = st.button('Remove Loaded Files')
        chat_context_length = st.number_input(
            "Chat Memory Length", min_value=1, max_value=30, value=10, on_change=clear_history) or 10

        if add_data_summarization:
            if not uploaded_file:
                st.warning(f'you need to load file first. Use Browse files button to load a file')
            else:
                with st.spinner('Reading, and chunking chunk, chunk, chunk....'):
                    bytes_data_sum = uploaded_file.read()
                    file_name = os.path.join("./", uploaded_file.name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data_sum)

                    data_sum = load_document(file_name)
                    chunks_sum = chunk_data_summarization(data_sum, chunk_size=chunk_size_summarization)

                    st.write(f'Summarization chunk size: {chunk_size_summarization}, Chunks: {len(chunks_sum)}')

                    tokens, call_cost = calculate_call_costs(chunks_sum)
                    st.session_state['call_costs'] = call_cost
                    st.write(f'No of tokens: {tokens}, Embedding cost: {call_cost:.4f}')

                    output = summarize_map_reduce(chunks_sum)
                    st.session_state['output'] = output
                    st.success(f'Summarization Complete')

        if add_data:

            if not uploaded_file:
                st.warning(f'you need to load file first. Use Browse files button to load a file')
            else:
                with st.spinner('Reading, chunking and embedding file...'):
                    bytes_data = uploaded_file.read()
                    file_name = os.path.join('./', uploaded_file.name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)

                    data = load_document(file_name)
                    chunks = chunk_data(data, chunk_size=chunk_size)
                    st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                    tokens, embedding_cost = calculate_embedding_cost(chunks)
                    st.session_state['embedding_cost'] = embedding_cost
                    st.write(f'Embedding cost: {embedding_cost:.4f}')

                    if file_name in st.session_state['vector_stores']:
                        vector_store = st.session_state['vector_stores'][file_name]
                        st.write(f'Embedding already exists. Embedding cost: 0')

                        st.success('File uploaded, chunked, and embedded successfully.')

                    else:
                        vector_store = create_embeddings(file_name, chunks)
                        st.session_state['vector_stores'][file_name] = vector_store
                        st.success('File uploaded, chunked, and embedded successfully.')

        if remove_data:
            file_name = os.path.join('./', uploaded_file.name)
            if file_name in st.session_state['vector_stores']:
                del st.session_state['vector_stores'][file_name]
                st.success(f'Embeddings for file: {file_name} deleted')

            else:
                st.warning(f"No embedding found")

    with st.expander('LLM Summarize: '):
        st.markdown(st.session_state['output']['output_text'])
        call_cost = st.session_state['call_costs']
        st.text(f'Cost of Summing the document: approx {call_cost} USD.')

    file_name = uploaded_file.name if uploaded_file else "Your file"

    with st.form(key="my form", clear_on_submit=True):
        q = st.text_input(f"Ask a question about the content of your file: {file_name}", key="user_question")
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        file_name = os.path.join('./', uploaded_file.name)
        if file_name in st.session_state['vector_stores']:

            vector_store = st.session_state['vector_stores'][file_name]

            if len(st.session_state['memory']) >= chat_context_length:
                st.session_state['memory'] = st.session_state['memory'][1:]

            answer = ask_and_get_answer(vector_store, q,  st.session_state['memory'], k)
            st.session_state['memory'].append((q, answer['answer']))

            with st.expander('LLM Answer: '):
                st.markdown(answer['answer'])

                st.divider()

                value = f"Q: {q} \nA: {answer['answer']}"
                st.session_state['history'] = f"{value} \n {'-' * 100} \n {st.session_state['history']} \n "
                h = st.session_state['history']
                st.text_area(label='Chat History', value=h, height=400)
