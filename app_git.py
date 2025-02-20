
import os
import pandas as pd
import numpy as np
import faiss
import tiktoken
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS as LCFAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain_groq import ChatGroq
from typing import Any, List, Tuple

# ---------------------------------------------------------------------------
# Configurações iniciais e chaves de API a partir de variáveis de ambiente
# ---------------------------------------------------------------------------
openai_key = os.getenv("OPENAI_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


st.set_page_config(
    page_title="🏦 Análise de Iniciativas Sicredi",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏦"
)

st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
    }
    .header {
        background-color: #009739;  /* Cor verde Sicredi */
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .content {
        padding: 2rem;
    }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<div class="header"><h1>🏦 Análise de Iniciativas Sicredi</h1></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Função para contar tokens (com fallback para o modelo groq)
# ---------------------------------------------------------------------------
def count_tokens(text: str, model_name: str = "llama-3.3-70b-versatile") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception as e:
        if model_name == "llama-3.3-70b-versatile":
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.get_encoding("gpt2")
    return len(encoding.encode(text))

# ---------------------------------------------------------------------------
# Função para converter uma linha do DataFrame em texto
# ---------------------------------------------------------------------------
def row_to_text(row):
    parts = []
    for col, value in row.items():
        parts.append(f"{col}: {value}")
    return "; ".join(parts)

# ---------------------------------------------------------------------------
# Seleciona chunks que caibam na janela de contexto
# ---------------------------------------------------------------------------
def select_chunks_for_context(
    docs_with_scores: List[Tuple[Document, float]], 
    max_context_tokens: int, 
    model_name: str = "llama-3.3-70b-versatile"
) -> Tuple[List[Document], int]:
    docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1])
    selected_docs = []
    total_tokens = 0
    for doc, score in docs_with_scores:
        tokens = count_tokens(doc.page_content, model_name=model_name)
        if total_tokens + tokens <= max_context_tokens:
            selected_docs.append(doc)
            total_tokens += tokens
        else:
            continue
    return selected_docs, total_tokens

# ---------------------------------------------------------------------------
# Classe simples para o docstore
# ---------------------------------------------------------------------------
class SimpleDocstore:
    def __init__(self, docs: dict):
        self.docs = docs
    def search(self, key: str):
        return self.docs.get(key)

# ---------------------------------------------------------------------------
# Retriever customizado que filtra os chunks com base no limite de tokens
# ---------------------------------------------------------------------------
class FilteredRetriever(BaseRetriever):
    vectorstore: Any
    max_context_tokens: int
    model_name: str = "llama-3.3-70b-versatile"
    high_k: int = 1000  # Recupera muitos chunks inicialmente

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.high_k)
        selected_docs, total_tokens = select_chunks_for_context(
            docs_with_scores, self.max_context_tokens, model_name=self.model_name
        )
        # st.write(f"Selecionados {len(selected_docs)} chunks (total tokens: {total_tokens}).")
        return selected_docs

    @property
    def search_kwargs(self) -> dict:
        return {}

# ---------------------------------------------------------------------------
# Função para carregar o chain de QA (incluindo index, vectorstore, retriever e LLM)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_qa_chain() -> RetrievalQA:
    # Caminho do arquivo Excel (certifique-se de que o arquivo existe neste caminho)
    file_path = "Base para Analisar Resultado Obtido Valido.xlsx"
    if not os.path.exists(file_path):
        st.error(f"Arquivo não encontrado: {file_path}")
        st.stop()
    
    # Carrega os dados
    df = pd.read_excel(file_path)
    # st.write(f"Dados carregados. Shape: {df.shape}")
    
    # Cria uma lista de Documentos a partir do DataFrame
    documents = []
    for idx, row in df.iterrows():
        text = row_to_text(row)
        doc = Document(page_content=text, metadata={"row_index": idx})
        documents.append(doc)
    # st.write(f"Total de documentos criados: {len(documents)}")
    
    # Função para gerar embeddings em batch
    def batch_embed_documents(embeddings, documents, batch_size=100):
        all_texts = [doc.page_content for doc in documents]
        all_embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i+batch_size]
            batch_embeddings = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            # st.write(f"Processado batch {i // batch_size + 1} de {((len(all_texts)-1) // batch_size) + 1}")
        return all_embeddings

    # Inicializa o gerador de embeddings e processa os documentos
    embeddings = OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-small")
    embedding_vectors = batch_embed_documents(embeddings, documents, batch_size=100)
    
    # Cria o índice FAISS
    dimension = len(embedding_vectors[0])
    faiss_index = faiss.IndexFlatL2(dimension)
    embedding_array = np.array(embedding_vectors).astype("float32")
    faiss_index.add(embedding_array)
    # st.write("FAISS index construído com {} vetores.".format(faiss_index.ntotal))
    
    # Cria o docstore e o vectorstore
    docstore_dict = {str(i): documents[i] for i in range(len(documents))}
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    docstore = SimpleDocstore(docstore_dict)
    vectorstore = LCFAISS(
        embedding_function=embeddings.embed_query,
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    # st.write("Vectorstore da LangChain criado.")
    
    # Configura o retriever com limite de tokens para o contexto
    max_context_tokens = 60000 - 500
    filtered_retriever = FilteredRetriever(
        vectorstore=vectorstore,
        max_context_tokens=max_context_tokens,
        model_name="llama-3.3-70b-versatile",
        high_k=800
    )
    
    # Define o prompt customizado para a análise de iniciativas do Sicredi
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
### Instruções:
Você é um assistente avançado de análise de dados para gerentes do Sicredi. Você tem acesso a uma base de dados que contém informações sobre iniciativas da instituição, suas métricas e seus resultados obtidos.
Sua principal função é fornecer insights estratégicos, análises detalhadas e identificar padrões ocultos nos dados referentes à eficácia das iniciativas implementadas.

## Suas Funções Principais:
1) Medição de Resultados
- Identifique quais iniciativas tiveram resultados comprovados.
- Analise a porcentagem de iniciativas com resultados efetivos versus aquelas sem resultados.
- Avalie a confiabilidade das medições reportadas.

2) Análises Profundas & Insights Estratégicos
- Responda perguntas abertas trazendo informações qualitativas e métricas quantitativas.
- Gere análises detalhadas sobre os tipos de iniciativas e suas classificações.
- Proponha sugestões de melhoria com base nos dados encontrados.

3) Respostas Contextualizadas & Inteligentes
- Priorize as colunas Chave do item, Métrica e Campo personalizado (Resultado(s) Obtido(s)) ao responder.
- Se a pergunta for sobre um tipo de iniciativa ou uma diretoria específica, direcione a resposta para esse contexto.
- Sempre que possível, inclua números, gráficos e benchmarks comparativos.

### Casos de Uso:
- Perguntas abertas:
    "Considerando as iniciativas apresentadas, quais/quantas tiveram algum resultado comprovado?"
    "Quais tipos de classificações podemos ter com essa base?"
    "Qual a porcentagem de iniciativas com resultados efetivos e sem resultado?"

- Perguntas específicas:
    "Quantas iniciativas atingiram seus objetivos?"
    "Quais iniciativas foram encerradas sem resultado comprovado?"
    "Qual diretoria apresentou o maior número de iniciativas bem-sucedidas?"

### Notas:
Se os dados forem inconclusivos, explique o motivo e sugira outras formas de análise.
SEMPRE forneça insights e recomendações baseadas nos dados.

Dados:
{context}

Pergunta: {question}

Resposta:
"""
    )
    
    # Inicializa o LLM utilizando o ChatGroq
    llm = ChatGroq(
        temperature=0.3,
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",  
        max_tokens=500
    )
    # st.write("LLM inicializado usando ChatGroq.")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=filtered_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt}
    )
    # st.write("RetrievalQA chain criado.")
    return qa_chain

# ---------------------------------------------------------------------------
# Carrega o sistema de QA
# ---------------------------------------------------------------------------
with st.spinner(""):
    qa_chain = load_qa_chain()

# ---------------------------------------------------------------------------
# Interface de consulta do usuário
# ---------------------------------------------------------------------------
st.markdown("<div class='content'>", unsafe_allow_html=True)
query = st.text_input("Digite sua pergunta:", placeholder="Ex: Qual diretoria apresentou o maior número de iniciativas bem-sucedidas?")
if st.button("Enviar"):
    if query:
        with st.spinner("Consultando..."):
            try:
                answer = qa_chain.run(query)
                st.markdown("### Resposta:")
                st.write(answer)
            except Exception as e:
                st.error(f"Ocorreu um erro ao processar a consulta: {e}")
    else:
        st.warning("Por favor, digite uma pergunta.")
st.markdown("</div>", unsafe_allow_html=True)
