import streamlit as st
import os
import functools
from typing import TypedDict, Annotated, List, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- 1. CONFIGURACIÓN DE PÁGINA Y SIDEBAR ---
st.set_page_config(page_title="Agente Redactor de Noticias", page_icon="🤖")

st.sidebar.header("Configuración de Seguridad")
google_api_key = st.sidebar.text_input("GOOGLE_API_KEY", type="password")
tavily_api_key = st.sidebar.text_input("TAVILY_API_KEY", type="password")

if not google_api_key or not tavily_api_key:
    st.sidebar.warning("⚠️ Introduce las API Keys para continuar.")
    st.warning("⚠️ Por favor, introduce ambas API Keys en la barra lateral para continuar.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

# --- 2. INTERFAZ PRINCIPAL ---
st.title("🤖 Agente Redactor de Noticias con LangGraph")
st.markdown("Este agente busca noticias relevantes, crea un esquema y redacta un artículo completo sobre el tema que elijas.")

topic = st.text_input("Tema de la noticia:", placeholder="Ej: Tendencias de IA en 2026")

# --- 3. LÓGICA LANGGRAPH (Fiel al Notebook Original) ---

# Definición del Estado
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Herramientas
tools = [TavilySearchResults(max_results=5)]

# Funciones Auxiliares
def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    if tools:
        return prompt | llm.bind_tools(tools)
    else:
        return prompt | llm

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        'messages': [result]
    }

# Prompts
search_template = """Your job is to search the web for related news that would be relevant to generate the article described by the user.

                  NOTE: Do not write the article. Just search the web for related news if needed and then forward that news to the outliner node.
                  """

outliner_template = """Your job is to take as input a list of articles from the web along with users instruction on what article they want to write and generate an outline
                       for the article.
                    """

writer_template = """Your job is to write an article, do it in this format:

                        TITLE: <title>
                        BODY: <body>

                      NOTE: Do not copy the outline. You need to write the article with the info provided by the outline.

                       ```
                    """

# Inicialización del Modelo (gemini-2.5-flash estrictamente)
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Creación de Agentes
search_agent = create_agent(llm, tools, search_template)
outliner_agent = create_agent(llm, [], outliner_template)
writer_agent = create_agent(llm, [], writer_template)

# Definición de Nodos
search_node = functools.partial(agent_node, agent=search_agent, name="Search Agent")
outliner_node = functools.partial(agent_node, agent=outliner_agent, name="Outliner Agent")
writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer Agent")
tool_node = ToolNode(tools)

# Lógica Condicional
def should_search(state) -> Literal["tools", "outliner"]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (send state to outliner)
    return "outliner"

# Construcción del Grafo
workflow = StateGraph(AgentState)

workflow.add_node("search", search_node)
workflow.add_node("tools", tool_node)
workflow.add_node("outliner", outliner_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("search")

workflow.add_conditional_edges(
    "search",
    should_search,
    {
        "tools": "tools",
        "outliner": "outliner"
    }
)

workflow.add_edge("tools", "search")
workflow.add_edge("outliner", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

# --- 4. EJECUCIÓN STREAMLIT ---
if st.button("Generar Artículo"):
    if not topic:
        st.error("Por favor, introduce un tema.")
    else:
        with st.status("Procesando noticia...", expanded=True) as status:
            try:
                initial_state = {"messages": [HumanMessage(content=topic)]}
                final_response = None
                
                # Ejecutar grafo
                for output in app.stream(initial_state):
                    for node_name, value in output.items():
                        messages = value.get("messages", [])
                        if not messages:
                            continue
                        
                        last_msg = messages[-1]
                        
                        if node_name == "search":
                            if last_msg.tool_calls:
                                st.write(f"🔍 **Search Agent:** Buscando información en la web ({len(last_msg.tool_calls)} queries)...")
                            else:
                                st.write("✅ **Search Agent:** Búsqueda completada.")
                                
                        elif node_name == "tools":
                             st.write("📥 **Tools:** Resultados de búsqueda obtenidos.")
                             
                        elif node_name == "outliner":
                            st.write("📝 **Outliner Agent:** Generando esquema del artículo...")
                            with st.expander("Ver Esquema Generado"):
                                st.markdown(last_msg.content)
                                
                        elif node_name == "writer":
                            status.update(label="¡Redacción Completada!", state="complete", expanded=False)
                            final_response = last_msg.content

                if final_response:
                    st.markdown("### Artículo Generado")
                    st.markdown(final_response)
                else:
                    st.error("No se pudo generar el artículo final.")

            except Exception as e:
                st.error(f"Error durante la ejecución: {e}")
                st.info("Asegúrate de que tus API Keys sean válidas y tengas acceso al modelo 'gemini-2.5-flash'.")
