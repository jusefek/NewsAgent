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
st.set_page_config(page_title="Agente Redactor de Noticias", page_icon="🤖", layout="wide")

st.sidebar.header("⚙️ Configuración")

# Gestión de Claves
google_api_key = st.sidebar.text_input("GOOGLE_API_KEY", type="password", help="Tu clave de Google AI Studio")
tavily_api_key = st.sidebar.text_input("TAVILY_API_KEY", type="password", help="Tu clave de Tavily Search")

# Selector de Modelo (Crítico para compatibilidad)
model_version = st.sidebar.selectbox(
    "Modelo Gemini",
    ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
    index=0,
    help="Si '2.5' falla, prueba con '1.5-flash'"
)

# Validación de Claves
if not google_api_key or not tavily_api_key:
    st.info("👋 ¡Hola! Para usar este agente, introduce tus API Keys en la barra lateral.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

# --- 2. INTERFAZ PRINCIPAL ---
st.title("🤖 Agente Redactor de Noticias")
st.markdown(f"**Modelo activo:** `{model_version}`")
st.markdown("Este agente busca noticias relevantes, crea un esquema y redacta un artículo completo.")

topic = st.text_input("¿Sobre qué quieres escribir hoy?", placeholder="Ej: Futuro de la IA Generativa en la educación")

# --- 3. LÓGICA LANGGRAPH (Cacheada para estabilidad) ---

# Definición del Estado
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Prompts
search_template = """Your job is to search the web for related news that would be relevant to generate the article described by the user.
NOTE: Do not write the article. Just search the web for related news if needed and then forward that news to the outliner node."""

outliner_template = """Your job is to take as input a list of articles from the web along with users instruction on what article they want to write and generate a VERY DETAILED and EXTENSIVE outline for the article. 
Ensure the outline covers multiple angles, background context, and deep analysis.
"""

writer_template = """Your job is to write a COMPREHENSIVE, DETAILED, AND LONG article. 
Format:
TITLE: <title> 
BODY: <body> 

Instructions:
1. Use the provided outline to structure the article.
2. EXPAND on every point in the outline significantly. Do not be brief.
3. Provide context, analysis, and specific details from the search results.
4. The final article should be in-depth and professional.
5. Do not copy the outline directly; write a flowing narrative.
```"""

# Funciones Auxiliares
def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt = prompt.partial(system_message=system_message)
    return prompt | llm.bind_tools(tools) if tools else prompt | llm

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {'messages': [result]}

# Grafo Cacheado
@st.cache_resource(show_spinner=False)
def get_graph(model_name):
    """Compila el grafo una sola vez por modelo."""
    # Inicialización
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    tools = [TavilySearchResults(max_results=5)]
    
    # Agentes
    search_agent = create_agent(llm, tools, search_template)
    outliner_agent = create_agent(llm, [], outliner_template)
    writer_agent = create_agent(llm, [], writer_template)
    
    # Nodos
    search_node = functools.partial(agent_node, agent=search_agent, name="Search Agent")
    outliner_node = functools.partial(agent_node, agent=outliner_agent, name="Outliner Agent")
    writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer Agent")
    tool_node = ToolNode(tools)
    
    # Lógica Condicional
    def should_search(state) -> Literal["tools", "outliner"]:
        last_message = state['messages'][-1]
        return "tools" if last_message.tool_calls else "outliner"

    # Construcción
    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("outliner", outliner_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("search")
    workflow.add_conditional_edges("search", should_search)
    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)
    
    return workflow.compile()

# --- 4. EJECUCIÓN ---
if st.button("🚀 Generar Artículo", type="primary"):
    if not topic:
        st.warning("⚠️ Por favor, escribe un tema antes de generar.")
    else:
        try:
            # Obtener grafo compilado
            app = get_graph(model_version)
            
            with st.status("🧠 Procesando solicitud...", expanded=True) as status:
                initial_state = {"messages": [HumanMessage(content=topic)]}
                final_response = None
                
                # Stream de eventos
                for output in app.stream(initial_state):
                    for node_name, value in output.items():
                        messages = value.get("messages", [])
                        if not messages: continue
                        
                        last_msg = messages[-1]
                        
                        if node_name == "search":
                            if last_msg.tool_calls:
                                st.write(f"� **Buscando:** {len(last_msg.tool_calls)} fuentes encontradas...")
                            else:
                                st.write("✅ **Búsqueda completada.**")
                                
                        elif node_name == "tools":
                             st.write("📥 **Lectura:** Procesando contenido web...")
                             
                        elif node_name == "outliner":
                            st.write("� **Esquema:** Estructurando el artículo...")
                            with st.expander("Ver Esquema Propuesto"):
                                st.markdown(last_msg.content)
                                
                        elif node_name == "writer":
                            status.update(label="¡Completado! 🎉", state="complete", expanded=False)
                            final_response = last_msg.content

                if final_response:
                    st.divider()
                    st.markdown("### 📰 Artículo Generado")
                    st.markdown(final_response)
                    
                    # Opción de descarga
                    st.download_button(
                        label="Descargar Artículo",
                        data=final_response,
                        file_name=f"articulo_{topic.replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                else:
                    st.error("❌ El agente finalizó sin generar contenido.")

        except Exception as e:
            st.error(f"❌ **Error Crítico:** {str(e)}")
            st.warning("""
            **Posibles Soluciones:**
            1. Verifica que tus API Keys sean correctas.
            2. Intenta cambiar el modelo a `gemini-1.5-flash` en la barra lateral (algunas claves no tienen acceso a la v2.5).
            3. Verifica tu conexión a internet.
            """)
