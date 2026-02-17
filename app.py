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
debug_mode = st.sidebar.checkbox("🛠️ Modo Debug", help="Muestra detalles internos de la ejecución")


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

outliner_template = """You are an Editorial Chief.
Given the search results and the user's topic, create a structured outline for a news article.
Focus on:
1. An engaging introduction.
2. Key developments and facts.
3. Analysis of implications.
4. A strong conclusion.
"""

writer_template = """You are a Senior Journalist. Write a high-quality news article based on the provided outline.

Guidelines:
- Tone: Professional, informative, and engaging.
- Structure: Use the outline to create a logical flow. Use Markdown headers (#, ##).
- Content: Incorporate specific details from the context. Avoid generic statements.
- Length: Substantial and comprehensive.
- **Format**: Start directly with the article Title and Body. Do not use conversational filler (e.g., "Here is the article").
"""

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
    # Inicialización con temperatura balanceada para redacción
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.4)
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
                latest_content = None  # Variable segura para rastrear contenido
                
                # Stream de eventos
                for output in app.stream(initial_state):
                    for node_name, value in output.items():
                        messages = value.get("messages", [])
                        if not messages: continue
                        
                        last_msg = messages[-1]
                        # Solo guardamos contenido si es un mensaje de IA válido y tiene texto real
                        # ignorando llamadas a herramientas (JSON) o mensajes vacíos
                        if isinstance(last_msg, BaseMessage) and last_msg.content and not last_msg.tool_calls:
                            latest_content = last_msg.content
                        
                        if node_name == "search":
                            if last_msg.tool_calls:
                                st.write(f"🔎 **Buscando:** {len(last_msg.tool_calls)} fuentes encontradas...")
                                if debug_mode:
                                    st.json(last_msg.tool_calls)
                            else:
                                st.write("✅ **Búsqueda completada.**")
                                
                        elif node_name == "tools":
                             st.write("📥 **Lectura:** Procesando contenido web...")
                             if debug_mode:
                                 st.expander("Resultados Raw").write(last_msg.content)
                             
                        elif node_name == "outliner":
                            st.write("📋 **Esquema:** Estructurando el artículo...")
                            with st.expander("Ver Esquema Propuesto"):
                                st.markdown(last_msg.content)
                                
                        elif node_name == "writer":
                            status.update(label="¡Completado! 🎉", state="complete", expanded=False)
                            final_response = last_msg.content

                # --- FAIL-SAFE: Si no hay respuesta final, usar lo último generado ---
                if not final_response:
                    if latest_content:
                        final_response = latest_content
                        st.warning("⚠️ El redactor final no completó la tarea. Mostrando el esquema/borrador generado.")
                    else:
                        st.error("❌ No se pudo generar contenido legible. Intenta reformular el tema.")
                
                if final_response:
                    st.divider()
                    st.markdown("### 📰 Resultado")
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
