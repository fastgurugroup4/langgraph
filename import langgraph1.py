import langgraph
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START



# from dotenv import load_dotenv
# import os

# load_dotenv()  # Load environment variables from .env file

# llm_research = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
# llm_analysis = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=os.getenv("OPENAI_API_KEY"))


OPENAI_API_KEY=""

llm_research = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=OPENAI_API_KEY)
llm_analysis = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=OPENAI_API_KEY)
llm_registration = ChatOpenAI(model="gpt-4", temperature=0.6, openai_api_key=OPENAI_API_KEY)
llm_metadata = ChatOpenAI(model="gpt-4", temperature=0.6, openai_api_key=OPENAI_API_KEY)


from langchain.tools import tool
@tool
def get_registered_models():
    """Retrieve all registered MLflow models along with metadata."""
    try:
        model_metadata="extracted metadata" 
        return model_metadata

    except Exception as e:
        return {"error": f"Error retrieving models: {str(e)}"}



# Define the research agent
def research_agent(state):
    query = state["query"]
    print(f"[Research Agent] Received query: {query}")
    response = llm_research([HumanMessage(content=f"Research this topic: {query}")])
    print(f"[Research Agent] Response: {response.content}")
    return {"research": response.content}

# Define the analysis agent
def analysis_agent(state):
    research_data = state["research"]
    print(f"[Analysis Agent] Received research data: {research_data}")
    response = llm_analysis([HumanMessage(content=f"Summarize and analyze this information: {research_data}")])
    print(f"[Analysis Agent] Response: {response.content}")
    return {"analysis": response.content}

# Define the model registration agent
def model_registration_agent(state):
    analysis = state["analysis"]
    print(f"[Model Registration Agent] Received analysis: {analysis}")
    response = llm_registration([HumanMessage(content=f"Register the model based on this analysis: {analysis}")])
    print(f"[Model Registration Agent] Response: {response.content}")
    return {"registration": response.content}

# Define the model metadata agent
def model_metadata_agent(state):
    registration = state["registration"]
    print(f"[Model Metadata Agent] Received registration data: {registration}")


    # Define a system message for context
    system_message = SystemMessage(content="You are an AI assistant specialized in getting metadata from Databricks when user is asking for it.")

    # Call the registered tool instead of a direct function call
    models = get_registered_models.invoke(None)

    # Send system and user messages to the LLM
    response = llm_metadata([
        system_message,
        HumanMessage(content=f"Generate metadata for the registered model: {registration}")
    ])



    print(f"[Model Metadata Agent] Response: {response.content}")

    return {"metadata": response.content, "models": models}

# Define the graph
workflow = StateGraph(dict)  # Define the state type
workflow.add_node("research", research_agent)
workflow.add_node("analysis", analysis_agent)
workflow.add_node("registration", model_registration_agent)
workflow.add_node("metadata", model_metadata_agent)

# Define edges (flow between nodes)
workflow.set_entry_point("research")
workflow.add_edge("research", "analysis")
workflow.add_edge("analysis", "registration")
workflow.add_edge("registration", "metadata")
workflow.set_finish_point("metadata")

# Compile the graph
app = workflow.compile()

# Run the multi-agent workflow
query = "get model metadata"
result = app.invoke({"query": query})

print("Final Metadata:", result["metadata"])