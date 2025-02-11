from typing import Annotated, Any, Dict, List, TypedDict
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI  # Corrected import
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_API_KEY = "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"

# Define the tools using the @tool decorator
@tool
def get_modelmetadata(databricks_url: str) -> str:
    """Get the model information for the given databricks workspace"""
    return f"The models in {databricks_url} is are ______"

@tool
def get_experiments(databricks_url: str) -> str:
    """Get the experiment information for the given databricks workspace"""
    return f"The experiments in {databricks_url} are ________"

@tool
def default_answer() -> str:
    """Provide a default response when unable to answer."""
    return "I'm sorry, I can't answer that."

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    on_topic: str
    classification: str
    system_message: SystemMessage

get_modelmetadata_tools = [get_modelmetadata]
get_experiments_tools = [get_experiments]
default_answer_tools = [default_answer]

# Instantiate the language models with async support
supervisor_llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
modelmetadata_llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY).bind_tools(get_modelmetadata_tools)
experiments_llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY).bind_tools(get_experiments_tools)
off_topic_llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY).bind_tools(default_answer_tools)

# Asynchronous functions to invoke the models
async def call_metadata_model(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    system_message = SystemMessage(
        content="You are an helpful assistant who Extract Model metadata details from a given Databricks workspace"
    )
    conversation = [system_message] + messages
    print(f"Metadata Model Input: {conversation}")
    response = await modelmetadata_llm.ainvoke(conversation)
    print(f"Metadata Model Output: {response}")
    return {"messages": state["messages"] + [response]}

async def call_experiments_model(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    system_message = SystemMessage(
        content="You are an helpful assistant who Extract Experiment details from a given Databricks workspace"
    )
    conversation = [system_message] + messages
    print(f"Experiments Model Input: {conversation}")
    response = await experiments_llm.ainvoke(conversation)
    print(f"Experiments Model Output: {response}")
    return {"messages": state["messages"] + [response]}

async def call_off_topic_model(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    system_message = SystemMessage(
        content="You are OffTopicBot. Apologize to the user and explain that you cannot help with their request, but do so in a friendly tone."
    )
    conversation = [system_message] + messages
    print(f"Off Topic Model Input: {conversation}")
    response = await off_topic_llm.ainvoke(conversation)
    print(f"Off Topic Model Output: {response}")
    return {"messages": state["messages"] + [response]}

async def call_supervisor_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    system_prompt = """You are a supervisor agent that decides whether the user's message is on topic and classifies it.
            Analyze the user's message and decide:

            Classify it as 'metadata', 'experiments', or 'other' if on topic.

            #### **Classification Criteria:**
            1. **Metadata**  
            - The message relates to retrieving, modifying, or understanding information about datasets, models, runs, or configurations.
            - Keywords: "metadata," "model," "parameters," "run details," "model configurations," "dataset properties."
            - Example:
                - "list of all the available models"
                - "Retrieve metadata for the model ID 3456."
                - "Get the parameters of model run XYZ."

            2. **Experiments**  
            - The message involves listing, retrieving, analyzing, or managing machine learning experiments.
            - Keywords: "experiments," "Databricks experiments," "performance metrics," "training results."
            - Example:
                - "Can you get list of experiments for the URL https://adb-7044699498606730.10.azuredatabricks.net/?"
                - "Fetch accuracy scores of experiment ID 5678."

            3. **Other**  
            - The message does not fit into "metadata" or "experiments."
            - Example:
                - "What is the best cloud provider for deep learning?"
                - "How do I deploy my model?"

            Provide your decision in the following structured format:
                "classification": "metadata" or "experiments" or "other"

    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "User Message:\n\n{user_message}")]
    )

    structured_supervisor_llm = supervisor_llm.with_structured_output(
        SupervisorDecision
    )
    evaluator = prompt | structured_supervisor_llm

    print(f"Supervisor Model Input: {last_message}")
    result = await evaluator.ainvoke({"user_message": last_message})
    print(f"Supervisor Model Output: {result}")

    classification = result.classification
    state["classification"] = classification

    return state

def should_continue_modelmetadata(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "modelmetadata_tools"
    return END

def should_continue_experiments(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "experiments_tools"
    return END

def should_continue_off_topic(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "off_topic_tools"
    return END

class SupervisorDecision(BaseModel):
    """Decision made by the supervisor agent."""
    classification: str = Field(
        description="Classify the message as 'metadata', 'experiments', or 'other'"
    )

modelmetadata_tool_node = ToolNode(tools=[get_modelmetadata])
experiments_tool_node = ToolNode(tools=[get_experiments])
off_topic_tool_node = ToolNode(tools=[default_answer])

def supervisor_router(state: AgentState) -> str:
    classification = state.get("classification", "")
    if classification == "metadata":
        return "metadata_model"
    elif classification == "experiments":
        return "experiments_model"
    else:
        return "off_topic_model"

workflow = StateGraph(AgentState)

workflow.add_node("supervisor_agent", call_supervisor_model)
workflow.add_node("metadata_model", call_metadata_model)
workflow.add_node("modelmetadata_tools", modelmetadata_tool_node)
workflow.add_node("experiments_model", call_experiments_model)
workflow.add_node("experiments_tools", experiments_tool_node)
workflow.add_node("off_topic_model", call_off_topic_model)
workflow.add_node("off_topic_tools", off_topic_tool_node)

workflow.add_edge(START, "supervisor_agent")
workflow.add_conditional_edges(
    "supervisor_agent",
    supervisor_router,
    ["metadata_model", "experiments_model", "off_topic_model"],
)
workflow.add_conditional_edges(
    "metadata_model", should_continue_modelmetadata, ["modelmetadata_tools", END]
)
workflow.add_edge("modelmetadata_tools", "metadata_model")
workflow.add_conditional_edges(
    "experiments_model", should_continue_experiments, ["experiments_tools", END]
)
workflow.add_edge("experiments_tools", "experiments_model")
workflow.add_conditional_edges(
    "off_topic_model", should_continue_off_topic, ["off_topic_tools", END]
)
workflow.add_edge("off_topic_tools", "off_topic_model")


app = workflow.compile()

import json
if __name__ == "__main__":
    from langchain.schema import HumanMessage
    import asyncio

    async def main():
        result = await app.ainvoke(
            {"messages": [HumanMessage(content="Hi How are you?")]}
        )
        print(result)

    asyncio.run(main())

mermaid_code = app.get_graph().draw_mermaid()
with open("workflow.mmd", "w") as f:
    f.write(mermaid_code)
