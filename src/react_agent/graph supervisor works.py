from dotenv import load_dotenv  
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def extract_text_from_pdf(pdf_path: str) -> str:
    """extracts data from pdf."""
    return ""

def extract_text_from_docx(docx_path: str) -> str:
    """extracts data from docs."""
    return ""

# Initialize Azure OpenAI model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
)

def resume_parser(resume_file_path: str):
    """returns a resume of file."""
    return ""

# Create resume parser agent
resume_parser_agent = create_react_agent(
    model,
    tools=[resume_parser],
    name="resume_parser_agent",
    prompt=(
        "You are a resume parser expert. "
        "Always use the one tool resume_parser to parse the resume."
    )
)

def general_question_answer(question: str):
    """answers questions using the llm."""
    response = model.invoke(question)
    return response.content

# Create general Q&A agent
general_question_answer_agent = create_react_agent(
    model,
    tools=[general_question_answer],
    name="general_question_answer_agent",
    prompt=(
        "You are a general question answer expert. "
        "Always use the one tool general_question_answer to answer the question."
    )
)

def google_search(query: str):
    """makes a web search for information."""
    return ""

# Create Google search agent
google_search_agent = create_react_agent(
    model,
    tools=[google_search],
    name="google_search_agent",
    prompt=(
        "You are a Google search expert. "
        "Always use the one tool google_search to search the internet."
    )
)  

# Create supervisor workflow
workflow = create_supervisor(
    [resume_parser_agent, google_search_agent, general_question_answer_agent],
    model=model,
    prompt=(
        "You are a smart team supervisor managing multiple agents. Analyze the user input and delegate to the appropriate agent:\n"
        "- If the input contains a file path or mentions 'resume', use resume_parser_agent.\n"
        "- If the input contains 'search' or asks to find something online, use google_search_agent.\n"
        "- For all other questions or queries, use general_question_answer_agent.\n"
        "Choose the most appropriate agent based on the user's input."
    ),
    output_mode="last_message"
)

# Initialize checkpointer
app = workflow.compile()
config = {"configurable": {"thread_id": "1"}}

# Main interaction loop
while True:
    user_input = input("\nEnter your query (or 'exit' to quit): ")

    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    result = app.invoke({
        "messages": [{
            "role": "user",
            "content": user_input
        }]
    }, config=config)

    for m in result["messages"]:
        print(m.content)