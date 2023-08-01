import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

# Page title
st.set_page_config(page_title='🦜🔗 Balance sheet Comparison')
st.title('🦜🔗 Balance sheet Comparison')

# Load CSV file
def load_csv(input_csv):
    df = pd.read_csv(input_csv)
    with st.expander('See DataFrame'):
        st.write(df)
    return df

# Generate LLM response using Langchain
def generate_langchain_response(csv_file, input_query, openai_api_key):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.2, openai_api_key=openai_api_key)
    df = load_csv(csv_file)
    # Create Pandas DataFrame Agent
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
    # Perform Query using the Agent
    response = agent.run(input_query)
    return st.success(response)

# Generate response using Pandas AI
# def generate_pandasai_response(csv_file):
#     df = pd.read_csv(csv_file)
#     llm = OpenAI(api_token="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")
#     pandas_ai = PandasAI(llm)
#     response = pandas_ai.run(df, prompt='Show me 2018 FY, the balance sheet with a bar graph.')
#     st.write(response)
# Generate response using Pandas AI and plot graphs
def generate_pandasai_response(csv_file):
    df = pd.read_csv(csv_file)
    llm = OpenAI(api_token="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")
    pandas_ai = PandasAI(llm)
    response = pandas_ai.run(df, prompt='Show me 2018 FY, the balance sheet with a bar graph.')

    # Parse the Pandas AI response to extract graph data
    graph_data = response['content']
    graph_title = response['title']
    graph_type = response['metadata']['graph_type']

    # Plot the graph using Streamlit
    st.header(graph_title)
    if graph_type == 'bar':
        st.bar_chart(graph_data)
    elif graph_type == 'line':
        st.line_chart(graph_data)
    # Add other graph types as needed
    else:
        st.write("Graph type not supported.")

    # Show the remaining text-based content
    st.write(response['content'])

# Input widgets
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
question_list = [
    'what are the Cash and Cash Equivalents  in 2018 and 2020 FY',
    'Plot the heatmap of all data?',
    'compare 2018 and 2019 fy data',
    'what is total assets for all FY',
    'Other'
]
query_text = st.selectbox('Select an example query:', question_list, disabled=not uploaded_file)
openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))

# App logic
if query_text == 'Other':
    query_text = st.text_input('Enter your query:', placeholder='Enter query here ...', disabled=not uploaded_file)

if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')

if openai_api_key.startswith('sk-') and uploaded_file is not None:
    st.header('Output')
    
    # Decide whether to use Langchain or Pandas AI based on user input
    if query_text.startswith('Show me'):
        generate_pandasai_response(uploaded_file)
    else:
        generate_langchain_response(uploaded_file, query_text, openai_api_key)
