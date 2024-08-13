import streamlit as st
import pandas as pd 
from decouple import config
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


SECRET_KEY = config('SECRET_KEY') # put your api key in .env file



llm = GoogleGenerativeAI(model="gemini-pro",temperatire = 0.6, google_api_key=SECRET_KEY)

st.set_page_config(page_title="Chat with CSV")
st.title('Chat With CSV')
st.write('Upload Your CSV file')
file = st.file_uploader('Select your file' , type=['csv'])

if file is not None :
    df = pd.read_csv(file)
    st.write(df.head(5))
    user_input = st.text_area("What question you want to ask ")

    if user_input is not None :
        button = st.button('Submit')
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True,handle_parsing_errors=True)

        if button:
            with open('prompt.txt', 'r') as file:
                prompt_template = file.read()
            
            # Format the prompt with user input and CSV data
            csv_data = df.to_csv(index=False)
            formatted_prompt = prompt_template.format(
                user_question=user_input,
                )
                
            with st.spinner(text="In progress..."):
                result = agent.invoke(formatted_prompt)
                st.write(result['output'])



