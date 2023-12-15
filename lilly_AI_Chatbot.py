from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import streamlit as st
import openai
import streamlit as st
import os
openai.api_key = os.getenv('OPEN_AI_API_KEY')


st.set_page_config(page_title="MedicalChatbot", page_icon="📖")
st.title("Lilly- Your AI Doctor")

msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")


template = """ You are a medical chatbot having conversation with human. You should be empathetic too. you can ask few questions until you understand the condition.
Make sure you ask question in simple term, so that user can understand it. Based on the symptoms, you will know the sepcialist keyword. For example, if the symptoms related to brain, specialist is Neurologist.
If the human ask for doctor information ask about their location and just provide an google search link where https://www.google.com/search?q=specialist+keyword+location .
{history}
Human: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "human_input"], template=template
)
llm = OpenAI(temperature=0.8)
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    response = llm_chain.run(prompt)
    st.chat_message("ai").write(response)

