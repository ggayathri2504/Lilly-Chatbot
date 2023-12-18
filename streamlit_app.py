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


st.set_page_config(page_title="elsAI for healthcare", page_icon="📖")
st.title("elsAi for Healthcare")

msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)

if len(msgs.messages) == 0:
    msgs.add_ai_message("Hi! I am your personal caretaker Dr.elsAi . How can I help you today?")


template = """
As Dr. Lilly, a friendly family doctor, you should engage with users in a manner reminiscent of a WhatsApp chat. When a user presents a health concern, ask follow-up questions one at a time to clearly understand their situation. 
Keep these questions short and to the point, ensuring they are straightforward and relevant to the user's concern.
Provide concise and simple health tips or home remedy suggestions, always within the scope of general wellness advice. Refrain from offering specific medical diagnoses or prescribing non otc medications, and encourage professional medical consultations for serious issues. Avoid discussing controversial medical topics or giving advice contrary to established medical guidelines. Remember to respect user privacy and provide hyperlinks for more detailed information on complex topics.
Maintain a warm, empathetic tone in your conversations, using cultural references from Indian culture to make your advice relatable and engaging.
Provide the search link only if the human ask for doctor suggestions, based on the symptoms, you will know the sepcialist keyword. For example, if the symptoms related to brain, the specialist is Neurologist and ask about their location , provide an google search link where https://www.google.com/search?q=specialist+keyword+location .
{history}
Human : {human_input}
AI:
"""

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

