import streamlit as st
from streamlit_option_menu import option_menu
from single_test import test_app
import time


def set_bar(t):
    word = st.empty()
    bar = st.progress(0)
    for i in range(100):
        word.text('加载进度...' + str(i + 1) + "%")
        bar.progress(i + 1)
        time.sleep(t)
    bar.empty()
    word.empty()


st.sidebar.header("语句判断")
menu1 = "结果一"
menu2 = "结果二"
menu3 = "结果三"

with st.sidebar:
    menu = option_menu("概率结果", [menu1, menu2, menu3],
                       icons=["list-task"], menu_icon="cast", default_index=0)

st.title("场景意图判断器")
st.markdown(":gray[请在下方输入框中输入搜索内容...]")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("📜请输入搜索语句...📜"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with (st.chat_message("assistant")):
        message_placeholder = st.empty()
        full_response = ""
        
        # assistant_response = bot(prompt)
        set_bar(0.02)
        
        dic = {"结果一": 0, "结果二": 1, "结果三": 2}
        first_row = "您的场景意图判断结果如下"
        second_row = "根据您的输入，场景判断结果为: " + test_app(prompt)[dic[menu]][0]
        third_row = "根据您的输入，意图判断结果为: " + test_app(prompt)[dic[menu]][1]
        fourth_row = "该场景意图判断概率为: " + str(test_app(prompt)[dic[menu]][2])
        fifth_row = "感谢您的使用！"
        assistant_response = first_row + '  \n' + second_row + '  \n' + \
                             third_row + '  \n' + fourth_row + '  \n' + fifth_row
        
        for chunk in assistant_response:
            full_response += chunk
            time.sleep(0.02)
            message_placeholder.markdown(full_response + "|")
        
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
