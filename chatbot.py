#20241022.cot: developed by v-v1150n & ligi2009, modifed by cot
import os
import subprocess
import streamlit as st
import retriever_chain as rc
import requests
import logging

logging.basicConfig(level=logging.INFO)
# Function to create a new session using the provided ID from the URL
#20241023.cot: SAS web will pass chemical_id to trigger chatbot & retriever_chain
#20241023.cot: chemical_id == SAS_chemical_number
def get_id_from_url():
    #20241026.cot: 
    #query_params = st.experimental_get_query_params()
    #chemical_id = query_params.get("id", None)
    chemical_id = st.query_params.id
    if chemical_id:
        #st.warning(st.query_params.id)
        return chemical_id  # Retrieve the ID from URL
    else:
        st.warning("No Chemical ID provided in URL")
        return None

#20241023.cot: replace this by API call
## 讀取化學物質對應的名稱
#def get_chemical_name(chemical_number, mapping_file='./chemical_mapping.txt'):
#    with open(mapping_file, 'r') as file:
#        for line in file:
#            number, name = line.strip().split(':')
#            if number == chemical_number:
#                return name
#    return "Unknown Chemical"  


# Main function for the Streamlit app
def main():
    chemical_id = get_id_from_url()
    if chemical_id is None:
        return

    # 根據用戶輸入的化學品號碼獲取對應的名稱
    #chemical_name = get_chemical_name(SAS_chemical_number)
    #20241023.cot: get chemical from SAS API
    #20241023.cot: maybe port should be placed in .env?
    chemical_name = get_api_response(f"http://172.16.146.197:13003/api/chemicals/name/{chemical_id}")
    #st.warning(chemical_name)
    # 設置RAG Chain 選用llm model, embedding model
    # chain = rc.chain(load_path=load_path)

    st.title('🧪 SAS GPT')
    st.caption("🦙 A SAS GPT powered by Llama3 & NeMo-Guardrails")
    st.warning('🤖 Chatbot with 🧪  '  + f"{chemical_name}")

    with st.sidebar:
        # 清除聊天歷史按鈕
        st.button('🧹 清除查詢記錄', on_click=lambda: st.session_state.update(messages=[{"role": "assistant", "content": "請輸入化學物質相關問題"}]))
        st.markdown(f"[🔙 回到SAS平台](https://sas.cmdm.tw/chemicals/{chemical_id})")

    # 初始化會話狀態中的消息列表，如果還沒有則創建一個默認的消息
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "請輸入化學物質相關問題"}]
    # 顯示會話狀態中的所有消息
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 接收用戶輸入的消息
    if prompt := st.chat_input("請輸入化學物質相關問題"):

        # 將用戶消息添加到會話狀態中
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        logging.info(f"User entered: [{chemical_name}] {prompt}")
        
        # 構建一個查詢，只包含目前使用者輸入的問題
        # query = prompt  # 只使用最新的使用者輸入作為查詢
        query = f"請用中文回答以下問題: {prompt}"

        with st.spinner("Thinking..."):
            response, error = get_response(query, chemical_id)

            if error:
                st.error(f"Error: {error}")
            else:
                # 將模型生成的回應添加到會話狀態中並顯示
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)

def is_summary_query(query):
    summary_keywords = ["總結", "概述", "摘要", "回顧", "重點", "要點", "整理", "summary", "summarize", "summarization", "conclude"]
    return any(keyword in query for keyword in summary_keywords)

def get_response(query, chemical_id):
    try:
        if is_summary_query(query):
            load_path = [f'./vector_db/chemicals/{chemical_id}/summary']
        else:
            #20241023.cot: We have 1) summary vector db, 2) all hazardous data w/o duplicates and 3) safer aternatives
            load_path = [
                f'./vector_db/chemicals/{chemical_id}/hazard_wo_duplicate',
                f'./vector_db/chemicals/{chemical_id}/summary'
            ]
            #20241023.cot: build the vector db if vector db doesn't exist
            print('check and create vector db')
            check_and_create_vector_db(load_path, chemical_id)
            #20241023.cot: include alternative vector db
            industrial_use_ids = get_api_response(f"http://172.16.146.197:13003/api/chemicals/industrial_use_ids/{chemical_id}")
            alternatives_path = [f"./vector_db/alternatives_by_industrial_use/{use_id}" for use_id in industrial_use_ids]
            check_and_create_vector_db_for_alternatives(alternatives_path)
            load_path.extend(alternatives_path)


        # 設置RAG Chain 選用llm model, embedding model
        chain = rc.chain(load_path=load_path)
        response = chain.invoke(query)
        if isinstance(response, dict):
            response_text = response.get('output', '')
        else:
            response_text = response

        if response_text.strip() == "I'm sorry, I can't respond to that.":
            response_text = "此問題無法回答，請試著詢問其他化學物質相關問題"

        return response_text, None
    except Exception as e:
        return None, str(e)

# 清除聊天歷史功能和按鈕
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "請輸入化學物質相關問題"}]

#20241024.cot: get RAG data source from SAS web
def generate_rag_datasource(datatype, chemical_id, output_name):
    if datatype == "chemicals":
        if output_name == "hazard_wo_duplicate":
            url = f"http://172.16.146.197:13003/chemical/{chemical_id}/report.csv"
            file_path = f"./rag_datasource/{datatype}/{chemical_id}/hazard_wo_duplicate"
            get_api_csv_response(url, chemical_id, file_path)

        elif output_name == "summary":
            #20241024.cot: generate summary with ligi2009's python script
            input_path = f"./rag_datasource/{datatype}/{chemical_id}/hazard_wo_duplicate"
            output_path = f"./rag_datasource/{datatype}/{chemical_id}/summary"
            command = ['python', 'gen_summary.py', input_path, output_path]
            print(f"Executing command: {' '.join(command)}")
            # Use subprocess to execute the command
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("Command executed successfully:", result.stdout)

    elif datatype == "alternatives_by_industrial_use":
        url = f"http://172.16.146.197:13003/alternatives_industrial_use/csv/{output_name}/"
        file_path = f"./rag_datasource/{datatype}/{output_name}"
        get_api_csv_response(url, '', file_path)

    else:
        print("unknown dataype")

def check_and_create_vector_db(paths, chemical_id):
    """
    Check if each path in the given list exists. If a path does not exist,
    executes a Python script to create the vector database.

    :param paths: List of paths to check
    """
    for path in paths:
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            #20241024.cot: for example, vector_db/chemicals/59/summary
            output_name = os.path.basename(path)
            input_rag = f'./rag_datasource/chemicals/{chemical_id}/{output_name}'
            #20241024.cot: create input by calling API
            if not os.path.exists(input_rag):
                generate_rag_datasource("chemicals", chemical_id, output_name)

            try:
                # Construct the command to execute the Python script with the specified arguments
                command = ['python', 'vectorstore.py', input_rag, '1000', '200', chemical_id, output_name]
                print(f"Executing command: {' '.join(command)}")
                # Use subprocess to execute the command
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print("Command executed successfully:", result.stdout)
            except subprocess.CalledProcessError as e:
                print("An error occurred while executing the command:", e.stderr)
        else:
            print(f"Path already exists: {path}")

def check_and_create_vector_db_for_alternatives(paths):
    """
    Check if each path in the given list exists. If a path does not exist,
    executes a Python script to create the vector database.

    :param paths: List of paths to check
    """
    for path in paths:
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            output_name = os.path.basename(path)
            input_rag = f'./rag_datasource/alternatives_by_industrial_use/{output_name}'
            #20241024.cot: create input by calling API
            if not os.path.exists(input_rag):
                generate_rag_datasource("alternatives_by_industrial_use", '', {output_name})

            try:
                # Construct the command to execute the Python script with the specified arguments
                command = ['python', 'vectorstore_alternative.py', path, '1000', '200', output_name]
                print(f"Executing command: {' '.join(command)}")
                # Use subprocess to execute the command
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print("Command executed successfully:", result.stdout)
            except subprocess.CalledProcessError as e:
                print("An error occurred while executing the command:", e.stderr)
        else:
            print(f"Path already exists: {path}")

def get_api_response(url):
    try:
        # Make a GET request
        response = requests.get(url)
        # Get the content type from the headers
        content_type = response.headers.get('Content-Type')

        # Check if the response is JSON
        if 'application/json' in content_type:
            data = response.json()  # Parse JSON
        # Check if the response is plain text
        elif 'text/plain' in content_type:
            data = response.text  # Get plain text response
        else:
            print(f"Unhandled Content-Type: {content_type}")
            print(response.text)
            logging.info(f"Unhandled Content-Type: {content_type}")
            # Handle other types as per your requirements, e.g., XML, HTML, etc.

        #return data, None
        return data

    except requests.exceptions.RequestException as e:
        # Handle any network-related errors
        print(f"An error occurred: {e}")

def get_api_csv_response(url, chemical_id, file_path):
    try:
        # Make a GET request to the API
        response = requests.get(url)

        # Raise an error for bad responses (4xx or 5xx)
        response.raise_for_status()

        # Open the file in write-binary mode and save the content
        with open(file_path, 'wb') as file:
            file.write(response.content)

        print(f"CSV file has been saved to {file_path}")

    except requests.exceptions.RequestException as e:
        # Handle any network-related errors or exceptions
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()