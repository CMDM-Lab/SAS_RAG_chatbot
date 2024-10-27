#20241024.cot: Developed by ligi2009
from langchain.chains.summarize import load_summarize_chain
from langchain_ollama import ChatOllama
from langchain.schema import Document
from langchain.prompts import PromptTemplate 
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import argparse

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

parser = argparse.ArgumentParser(description='Generate summary from hazard data')
parser.add_argument('input_file', type=str, help='input path')
parser.add_argument('output_file', type=str, help='output path')
args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file

# read benzene data
#input_file='Benzene_txt/Benzene_remove_duplicate.txt'
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

docs = [Document(page_content=text)]

# use llama3 to summarize
# llm = ChatOllama(model="llama3", temperature=0.3, max_tokens=4096)

# use gpt4o to summarize
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=4096)

prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
    請詳細用繁體中文總結以下文字：
    {text}
    
    總結內容須包含：
    1. 化學物的基本資訊（化學物名稱、中文名稱、風險等級、PubChem CID、CAS No.）
    2. 化學物的所有危害組別(把有的全部都列出），格式：某化學物的危害可歸類為1. ..., 2. ...
    3. 列舉一些化學物危害名稱
    4. 說明各機構對化學物的危害評估
    5. 最後簡短整理化學物特性作為收尾
    """
)

chain_stuff = load_summarize_chain(llm, chain_type="stuff", prompt=prompt_template)
result_stuff = chain_stuff.invoke(docs)

summary_result = (
    "危害組別共有5個：\n"
    "慢毒性危害 (5項)：致癌性(C)、致突變性(M)、生殖毒性(R)、發育毒性(D)、內分泌活性(E)。\n"
    "急毒性危害 (7項)：可分為單一暴露或持續暴露。系統毒性與神經毒性這兩者危害終點有區分為單一暴露與持續暴露兩類，其他則僅屬於單一類別。#單一暴露：(哺乳類)急性毒性(AT)、全身毒性 - 單次劑量(ST)、神經毒性 - 單次劑量(N)、皮膚刺激(IrS)、眼睛刺激(IrE)。#持續暴露：皮膚致敏 - 重複劑量(SnS*)、呼吸致敏 - 重複劑量(SnR*)、全身毒性 - 重複劑量(ST*)、神經毒性 - 重複劑量(N*)。\n"
    "環境危害 (2項)：急性水生毒性(AA)、慢性水生毒性(CA)。\n"
    "環境流佈 (2項)：持續性(P)、生物累積性(B)。\n"
    "物理危害 (2項)：反應性(Rx)、可燃性(F)。\n\n"
    + result_stuff["output_text"] +
    "\n化學安全替代物：\n"
    "安全替代物參考目前安全替代資料庫與軟體運作邏輯，分為以下三類：\n"
    "1. 團隊透過美國環保署ChemView資料庫整理各化學物工業用途，結合與美國環保署更安全化學物質，依照工業用途建立個別化學品安全替代物清單，清單內即為可替代此化學物之安全替代物。化學品可能具有多個工業用途，故系統採取手風琴式展開安全替代物清單。若工業用途較為廣泛，例如：「商業用途」，則安全替代物清單內可能出現多種較為精確工業用途（例：溶劑）之安全替代物。若無資訊，可參考「依工業用途（完整清單）並進行人工篩選」。\n"
    "2. 依照工業用途區分（完整清單），此清單為整理工業用途較安全化學物，而非可替代此頁面之化學物質，建議可由此頁籤右上方搜尋欄過濾工業用途再行篩選，例如可篩選「抗氧化劑」。資料來源： 美國環保署更安全化學物質成分清單 Safer Chemical Ingredients List。\n"
    "3. 依照結構相似性分類，需具備\n"
    "a. 由結構相似性尋找具有相似官能基之化學物質，相近於具有類似功能之化學物，篩選條件為谷本相關係數 0.8以上。\n"
    "b. 由風險評估判斷為較低風險之化學物質，如風險評估等級為三級、四級、資料不足。\n"
    "實務上是否能替代仍須參照該領域知識，並參考其物理化學特性作為選擇參考，若為資料不足（U），則代表目前對於該化學品危害與風險評估較少，故選擇上需多謹慎，並多留意該化學物之相關毒理研究。\n"
    "4. 依照實際案例探討，資料來源： OECD SAAT\n\n"
    "系統依照子結構，列出風險等級為三級（可用、但可尋求更安全替代物）、四級（安全化學物質）、與資料不足（U、無足夠資料可供判別風險等級）\n\n"
    "風險等級依照GreenScreen規則訂為一至四級，數字越小越不建議使用。\n"
    "風險等級1: 不可用，尋求安全替代物。\n"
    "風險等級2: 可用，但尋求安全替代物。\n"
    "風險等級3: 可用，但建議尋求更安全替代物。\n"
    "風險等級4: 安全可使用。\n"
    "風險等級U: 資料不足無法判別。\n\n"
    "依據美國環保署ChemView工業用途分類，依工業用途可替換之安全替代物可分類為消費者或商業用途(consumer or commercial)、商業用途(commercial)、工業用途(industrial)、液壓液(hydraulic fluid)、聚合物(polymers)、兒童產品(children product)、消費者(consumer)，共七類。想了解每個分類有哪些化學物以及其他細項資訊可查詢網站內文件。\n\n"
    "想了解更多細節資訊可以查看網站內的統整圖表及各項資料。\n"
)

# genetate summaruy txt
#output_file='./Benzene_txt/Benzene_summary_gpt.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(summary_result)