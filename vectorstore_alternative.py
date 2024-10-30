#20241022.cot: developed by v-v1150n & ligi2009, modified by cot
import time
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
import load_csv_to_txt as lc
import argparse

# 設置命令行參數解析
parser = argparse.ArgumentParser(description='Generate vector database by industrial uses for alternatives')
parser.add_argument('file_path', type=str, help='file path for specify input')
parser.add_argument('chunk_size', type=int, help='chunk_size')
parser.add_argument('chunk_overlap', type=int, help='chunk_overlap')
parser.add_argument('iu_id', type=str, help='industrial use id')

args = parser.parse_args()
file_path = args.file_path
chunk_size = args.chunk_size
chunk_overlap = args.chunk_overlap
#chemical_id = args.chemical_id
iu_id = args.iu_id
output_path = f'./vector_db/alternatives_by_industrial_use/{iu_id}'


def load_and_split_documents(file_path, chunk_size, chunk_overlap):
    """加載文檔並進行文本分割"""
    #20241030.cot: using CSVLoader to make a prototype
    loader = CSVLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)
def initialize_embeddings(model_name="all-MiniLM-L6-v2", device="cpu", normalize_embeddings=False):
    """初始化嵌入模型"""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': normalize_embeddings},
        multi_process=True,
        show_progress=True
    )

def save_to_chroma(docs, embeddings, output_path):
    """保存文檔到向量數據庫"""
    try:
        Vector_db = Chroma.from_documents(docs, embeddings, persist_directory=output_path)
        print(f"Vector_db successfully saved to {output_path}")
    except Exception as e:
        print(f"An error occurred while saving data: {e}")

def main():
    #chunk_size = int(input("Enter the chunk size: "))
    #chunk_overlap = int(input("Enter the chunk overlap: "))
    #output_name = input("Enter the output file name(SAS chemical number): ")
    #output_path = f'./Vector_db/{output_name}'

    start_time = time.time()

    # 加載和分割文檔
    docs = load_and_split_documents(file_path, chunk_size, chunk_overlap)

    # 初始化嵌入模型
    hf = initialize_embeddings()

    # 保存為向量數據庫
    save_to_chroma(docs, hf, output_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Script executed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    # file_path = './SAS_txt_file/Benzene.txt'
    # file_path = './Benzene_txt/Benzene_summary.txt' # 59_sum
    # file_path = './Benzene_txt/Benzene_remove_duplicate.txt' # 59_rm_duplicate
    # file_path = './Benzene_txt/Benzene_alternatives_Childrens_Products.txt'
    main()