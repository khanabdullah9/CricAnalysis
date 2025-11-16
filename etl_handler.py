from html_reader import HTMLReader
from vector import VectorEmbeddings
import pandas as pd
from utils import is_float
import chromadb
from uuid import uuid4
from tqdm import tqdm

class ETLHandler():
    def __init__(self):
        self.db_location = f"player_stats"
        self.client = chromadb.PersistentClient(path = self.db_location)

    def invoke(self):
        collection = self.get_or_create_chroma_collection()
        try:
            html_reader = HTMLReader()
            for team in ["Men", "Women"]:
                html_reader.team = team
                player_urls, player_names = html_reader.find_player_url()
                self.process_url(player_urls, player_names, collection)
        except Exception as err:
            print(f"[ERR:] {err}")
        finally:
            return collection
        
    def process_url(self, player_urls, player_names, collection):
        """
        Appending player stats in the collection
        """
        for idx, url in enumerate(tqdm(player_urls)):
            name = player_names[idx]
            data = pd.read_html(url)
            
            batting = self.stack_df(data[:3], name)
            bowling = self.stack_df(data[3:], name)

            for role in [batting, bowling]:
                ids, documents, metadatas = self.prepare_data(role)
                collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                )

    def prepare_data(self, df):
        ids = [str(uuid4()) for _ in range(len(df))]
        documents = df["document"].tolist()
        metadatas = df.drop(columns=["document"]).to_dict(orient="records")
        return ids, documents, metadatas

    def stack_df(self, data, player_name):
        """
        Stack all three formats in one dataframe
        returns:
            dataframe
        """
        data_frames = []
        for i in range(3):
            df = self.create_pd_df(data[i], player_name)
            data_frames.append(df)
        stacked = pd.concat(data_frames)
        return stacked

    def create_pd_df(self, data, player_name):
        """
        Structuring the unstructured scraped data
        returns:
            dataframe: Contains bowling/batting (per format) stats of a single player
        """
        text = []; numbers = []
        for _, row in data.iterrows():
            for idx in range(data.shape[1]):
                slugs = row[idx].split(" ")
                for s in slugs:
                    if is_float(s):
                        numbers.append(s)
                    elif not is_float(s) and s:
                        text.append(s)
        
        df = pd.DataFrame([numbers], columns = text[1:])
        df["Format"] = text[0]
        df["Player"] = player_name
        df["document"] = df.apply(lambda row: " ".join([f"{col}: {row[col]}" 
                                                    for col in df.columns]), axis=1)
        return df
    
    def get_or_create_chroma_collection(self):
        collection = self.client.get_or_create_collection(
            name=f"player_stats",
            metadata={"hnsw:space": "cosine"}  # for better embeddings
        )
        
        return collection
    
    
    
    