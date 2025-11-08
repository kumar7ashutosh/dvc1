
import os,pandas as pd,logging,yaml
from sklearn.model_selection import train_test_split

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger('data ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)->dict:
    with open(params_path,'r') as file:
        params=yaml.safe_load(file)
    return params

def load_data(data_url:str)->pd.DataFrame:
    df=pd.read_csv(data_url,encoding='latin')
    logger.debug('data loaded from %s',data_url)
    return df

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    df=df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
    df.rename(columns={'v1':'target','v2':'text'},inplace=True)
    logger.debug('data preprocessing completed')
    return df

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    raw_data_path=os.path.join(data_path,'raw')
    os.makedirs(raw_data_path,exist_ok=True)
    train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
    test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)

def main():
    params=load_params('params.yaml')
    test_size=params['data_ingestion']['test_size']
    data_path="https://raw.githubusercontent.com/kumar7ashutosh/datasets/main/spam.csv"
    df=load_data(data_path)
    final_df=preprocess_data(df)
    train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=2)
    save_data(train_data,test_data,'data')
    
if __name__=='__main__':
    main()