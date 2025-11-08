import os,pandas as pd,logging,nltk,neattext.functions as nfx
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger('data preprocessing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    let= WordNetLemmatizer()
    text=text.lower()
    text=nfx.remove_emojis(text)
    text=nfx.remove_urls(text)
    text=nfx.remove_punctuations(text)
    text=nfx.remove_special_characters(text)
    text=nfx.remove_stopwords(text)
    text=nltk.word_tokenize(text)
    
    text=[let.lemmatize(i) for i in text]
    return " ".join(text)

def preprocessing_text(df:pd.DataFrame)->pd.DataFrame:
    logger.debug('text preprocessing started')
    le=LabelEncoder()
    df['target']=le.fit_transform(df['target'])
    logger.debug('target column is encoded')
    df=df.drop_duplicates(keep='first')
    logger.debug('duplicates removed')
    df.loc[:,'text']=df['text'].apply(transform_text)
    logger.debug('text colum transformed')
    return df

def main():
    train_data=pd.read_csv('data/raw/train.csv')
    test_data=pd.read_csv('data/raw/test.csv')
    logger.debug('data loaded')
    
    train_preprocessed=preprocessing_text(train_data)
    test_preprocessed=preprocessing_text(test_data)
    
    data_path=os.path.join('data','processed')
    os.makedirs(data_path,exist_ok=True)
    
    train_preprocessed.to_csv(os.path.join(data_path,'new_train.csv'),index=False)
    test_preprocessed.to_csv(os.path.join(data_path,'new_test.csv'),index=False)
    
    logger.debug('data stored at %s',data_path)
    
if __name__=='__main__':
    main()