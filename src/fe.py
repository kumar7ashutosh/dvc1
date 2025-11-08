import numpy as np, pandas as pd,logging,os
from sklearn.feature_extraction.text import TfidfVectorizer

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger('feature engineering')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'feature_engineering.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_path:str)->pd.DataFrame:
    df=pd.read_csv(data_path)
    df.fillna('',inplace=True)
    logger.debug('data added from %s',data_path)
    return df

def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame)->tuple:
    vectorizer=TfidfVectorizer(max_features=500)
    x_train=train_data['text'].values
    y_train=train_data['target'].values
    x_test=test_data['text'].values
    y_test=test_data['target'].values
    
    x_train_bow=vectorizer.fit_transform(x_train)
    x_test_bow=vectorizer.transform(x_test)
    
    train_df=pd.DataFrame(x_train_bow.toarray())
    train_df['label']=y_train
    
    test_df=pd.DataFrame(x_test_bow.toarray())
    test_df['label']=y_test
    
    logger.debug('tfidf applied and data is transformed')
    return train_df,test_df

def save_data(df:pd.DataFrame,data_path:str)->None:
    os.makedirs(os.path.dirname(data_path),exist_ok=True)
    df.to_csv(data_path,index=False)
    logger.debug('transformed data saved at %s', data_path)

def main():
    train_df=load_data(r'data\processed\new_train.csv')
    test_df=load_data(r'data\processed\new_test.csv')
    train_df,test_df=apply_tfidf(train_df,test_df)
    save_data(train_df,os.path.join('data','transformed','train_tfidf.csv'))
    save_data(test_df,os.path.join('data','transformed','test_tfdif.csv'))
    
if __name__=='__main__':
    main()