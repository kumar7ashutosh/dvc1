import numpy as np, pandas as pd,logging,os,pickle
from sklearn.ensemble import RandomForestClassifier

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger('model training')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model training.log')
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

def train_model(x_train:np.ndarray,y_train:np.ndarray)->RandomForestClassifier:
    if x_train.shape[0]!=y_train.shape[0]:
        print("x_train & y_train aren't of same size")
    rfc=RandomForestClassifier(n_estimators=22,random_state=2)
    rfc.fit(x_train,y_train)
    logger.debug('model training completed')
    return rfc

def save_model(model,file_path:str)->None:
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    with open(file_path,'wb') as file:
        pickle.dump(model,file)
    logger.debug('model saved at %s',file_path)
    
def main():
    train_data=load_data('data/transformed/train_tfidf.csv')
    x_train=train_data.iloc[:,:-1].values
    y_train=train_data.iloc[:,-1].values
    clf=train_model(x_train,y_train)
    save_model(clf,os.path.join('models','model.pkl'))
    
if __name__=='__main__':
    main()