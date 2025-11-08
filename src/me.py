import numpy as np, pandas as pd,logging,os,pickle,json,yaml
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score
from dvclive import Live
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger('model evaluation')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model evaluation.log')
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

def load_model(file_path:str):
    with open (file_path,'rb') as file:
        model=pickle.load(file)
    logger.debug('model loaded from %s',file_path)
    return model

def load_data(file_path:str)->pd.DataFrame:
    df=pd.read_csv(file_path)
    logger.debug('data loaded from %s',file_path)
    return df

def evaluate_model(clf,x_test:np.ndarray,y_test:np.ndarray)->dict:
    y_pred=clf.predict(x_test)
    y_pred_proba=clf.predict_proba(x_test)[:,1]
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    auc=roc_auc_score(y_test,y_pred_proba)
    
    metrics_dict={
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'auc':auc
    }
    return metrics_dict

def save_metrics(metrics:dict,file_path:str)->None:
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    with open(file_path,'w') as file:
        json.dump(metrics,file,indent=4)
    
def main():
    params=load_params('params.yaml')
    clf=load_model('models/model.pkl')
    test_data=load_data('data/transformed/test_tfidf.csv')
    x_test=test_data.iloc[:,:-1].values
    y_test=test_data.iloc[:,-1].values
    metrics=evaluate_model(clf,x_test,y_test)
    y_pred=clf.predict(x_test)
    with Live(save_dvc_exp=True) as live:
        live.log_metric('accuracy',accuracy_score(y_test,y_pred))
        live.log_metric('precision',precision_score(y_test,y_pred))
        live.log_metric('recall',recall_score(y_test,y_pred))
        
        live.log_params(params)
    save_metrics(metrics,'report/metrics.json')
    
if __name__=='__main__':
    main()