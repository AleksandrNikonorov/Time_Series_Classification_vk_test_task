from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import joblib

class ModelsPipline:

    def __init__(self, models, param_grids, dataframes_path):
        self.models = models # models
        self.param_grids = param_grids # param grids for models
        self.dataframes_path = dataframes_path # paths on preprocessed datasets
        self.best_model = None # Best model
        self.trained_model_params = None # Best params for best model ._. best of the best of the best, sir ._.
        self.train_control_res = [] # train results. model loss on cv folds 
        self.results = [] # General result. Dataset - best model on this dataset - score


    def load_train_test_data(self, name, test_size = 0.3, train_flag = True):
        ''' Выгружаю датасет по name из self.dataframes_path,
            если такого нет - выгружаю дефолтный (выберу какой) '''
        
        default_dataset = 'intervals_dataframe0.csv'
        print(name)
        print(self.dataframes_path.get(name, default_dataset))
        data = pd.read_csv(self.dataframes_path.get(name, default_dataset), index_col= False)
        
        if not train_flag:
            print(data.columns)
            return data.drop(['target'], axis = 1), data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(data.drop(['target'], axis = 1),
                                                            data['target'],
                                                            test_size= test_size,
                                                            stratify= data['target']
        )
        
        return X_train, X_test, y_train, y_test
    
    def tune_hyperparameters(self, model, param_grid, X_train, y_train, n_splits = 5):
        ''' Подбираю гиперпараметры модели
            Записываю промежуточные результаты
            Возвращаю лучшие гиперпарметры для модели'''
        
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=model, 
                                   param_grid=param_grid, 
                                   cv=stratified_kfold,
                                   scoring='roc_auc', 
                                   n_jobs=-1
                                   )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        self.train_control_res.append({
            'model': model,
            'best_params': best_params,
            'cv_results': grid_search.cv_results_
            })
        return best_params

    def run(self):
        '''Основной метод для запуска всего процесса:
           - Загрузка данных
           - Обучение и оценка моделей
           - Подбор гиперпараметров
           - Запись результатов'''
        for dataset_name in tqdm(self.dataframes_path):
            if dataset_name != 'Test':
                X_train, X_test, y_train, y_test = self.load_train_test_data(dataset_name, train_flag= True)

                for model_name, model_type in tqdm(self.models.items()):
                    model = model_type()

                    best_params = self.tune_hyperparameters(model, self.param_grids[model_name], X_train, y_train, n_splits=1)
                    model.set_params(**best_params)
                    model.fit(X_train, y_train)

                    score = roc_auc_score(y_test, model.predict(X_test))
                    self.results.append({
                        'dataset': dataset_name,
                        'model': (model_name, f'trained_{model_name}_on_{dataset_name}.pkl'),
                        'score': score
                    })

                    # Обновление лучшей модели, если текущая модель лучше
                    if self.best_model is None or score > self.best_model_score:
                        self.best_model = model_type()
                        self.best_model_params = best_params
                        self.best_model_score = score

    def final_model_fit_save(self, name_data): #X_train - полный датасет
        model = self.best_model
        model.set_params(**self.best_model_params)
        X_train, y_train = self.load_train_test_data(name_data, train_flag= False)

        model.fit(X_train, y_train)
        trained_model_name = f'trained_BestModel_catboost_on_{name_data}.pkl'
        joblib.dump(model, trained_model_name)
        return trained_model_name