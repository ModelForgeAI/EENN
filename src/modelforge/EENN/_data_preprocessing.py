from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

class _preprocess_data:
    """ class for data preparation """
    def __init__(self,params:dict):
        """
        Initialize data_prep class.
        Args:
            params: Dictionary of parameters.
        """
        self.params = params

    def set_dtypes(self):
        """
        Set the data types and embedding index for the columns in the dataset.
        Args:
            None
        Returns:
            None
        """
        dataset = self.params['data'][0].copy()
        dataset[self.params['data'][1].columns[0]] = self.params['data'][1]

        dataset.fillna(0,inplace=True)
        emb_layers = {}
        for col in dataset.columns:
            if col in self.params['emb_layers']:
                if dataset[col].dtype == object:
                    dataset.loc[:,col] = dataset[col].astype('category')
                    dataset.loc[:,col] = dataset.loc[:,col].cat.add_categories('NoVal')
                else:
                    dataset.loc[:,col] = dataset[col].astype('int32')
                emb_layers[col] = dataset[col].unique()
            else:
                if dataset[col].dtype == int:
                    dataset.loc[:,col] = dataset[col].astype('int32')
                else:
                    dataset.loc[:,col] = dataset[col].astype('float32')
        self.params['data'] = (dataset,dataset[[self.params['data'][1].columns[0]]])
        self.params['emb_layers'] = emb_layers

    def data_split(self):
        """
        Split the dataset into train, validation and test sets.
        Args:
            None
        Returns:
            None
        """
        X_df = self.params['data'][0]
        y_df = self.params['data'][1]
        
        X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2,random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5,random_state=42)

        self.params['data'] = {'train':{'X_df':X_train,'y_df':y_train},
                               'val':{'X_df':X_val,'y_df':y_val},
                               'test':{'X_df':X_test,'y_df':y_test}}
        
    def train_normalize(self):
        """ 
        Train the normalizer on the training set.
        Args:
            None
        Returns:
            None
        """
        norm_cols = []
        X_df = self.params['data']['train']['X_df'].copy()
        for col in X_df.columns:
            if col not in self.params['emb_layers'].keys():
                if (X_df[col].max() > 10) | (X_df[col].min() < -10):
                    norm_cols += [col]

        if 'models' not in self.params:
            self.params['models'] = {}
        
        self.params['models']['normalize'] = {'robust':RobustScaler().fit(X_df[norm_cols]), 
                                              'features':norm_cols}
        
        X_df.loc[:,norm_cols] = self.params['models']['normalize']['robust'].transform(X_df[norm_cols])
        self.params['models']['normalize']['minmax'] = MinMaxScaler(feature_range=(0,1)).fit(X_df[norm_cols])

    def normalize(self,split_name:str):
        """
        Normalize the data.
        Args:
            split_name: Name of the split to normalize.
        Returns:
            None
        """
        dataset = self.params['data'][split_name]['X_df'].copy()
        norm_cols = self.params['models']['normalize']['features']

        dataset.loc[:,norm_cols] = self.params['models']['normalize']['robust'].transform(dataset[norm_cols])
        dataset.loc[:,norm_cols] = self.params['models']['normalize']['minmax'].transform(dataset[norm_cols])

        self.params['data'][split_name]['X_df'] = dataset

    def data_pipeline(self) -> dict:
        """
        Run the data preparation pipeline.
        Args:
            params: Dictionary of parameters.
        Returns:
            params: Dictionary of parameters with transformed datasets.
        """
        self.set_dtypes()
        self.data_split()
        self.train_normalize()
        self.normalize('train')
        self.normalize('val')
        self.normalize('test')

        return self.params