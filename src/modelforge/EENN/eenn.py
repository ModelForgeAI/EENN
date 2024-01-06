import pandas as pd
from ._data_preprocessing import _preprocess_data

class EENN:
    def __init__(self, backend='tensorflow'):
        """ Initialize the EENN model. """
        self.backend = backend
        self.model = None

    def fit(self, 
            X_df:pd.DataFrame, 
            y_df:pd.DataFrame, 
            activation:str='linear', 
            emb_layers:list=None, 
            min_batch_size:int=32, 
            max_batch_size:int=1024,
            shape:dict={'size':'auto'}) -> 'EENN':
        """
        Fit the EENN model.
        Args:
            X_df: Input data.
            y_df: Target data.
            activation: Target activation function, either 'linear' or 'sigmoid'.
            emb_layers: List of columns to be embedded.
            min_batch_size: Minimum batch size for training.
            max_batch_size: Maximum batch size for training.
            shape: Size of the model, either 'Auto', 'xs', 'sm', 'md', 'lg', 'xl' or manually set layers.
        Returns:
            self
        """
        # Depending on the backend, import the appropriate class
        if self.backend == 'tensorflow':
            from .tensorflow._eenn_tf import EENN_tf as BackendClass
        elif self.backend == 'pytorch':
            from .pytorch._eenn_pt import EENN_pt as BackendClass
        else:
            raise ValueError("Unsupported backend: choose 'tensorflow' or 'pytorch'")
        
        params = {
            'data':(X_df,y_df),
            'target':{y_df.columns[0]:activation},
            'emb_layers':emb_layers,
            'min_batch_size':min_batch_size,
            'max_batch_size':max_batch_size,
            'shape':shape
        }
        
        ### ADD ABILITY TO PASS IN CUSTOM SPLITS ###
        params = _preprocess_data(params).data_pipeline()
        self.model = BackendClass(params).training_pipeline()
        return self
    
    ### Placeholder for the predict function ###
    def predict(self, X_df:pd.DataFrame) -> pd.DataFrame:
        """
        Predict the target.
        Args:
            X_df: Input data.
        Returns:
            yhat_df: Prediction.
        """
        # Ensure the model has been fitted
        if self.model is None:
            raise Exception("Model has not been fitted. Please call 'fit' first.")

        # Delegate the predict call to the fitted model
        yhat_df = self.model.predict(X_df)
        return yhat_df
    
    ### Placeholder for the evaluate function ###
    def evaluate(self, X_df:pd.DataFrame, y_df:pd.DataFrame) -> dict:
        """
        Evaluate the model.
        Args:
            X_df: Input data.
            y_df: Target data.
        Returns:
            metrics: Dictionary of metrics.
        """
        # Ensure the model has been fitted
        if self.model is None:
            raise Exception("Model has not been fitted. Please call 'fit' first.")

        # Delegate the evaluate call to the fitted model
        metrics = self.model.evaluate(X_df, y_df)
        return metrics
    
    ### Placeholder for the save function ###
    def save(self, filepath:str):
        """
        Save the model.
        Args:
            filepath: Path to save the model.
        Returns:
            None
        """
        # Ensure the model has been fitted
        if self.model is None:
            raise Exception("Model has not been fitted. Please call 'fit' first.")

        # Delegate the save call to the fitted model
        self.model.save(filepath)

    ### Placeholder for the load function ###
    def load(self, filepath:str) -> 'EENN':
        """
        Load the model.
        Args:
            filepath: Path to load the model.
        Returns:
            self
        """
        self.model = self.backend_model.load(filepath)
        return self