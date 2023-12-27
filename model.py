import tensorflow as tf
import pandas as pd
import numpy as np

class EENN:
    def __init__(self,params:dict,shape:dict={'size':'auto'}):
        self.params = params
        self.shape = self.tree_size(shape)

    def tree_size(self,shape:dict) -> dict:
        """
        Creates shape of the neural network based on inputs
        Args
            shape: dictionary containing the size (auto,sm,md,lg,xl,<int>) 
                for each layer type including trees, inputs, aux, concat, 
                layers or build (full tree)
        Returns
            shape_vals: dictionary with layer sizes used for building the EENN
        """
        size_dict = {
            'trees':{'sm':8,'md':16,'lg':16,'xl':32},
            'inputs':{'sm':8,'md':16,'lg':32,'xl':32},
            'aux':{'sm':4,'md':4,'lg':8,'xl':8},
            'concat':{'sm':64,'md':128,'lg':256,'xl':512},
            'layers':{'sm':128,'md':256,'lg':512,'xl':1024}}
        
        if shape.get('size', 'auto') == 'auto':
            feature_cnt = len(self.params['data']['train_df'].columns)
            if (feature_cnt >= 224):
                auto_size = 'xl'
            elif feature_cnt >= 112:
                auto_size = 'lg'
            elif feature_cnt >= 48:
                auto_size = 'md'
            else:
                auto_size = 'sm'
        else:
            auto_size = shape['size']

        shape_vals = {layer: shape.get(layer, auto_size) for layer in 
                      ['trees', 'inputs', 'aux', 'concat', 'layers']}
        
        for layer, size in shape_vals.items():
            if isinstance(size, int):
                shape_vals[layer] = size
            else:
                shape_vals[layer] = size_dict[layer][size]

        return shape_vals
    
    def build_dataset(self,df:pd.DataFrame,outputs:list=None,loss:bool=False) -> tf.data.Dataset:
        """
        Converts dataframe to a tensorflow dataset and builds loss function schedule
        Args
            df: input dataframe
            outputs: feature models
            loss: indicator for building loss function schedule
        Returns
            dataset: tensorflow dataset
        """
        y_df = df[outputs].copy()
        y_df.columns = y_df.columns + "_out"
        y_df['model_target'] = df[self.params['target']]
        
        if loss:
            loss_functions = {}
            for col in y_df:
                if ((y_df[col].dtype == int) & (y_df[col].between(0,1,inclusive='both').all())):
                    loss_functions[col] = 'binary_crossentropy'
                else:
                    loss_functions[col] = 'mean_squared_error'
            self.params['loss_functions'] = loss_functions
        dataset = tf.data.Dataset.from_tensor_slices((dict(df.drop(self.params['target'],axis=1)),dict(y_df)))
        return dataset

    def weight_initializer(self, shape, dtype=None):
        """
        assigns pre-trained weights for the first node and randomly initializes the others
        Args
            shape: shape of the layer
            dtype: TensorFlow input
        Returns
            new_weights: initialized outputs
        """
        glorot_uniform = tf.keras.initializers.GlorotUniform()
        new_weights = glorot_uniform(shape, dtype=dtype)
        new_weights = tf.reshape(new_weights[:, 1:], shape=(-1, shape[1] - 1))
        pretrained_weights = self.params['models']['weights'].flatten()
        pretrained_weights = tf.reshape(pretrained_weights, (-1, 1))
        new_weights = tf.concat([pretrained_weights, new_weights], axis=1)
        return new_weights
    
    def bias_initializer(self, shape, dtype=None):
        """
        assigns pre-trained bias for the first node and sets the rest to 0
        Args
            shape: shape of the layer
            dtype: TensorFlow input
        Returns
            new_biases: initialized outputs
        """
        zeros = tf.zeros(shape, dtype=dtype)
        pretrained_bias = self.params['models']['biases']
        if tf.is_tensor(pretrained_bias) and len(pretrained_bias.shape) > 0:
            pretrained_bias = tf.reshape(pretrained_bias, [])
        first_bias = tf.fill([1], pretrained_bias)
        new_biases = tf.concat([first_bias, zeros[1:]], axis=0)
        return new_biases
    
    def batch_scaling(self):
        print("placeholder")
    
    def feature_model(self,X_df:pd.DataFrame,y_df:pd.DataFrame,prune:bool=False) -> tf.keras.models.Sequential:
        """
        Trains a feature model
        Args
            X_df: input variables to the model
            y_df: target variable for the model
            prune: adds dense hidden layer for second pass pruned model
        Returns
            model: trained model
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(len(X_df.columns),)))
        model.add(tf.keras.layers.Dropout(0.25))
        if prune:
            model.add(tf.keras.layers.Dense(self.shape['aux'],activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                            kernel_initializer=self.weight_initializer,
                                            bias_initializer=self.bias_initializer))
            model.add(tf.keras.layers.Dropout(0.125))
        model.add(tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(0.01)))

        if ((y_df.dtype == int) & (y_df.between(0,1,inclusive='both').all())):
            loss = 'binary_crossentropy'
        else:
            loss = 'mean_squared_error'
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss=loss,metrics='accuracy')
        model.fit(X_df,y_df,epochs=3, batch_size=2048)
        return model
    
    def top_weights(self,model:tf.keras.models.Sequential,cols:list,feature_cnt:int):
        """
        Extracts model weights for most important features
        Args
            model: pre-trained TensorFlow model
            cols: list of input column names
            feature_cnt: number of top features to extract
        Returns
            top_features_idx
        """
        weights, biases = model.layers[1].get_weights()
        wgt_features = np.abs(weights[:,0])
        top_features_idx = np.argsort(wgt_features)[-feature_cnt:]
        top_features_nms = [cols[i] for i in top_features_idx]
        top_features_wgt = weights[top_features_idx, :]
        return top_features_nms, top_features_wgt, biases
    
    def build_feature_models(self):
        """
        Trains and prunes feature models
        Args
            None
        Returns
            None
        """
        input_features = [col for col in self.params['data']['train_df'].columns if col not in self.params['emb_layers']]
        input_features.remove(self.params['target'])
        model = self.feature_model(self.params['data']['train_df'][input_features],
                                   self.params['data']['train_df'][self.params['target']])
        
        feature_cnt = self.shape['concat'] - ((self.shape['trees']*self.shape['aux'])+self.shape['trees']) ### Add embeddings
        self.params['passthrough_features'], _, _ = self.top_weights(model,input_features,feature_cnt)
        tree_features, _, _ = self.top_weights(model,input_features,self.shape['trees']-1)

        self.params['models'] = {self.params['target']:{}}
        self.params['models'][self.params['target']]['features'], self.params['models']['weights'], self.params['models']['biases'] = self.top_weights(
            model,input_features,self.shape['inputs'])
        
        self.params['models'][self.params['target']]['model'] = self.feature_model(
            self.params['data']['train_df'][self.params['models'][self.params['target']]['features']],
            self.params['data']['train_df'][self.params['target']],
            prune=True)
        
        for tree in tree_features:
            print('Building Tree:',tree)
            self.params['models'][tree] = {}
            input_tree = [feature for feature in input_features if feature != tree]
            model = self.feature_model(self.params['data']['train_df'][input_tree],self.params['data']['train_df'][tree])
            self.params['models'][tree]['features'], self.params['models']['weights'], self.params['models']['biases'] = self.top_weights(
                model,input_tree,self.shape['inputs'])
            
            self.params['models'][tree]['model'] = self.feature_model(
                self.params['data']['train_df'][self.params['models'][tree]['features']],
                self.params['data']['train_df'][tree],
                prune=True)
        return self.params