import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

class data_prep:
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
        Set the data types for the columns in the dataset.
        Args:
            None
        Returns:
            None
        """
        dataset = self.params['data'].copy()
        dataset.fillna(0,inplace=True)
        for col in dataset.columns:
            if col in self.params['emb_layers']:
                if dataset[col].dtype == object:
                    dataset.loc[:,col] = dataset[col].astype('category')
                    dataset.loc[:,col] = dataset.loc[:,col].cat.add_categories('NoVal')
                else:
                    dataset.loc[:,col] = dataset[col].astype('int32')
            else:
                if dataset[col].dtype == int:
                    dataset.loc[:,col] = dataset[col].astype('int32')
                else:
                    dataset.loc[:,col] = dataset[col].astype('float32')
        self.params['data'] = dataset

    def data_split(self):
        """
        Split the dataset into train, validation and test sets.
        Args:
            None
        Returns:
            None
        """
        X_df = self.params['data'].copy()
        y_df = X_df.pop('CatalogPrice')
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
            if col not in self.params['emb_layers']:
                if (X_df[col].max() > 10) | (X_df[col].min() < -10):
                    norm_cols += [col]

        if 'models' not in self.params:
            self.params['models'] = {}
        
        self.params['models']['normalize'] = {'robust':RobustScaler().fit(X_df[norm_cols]), 
                                              'norm_cols':norm_cols}
        
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
        norm_cols = self.params['models']['normalize']['norm_cols']

        dataset.loc[:,norm_cols] = self.params['models']['normalize']['robust'].transform(dataset[norm_cols])
        dataset.loc[:,norm_cols] = self.params['models']['normalize']['minmax'].transform(dataset[norm_cols])

        self.params['data'][split_name]['X_df'] = dataset

    def training_pipeline(self) -> dict:
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

class EENN:
    def __init__(self,params:dict,shape:dict={'size':'auto'}):
        self.params = params
        self.target = params['target']
        self.shape = self.tree_size(shape)

    def tree_size(self,shape:dict) -> dict:
        """
        Creates shape of the neural network based on inputs.
        Args:
            shape: dictionary containing the size (auto,sm,md,lg,xl,<int>) 
                for each layer type including trees, inputs, aux, concat, 
                layers or build (full tree)
        Returns:
            shape_vals: dictionary with layer sizes used for building the EENN
        """
        size_dict = {
            'trees':{'xs':4,'sm':8,'md':16,'lg':16,'xl':32},
            'inputs':{'xs':4,'sm':8,'md':16,'lg':32,'xl':32},
            'aux':{'xs':2,'sm':4,'md':4,'lg':8,'xl':8},
            'concat':{'xs':32,'sm':64,'md':128,'lg':256,'xl':512},
            'layers':{'xs':64,'sm':128,'md':256,'lg':512,'xl':1024}}
        
        if shape.get('size', 'auto') == 'auto':
            feature_cnt = len(self.params['data']['train']['X_df'].columns)
            if (feature_cnt >= 224):
                auto_size = 'xl'
            elif feature_cnt >= 112:
                auto_size = 'lg'
            elif feature_cnt >= 48:
                auto_size = 'md'
            elif feature_cnt >= 16:
                auto_size = 'sm'
            else:
                auto_size = 'xs'
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

    def weight_initializer(self, shape, dtype=None):
        """
        Assigns pre-trained weights for the first node and randomly initializes the others.
        Args:
            shape: shape of the layer
            dtype: TensorFlow input
        Returns:
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
        assigns pre-trained bias for the first node and sets the rest to 0.
        Args:
            shape: shape of the layer
            dtype: TensorFlow input
        Returns:
            new_biases: initialized outputs
        """
        zeros = tf.zeros(shape, dtype=dtype)
        pretrained_bias = self.params['models']['biases']
        if tf.is_tensor(pretrained_bias) and len(pretrained_bias.shape) > 0:
            pretrained_bias = tf.reshape(pretrained_bias, [])
        first_bias = tf.fill([1], pretrained_bias)
        new_biases = tf.concat([first_bias, zeros[1:]], axis=0)
        return new_biases
    
    def LSR_dense(self, units:int, first_node_activation:str='linear') -> tf.keras.layers.Layer:
        """
        Creates a custom dense layer for the first node.
        Args:
            units: number of nodes in the layer
            first_node_activation: activation function for the first node
        Returns:
            CustomDense: custom dense layer
        """
        class CustomDense(tf.keras.layers.Layer):
            def __init__(self, units=32, 
                         weight_initializer='random_normal', 
                         bias_initializer='zeros',
                         first_node_activation='linear', **kwargs):
                super(CustomDense, self).__init__(**kwargs)
                self.units = units
                self.weight_initializer = weight_initializer
                self.bias_initializer = bias_initializer
                self.first_node_activation = first_node_activation

            def build(self, input_shape):
                self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                         initializer=self.weight_initializer,
                                         trainable=True)
                self.b = self.add_weight(shape=(self.units,),
                                         initializer=self.bias_initializer,
                                         trainable=True)

            def call(self, inputs):
                # Apply the appropriate activation function for the first node
                if self.first_node_activation == 'linear':
                    y = tf.matmul(inputs, self.w[:, :1]) + self.b[:1]
                elif self.first_node_activation == 'sigmoid':
                    y = tf.sigmoid(tf.matmul(inputs, self.w[:, :1]) + self.b[:1])
                else:
                    raise ValueError('Invalid activation function')

                # LeakyReLU activation for the other nodes
                y_rest = tf.nn.leaky_relu(tf.matmul(inputs, self.w[:, 1:]) + self.b[1:])
                return tf.concat([y, y_rest], axis=-1)

        return CustomDense(units, self.weight_initializer, self.bias_initializer, first_node_activation)

    def dynamic_training(
            self, model: tf.keras.Model, 
            train_data: Union[Tuple[pd.DataFrame, pd.DataFrame], tf.data.Dataset],
            validation_data: Union[Tuple[pd.DataFrame, pd.DataFrame], tf.data.Dataset],
            min_batch_size:int=32, max_batch_size:int=2048, patience:int=3, lock=False) -> tf.keras.Model:
        """
        Trains a model with dynamic batch size and learning rate.
        Args:
            model: pre-trained TensorFlow model
            X_train: input variables to train the model
            y_train: target variable to train the model
            X_val: input variables to validate the model
            y_val: target variable to validate the model
            min_batch_size: minimum batch size
            max_batch_size: maximum batch size
            patience: numper of epochs without improvement before stopping
        Returns:
            model: trained model
        """

        batch_size = min_batch_size
        best_val_loss = float('inf')
        best_weights = None
        cascade = False
        fails = 0
        success = 0

        if lock:
            for layer in model.layers[:-3]:
                layer.trainable = False

        while fails < patience:
            print("current batch size:",batch_size)
            if isinstance(train_data, tuple):
                history = model.fit(x=train_data[0], y=train_data[1],
                                    epochs=1, batch_size=batch_size,
                                    validation_data=validation_data, 
                                    validation_batch_size=max_batch_size,verbose=1)
            else:
                history = model.fit(train_data.batch(batch_size).shuffle(
                    self.params['data']['train']['X_df'].shape[0]).prefetch(tf.data.experimental.AUTOTUNE),
                    validation_data=validation_data.batch(max_batch_size), 
                    epochs=1, verbose=1)
                
            for layer in model.layers:
                layer.trainable = True
            
            # Get the validation accuracy for this epoch
            current_val_loss = history.history['val_loss'][-1]

            # Save weights if validation accuracy has improved
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_weights = model.get_weights()
                fails = 0
                # If cascade is False, double batch size & learning rate
                if cascade == False:
                    if batch_size < max_batch_size:
                        batch_size *= 2
                        model.optimizer.lr.assign(model.optimizer.lr * 2)
                    elif success == 2:
                        success = 0
                        model.optimizer.lr.assign(model.optimizer.lr * 2)
                    else:
                        success += 1
            else:
                # If validation accuracy has not improved, restore best weights and halve learning rate
                model.set_weights(best_weights)
                model.optimizer.lr.assign(model.optimizer.lr / 2)
                fails += 1
                success = 0
                if fails == 3 and cascade == False:
                    cascade = True
                    print("cascade activated")
                    fails = 0
                # If cascade is True, halve batch size
                if cascade and batch_size > min_batch_size:
                    batch_size = int(batch_size / 2)
                    model.optimizer.lr.assign(model.optimizer.lr / 2)
        return model
    
    def feature_model(self,X_train:pd.DataFrame,y_train:pd.DataFrame,
                      X_val:pd.DataFrame,y_val:pd.DataFrame,
                      prune:bool=False) -> tf.keras.models.Sequential:
        """
        Trains a feature model.
        Args:
            X_train: input variables to train the model
            y_train: target variable to train the model
            X_val: input variables to validate the model
            y_val: target variable to validate the model
            prune: adds dense hidden layer for second pass pruned model
        Returns:
            model: trained model
        """
        if ((y_train.dtype == int) & (y_train.between(0,1,inclusive='both').all())):
            loss = 'binary_crossentropy'
            output = 'sigmoid'
        else:
            loss = 'mean_squared_error'
            output = 'linear'
        
        lock = False
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(len(X_train.columns),)))
        if prune:
            model.add(tf.keras.layers.Dropout(0.125))
            model.add(self.LSR_dense(self.shape['aux'],first_node_activation=output))
            model.add(tf.keras.layers.Dropout(0.125))
            lock = True
        model.add(tf.keras.layers.Dense(1,activation=output,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.01)))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss=loss,metrics='accuracy')
        model = self.dynamic_training(
            model=model, train_data=(X_train, y_train), validation_data=(X_val, y_val),
            min_batch_size=128,max_batch_size=4096,lock=lock)

        return model
    
    def top_weights(self,model:tf.keras.models.Sequential,cols:list,feature_cnt:int):
        """
        Extracts model weights for most important features.
        Args:
            model: pre-trained TensorFlow model
            cols: list of input column names
            feature_cnt: number of top features to extract
        Returns:
            top_features_nms: list of top feature names
            top_features_wgt: list of top feature weights
            biases: bias for the first node
        """
        #weights, biases = model.layers[0].get_weights()
        weights, biases = model.layers[-1].get_weights()
        wgt_features = np.abs(weights[:,0])
        top_features_idx = np.argsort(wgt_features)[-feature_cnt:]
        top_features_nms = [cols[i] for i in top_features_idx]
        top_features_wgt = weights[top_features_idx, :]
        return top_features_nms, top_features_wgt, biases
    
    def build_feature_models(self):
        """
        Trains and prunes feature models.
        Args:
            None
        Returns:
            None
        """
        input_features = [col for col in self.params['data']['train']['X_df'].columns 
                          if col not in self.params['emb_layers'].keys() and col != self.target]
        
        print('building intial model')
        model = self.feature_model(self.params['data']['train']['X_df'][input_features],
                                   self.params['data']['train']['X_df'][self.target],
                                   self.params['data']['val']['X_df'][input_features],
                                   self.params['data']['val']['X_df'][self.target])
        
        feature_cnt = self.shape['concat']
        feature_cnt -= (self.shape['trees']*self.shape['aux'])
        feature_cnt -= self.shape['trees']
        feature_cnt -= sum(int(len(emb)**0.25) for emb in params['emb_layers'].values())

        self.params['passthrough_features'], _, _ = self.top_weights(model,input_features,feature_cnt)
        tree_features, _, _ = self.top_weights(model,input_features,self.shape['trees']-1)
        
        self.params['models']['trees'] = {}
        self.params['models']['trees'][self.target] = {}
        self.params['models']['trees'][self.target]['features'], self.params['models']['weights'], self.params['models']['biases'] = self.top_weights(
            model,input_features,self.shape['inputs'])
        
        print('Pruning initial model')
        self.params['models']['trees'][self.target]['model'] = self.feature_model(
            self.params['data']['train']['X_df'][self.params['models']['trees'][self.target]['features']],
            self.params['data']['train']['X_df'][self.target],
            self.params['data']['val']['X_df'][self.params['models']['trees'][self.target]['features']],
            self.params['data']['val']['X_df'][self.target],
            prune=True)
        
        for tree in tree_features:
            print('Building Tree:',tree)
            self.params['models']['trees'][tree] = {}
            input_tree = [feature for feature in input_features if feature != tree]

            model = self.feature_model(
                self.params['data']['train']['X_df'][input_tree],
                self.params['data']['train']['X_df'][tree],
                self.params['data']['val']['X_df'][input_tree],
                self.params['data']['val']['X_df'][tree])
            
            self.params['models']['trees'][tree]['features'], self.params['models']['weights'], self.params['models']['biases'] = self.top_weights(
                model,input_tree,self.shape['inputs'])
            
            print('Pruning Tree',tree)
            self.params['models']['trees'][tree]['model'] = self.feature_model(
                self.params['data']['train']['X_df'][self.params['models']['trees'][tree]['features']],
                self.params['data']['train']['X_df'][tree],
                self.params['data']['val']['X_df'][self.params['models']['trees'][tree]['features']],
                self.params['data']['val']['X_df'][tree],
                prune=True)
        return self.params
    
    def build_dataset(self,X_df:pd.DataFrame,y_df:pd.DataFrame) -> tf.data.Dataset:
        """
        Converts dataframes to a tensorflow dataset.
        Args:
            X_df: input dataframe
            y_df: target dataframe
        Returns:
            dataset: tensorflow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((dict(X_df.drop(self.target,axis=1)),dict(y_df)))
        return dataset
    
    def model_inputs(self):
        """
        Creates model inputs.
        Args:
            None
        Returns:
            model_inputs: dictionary of model inputs
            feature_outputs: output of the feature layer
        """
        model_inputs = {}

        if len(self.params['emb_layers']) > 0:
            emb_features = []
            for feature, idx in self.params['emb_layers'].items():
                if self.params['data']['train']['X_df'][feature].dtypes.name == 'category':
                    model_inputs[feature] = tf.keras.Input(shape=(1,), name=feature,dtype='string')
                else:
                    model_inputs[feature] = tf.keras.Input(shape=(1,), name=feature,dtype='int32')
                
                catg_col = tf.feature_column.categorical_column_with_vocabulary_list(feature, idx)
                emb_col = tf.feature_column.embedding_column(
                    catg_col,dimension=int(len(idx)**0.25))
                emb_features.append(emb_col)
            
            emb_layer = tf.keras.layers.DenseFeatures(emb_features)
            emb_outputs = emb_layer(model_inputs)
        else:
            emb_outputs = None

        all_features = self.params['passthrough_features']
        for tree in self.params['models']['trees'].values():
            all_features += tree['features']
        all_features = list(set(all_features))

        feature_columns = []
        for feature in all_features:
            model_inputs[feature] = tf.keras.Input(shape=(1,), name=feature)
            feature_columns.append(tf.feature_column.numeric_column(feature))
            
        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        feature_outputs = feature_layer({k:v for k,v in model_inputs.items() 
                                         if k not in self.params['emb_layers'].keys()})

        return model_inputs, emb_outputs, feature_outputs
    
    def base_model(self):
        """
        Builds the final model.
        Args:
            None
        Returns:
            model: final model
        """
        model_inputs, emb_outputs, feature_outputs = self.model_inputs()

        tree_inputs = []
        tree_outputs = []
        for model in self.params['models']['trees'].values():
            concat = tf.keras.layers.Concatenate()([v for k,v in model_inputs.items() if k in model['features']])
            tree_inputs.append(model['model'](concat))
            tree_outputs.append(model['model'].layers[-3](concat))

        combined = tf.keras.layers.Concatenate()(tree_inputs + tree_outputs + [emb_outputs, feature_outputs])
        #combined = tf.keras.layers.Concatenate()(tree_inputs + [emb_outputs, feature_outputs])
        layer = tf.keras.layers.Dropout(0.125)(combined)
        output = tf.keras.layers.Dense(1,name="CatalogPrice")(layer)

        base_model = tf.keras.Model(inputs=model_inputs, outputs=output)
        base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss='mean_squared_error',metrics='accuracy')

        self.params['data']['train']['dataset'] = self.build_dataset(
            self.params['data']['train']['X_df'],self.params['data']['train']['y_df']).cache()
        self.params['data']['val']['dataset'] = self.build_dataset(
            self.params['data']['val']['X_df'],self.params['data']['val']['y_df']).cache()
        base_model = self.dynamic_training(
            model=base_model, train_data=self.params['data']['train']['dataset'], 
            validation_data=self.params['data']['val']['dataset'],
            min_batch_size=128,max_batch_size=4096)
        #self.params['models']['EENN'] = base_model
        #return self.params
        return base_model
    
    def grow_model(self,base_model):
        """
        Builds a new model with an additional Dense layer, and copies the weights from the base model.
        Args:
            base_model: base model
        Returns:
            new_model: new model
        """
        #base_model = self.params['models']['EENN']
        self.params['models']['weights'], self.params['models']['biases'] = base_model.layers[-1].get_weights()

        model_inputs = base_model.inputs
        combined = base_model.layers[-2].output
        layer = self.LSR_dense(self.shape['layers'],first_node_activation='linear')(combined) ### Combine with other method, activation auto ###
        layer = tf.keras.layers.Dropout(0.125)(layer)
        output = tf.keras.layers.Dense(1, name=self.params['target'])(layer)
        new_model = tf.keras.Model(inputs=model_inputs, outputs=output)

        for new_layer, old_layer in zip(new_model.layers[:-2], base_model.layers[:-1]):
            new_layer.set_weights(old_layer.get_weights())

        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss='mean_squared_error', metrics='accuracy')

        ### Need to save this as variable next run:
        #trainset = self.build_dataset(self.params['data']['train']['X_df'],self.params['data']['train']['y_df']).cache()
        #valset = self.build_dataset(self.params['data']['val']['X_df'],self.params['data']['val']['y_df']).cache()
        new_model = self.dynamic_training(
            model=new_model, train_data=self.params['data']['train']['dataset'],
            validation_data=self.params['data']['val']['dataset'],
            min_batch_size=128,max_batch_size=4096,lock=True)
        
        return new_model
    
    def evolve_model(self,base_model):

        #base_model = self.params['models']['EENN']
        #self.params['models']['weights'], self.params['models']['biases'] = base_model.layers[-1].get_weights()

        best_val_loss = float('inf')
        grow = True
        layers = 0

        while grow:
            layers += 1
            print("Current Layers:",str(layers))
            new_model = self.grow_model(base_model)
            loss, _ = new_model.evaluate(self.params['data']['val']['dataset'].batch(4096))
            if loss < best_val_loss:
                best_val_loss = loss
                base_model = new_model
            else:
                grow = False

        return base_model
