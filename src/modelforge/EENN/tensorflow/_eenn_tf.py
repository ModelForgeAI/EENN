import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from typing import Tuple, Union

class EENN_tf:
    """ Class for training an EENN model with TensorFlow. """
    def __init__(self,params:dict):
        """ Initialize EENN class. """
        self.params = params
        self.shape = self.tree_size(params['shape'])
        self.target = list(params['target'].keys())[0]

        if self.params['target'][self.target] == 'linear':
            self.activation = 'linear'
            self.loss = 'mean_squared_error'
        else:
            self.activation = 'sigmoid'
            self.loss = 'binary_crossentropy'

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
        
        # Set default size based on number of features
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
            print("auto size:",auto_size)
        else:
            auto_size = shape['size']
        
        # Set shape values for each layer
        shape_vals = {layer: shape.get(layer, auto_size) for layer in 
                      ['trees', 'inputs', 'aux', 'concat', 'layers']}
        
        # Set shape values for each layer
        for layer, size in shape_vals.items():
            if isinstance(size, int):
                shape_vals[layer] = size
            else:
                shape_vals[layer] = size_dict[layer][size]

        return shape_vals
    
    def dynamic_training(
            self, model: tf.keras.Model, 
            train_data: Union[Tuple[pd.DataFrame, pd.DataFrame], tf.data.Dataset],
            validation_data: Union[Tuple[pd.DataFrame, pd.DataFrame], tf.data.Dataset],
            patience:int=3, lock=False) -> tf.keras.Model:
        """
        Trains a model with dynamic batch size and learning rate.
        Args:
            model: pre-trained TensorFlow model
            train_data: input variables to train the model
            validation_data: input variables to validate the model
            patience: numper of epochs without improvement before stopping
            lock: boolean to determine if layers should be locked
        Returns:
            model: trained model
        """

        # Lock layers for first training round if pruning
        if lock:
            for layer in model.layers[:-3]:
                layer.trainable = False

        # Train model until patience is reached
        fails = 0
        success = 0
        cascade = False
        sprint = False
        best_val_loss = float('inf')
        batch_size = self.params['min_batch_size']
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience,restore_best_weights=True)
        
        while fails < patience:
            if sprint:
                sprint = False
                cascade = True
                epochs = 1000
            else:
                epochs = 1

            if isinstance(train_data, tuple):
                history = model.fit(x=train_data[0], y=train_data[1],
                                    epochs=epochs, batch_size=batch_size,
                                    validation_data=validation_data, 
                                    validation_batch_size=self.params['max_batch_size'],
                                    verbose=1, callbacks=[early_stopping])
                
            # Train model for one epoch (tf.data.Dataset)
            else:
                history = model.fit(train_data.batch(batch_size).shuffle(
                    self.params['data']['train']['X_df'].shape[0]).prefetch(tf.data.experimental.AUTOTUNE),
                    validation_data=validation_data.batch(self.params['max_batch_size']), 
                    epochs=epochs, verbose=1, callbacks=[early_stopping])
                
            # Unlock layers after first training round
            if cascade == False and batch_size == self.params['min_batch_size']:
                print("unlocking layers")
                for layer in model.layers:
                    layer.trainable = True

            # Get the validation accuracy for this epoch
            current_val_loss = history.history['val_loss'][-1]

            # Save weights if validation accuracy has improved
            if current_val_loss < best_val_loss:
                fails = 0
                success += 1
                best_val_loss = current_val_loss
                best_weights = model.get_weights()
                if cascade == False:
                    lr_adj = 2
                    batch_size *= 2
                else:
                    lr_adj = 1
                print("success:",success)

            else:
                model.set_weights(best_weights)
                fails += 1
                success = 0
                lr_adj = 0.5
                if fails == 3 and cascade==False:
                    fails = 0
                    cascade = True
                    sprint = True
                if cascade:
                    batch_size = int(batch_size / 2)

            if model.optimizer.lr * lr_adj > 0.25:
                model.optimizer.lr.assign(0.25)
            else:
                model.optimizer.lr.assign(model.optimizer.lr * lr_adj)

            if batch_size > self.params['max_batch_size']:
                batch_size = self.params['max_batch_size']
            elif batch_size < self.params['min_batch_size']:
                batch_size = self.params['min_batch_size']
            
            print("batch size:",batch_size)
            if batch_size >= self.params['max_batch_size'] and cascade == False and success >= 3:
                sprint = True
                success = 0

        return model
    
    def tree_model(
            self, X_train: pd.DataFrame, y_train: pd.DataFrame,
            X_val: pd.DataFrame, y_val: pd.DataFrame, grow:bool=False,
            activation:str='linear',
            loss:str='mean_squared_error') -> tf.keras.models.Sequential:
        """
        Builds a regression model.
        Args:
            train_data: input variables to train the model
            validation_data: input variables to validate the model
            grow: boolean to determine if model should be grown
            activation: activation function for the output layer
            loss: loss function for compiling the model
        Returns:
            model: trained regression model
        """
        # Build model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(len(X_train.columns),)))
    
        # add dropout and dense layers if second iteration
        if grow:
            model.add(tf.keras.layers.Dropout(0.125))
            model.add(tf.keras.layers.Dense(
                self.shape['aux'],activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
            model.add(tf.keras.layers.Dropout(0.125))

        model.add(tf.keras.layers.Dense(1,activation=activation,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.01)))

        # Compile and train model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss=loss,metrics='accuracy')
        model = self.dynamic_training(
            model=model, train_data=(X_train,y_train), validation_data=(X_val,y_val))
        
        return model
    
    def top_weights(self,model:tf.keras.models.Sequential,cols:list,feature_cnt:int) -> list:
        """
        Determines most important features based on model weights.
        Args:
            model: pre-trained TensorFlow model
            cols: list of input column names
            feature_cnt: number of top features to extract
        Returns:
            top_features_nms: list of top feature names
        """
        # Extract weights and biases
        weights, _ = model.layers[-1].get_weights()

        # Extract top features and their weights
        wgt_features = np.abs(weights[:,0])
        top_features_idx = np.argsort(wgt_features)[-feature_cnt:]
        top_features_nms = [cols[i] for i in top_features_idx]
        return top_features_nms
    
    def build_feature_models(self):
        """
        Trains and prunes feature models.
        Args:
            None
        Returns:
            None
        """
        # Create list of model input features
        input_features = [col for col in self.params['data']['train']['X_df'].columns 
                          if col not in self.params['emb_layers'].keys() and col != self.target]
        
        print('Running initial regression')
        model = self.tree_model(
            X_train=self.params['data']['train']['X_df'][input_features],
            y_train = self.params['data']['train']['X_df'][self.target],
            X_val = self.params['data']['val']['X_df'][input_features],
            y_val = self.params['data']['val']['X_df'][self.target],
            activation=self.activation,loss=self.loss)
        
        # Calculate count of additional passthrough features
        feature_cnt = self.shape['concat']
        feature_cnt -= (self.shape['trees']*self.shape['aux'])
        feature_cnt -= self.shape['trees']
        feature_cnt -= sum(int(len(emb)**0.25) for emb in self.params['emb_layers'].values())

        # Extract top features from initial model and assign passthroughs
        self.params['passthrough_features'] = self.top_weights(model,input_features,feature_cnt)

        # define trees based on top features
        tree_features = self.top_weights(model,input_features,self.shape['trees']-1)
        
        # Prune initial model and add layer
        self.params['models']['trees'] = {}
        self.params['models']['trees'][self.target] = {}
        self.params['models']['trees'][self.target]['features'] = self.top_weights(
            model,input_features,self.shape['inputs'])
        
        print('Growing initial output tree')
        self.params['models']['trees'][self.target]['model'] = self.tree_model(
            X_train=self.params['data']['train']['X_df'][self.params['models']['trees'][self.target]['features']],
            y_train=self.params['data']['train']['X_df'][self.target],
            X_val=self.params['data']['val']['X_df'][self.params['models']['trees'][self.target]['features']],
            y_val=self.params['data']['val']['X_df'][self.target],
            grow=True, activation=self.activation,loss=self.loss)
        
        # Build and prune feature models for each tree
        for tree in tree_features:
            print('Running tree regression:',tree)
            self.params['models']['trees'][tree] = {}
            input_tree = [feature for feature in input_features if feature != tree]

            # Determine loss function based on feature characteristics
            if ((self.params['data']['train']['X_df'][tree].dtype == int) &
                (self.params['data']['train']['X_df'][tree].between(0,1,inclusive='both').all())):
                activation = 'sigmoid'
                loss = 'binary_crossentropy'
            else:
                activation = 'linear'
                loss = 'mean_squared_error'

            model = self.tree_model(
                X_train=self.params['data']['train']['X_df'][input_tree],
                y_train=self.params['data']['train']['X_df'][tree],
                X_val=self.params['data']['val']['X_df'][input_tree],
                y_val=self.params['data']['val']['X_df'][tree],
                activation=activation, loss=loss)
            
            # Extract top features from tree model for pruning
            self.params['models']['trees'][tree]['features'] = self.top_weights(
                model,input_tree,self.shape['inputs'])
            
            print('Growing Tree',tree)
            self.params['models']['trees'][tree]['model'] = self.tree_model(
                X_train=self.params['data']['train']['X_df'][self.params['models']['trees'][tree]['features']],
                y_train=self.params['data']['train']['X_df'][tree],
                X_val=self.params['data']['val']['X_df'][self.params['models']['trees'][tree]['features']],
                y_val=self.params['data']['val']['X_df'][tree],
                grow=True, activation=activation,loss=loss)
        #return self.params
    
    def build_dataset(self,X_df:pd.DataFrame,y_df:pd.DataFrame,output_name) -> tf.data.Dataset:
        """
        Converts dataframes to a tensorflow dataset.
        Args:
            X_df: input dataframe
            y_df: target dataframe
            output_name: name of the target column
        Returns:
            dataset: tensorflow dataset
        """
        # Rename target column to match model output
        y_df = y_df.rename(columns={self.target:output_name})

        dataset = tf.data.Dataset.from_tensor_slices((
            dict(X_df.drop(self.target,axis=1)),dict(y_df))).cache()
        return dataset
    
    def model_inputs(self) -> Tuple[dict, tf.keras.layers.DenseFeatures, tf.keras.layers.DenseFeatures]:
        """
        Creates model inputs.
        Args:
            None
        Returns:
            model_inputs: dictionary of model inputs
            emb_outputs: output of the embedding layer
            feature_outputs: output of the feature layer
        """
        model_inputs = {}

        # Create embedding layer
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

        # Create feature layer
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
        
        self.params['models']['EENN'] = {}
        self.params['models']['EENN']['features'] = list(model_inputs.keys())
        return model_inputs, emb_outputs, feature_outputs
    
    def build_base_model(self):
        """
        Builds the base model.
        Args:
            None
        Returns:
            None
        """
        model_inputs, emb_outputs, feature_outputs = self.model_inputs()
        tree_inputs = []
        tree_outputs = []

        # Extract tree model outputs
        for model in self.params['models']['trees'].values():
            input = tf.keras.layers.Concatenate()([v for k,v in model_inputs.items() if k in model['features']])

            layer = input
            for l in model['model'].layers[:-1]:
                layer = l(layer)
            output = model['model'].layers[-1](layer)

            tree_inputs.append(input)
            tree_outputs.append(layer)
            tree_outputs.append(output)
        
        # Combine tree models into one base model
        combined = tf.keras.layers.Concatenate()(tree_outputs + [emb_outputs, feature_outputs])
        output_name = self.target + '_0'
        output = tf.keras.layers.Dense(1,activation=self.activation,name=output_name)(combined)

        # Build and compile model
        base_model = tf.keras.Model(inputs=model_inputs, outputs=output)
        base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss=self.loss,metrics='accuracy')

        # Convert dataframes to tensorflow datasets
        self.params['data']['train']['dataset'] = self.build_dataset(
            self.params['data']['train']['X_df'],self.params['data']['train']['y_df'],output_name)
        self.params['data']['val']['dataset'] = self.build_dataset(
            self.params['data']['val']['X_df'],self.params['data']['val']['y_df'],output_name)
        
        # Train model
        base_model = self.dynamic_training(
            model=base_model, train_data=self.params['data']['train']['dataset'], 
            validation_data=self.params['data']['val']['dataset'], lock=True)
        
        self.params['models']['EENN']['model'] = base_model
        #return self.params
    
    def grow_model(
            self, model:tf.keras.models.Model,
            dropout:float=0.125,
            output_name:str='target') -> tf.keras.models.Model:
        """
        Builds a model with an additional output concatinated Dense layer with dropout.
        Args:
            model: pre-trained TensorFlow model
            dropout: dropout rate for the new layer
            output_name: name of the output layer
        Returns:
            new_model: new model
        """
        base_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-2].output)

        # Add new layer and dropout on top of second last layer output
        new_dense = tf.keras.layers.Dense(
            units=self.shape['layers'],activation=tf.keras.layers.LeakyReLU(alpha=0.01))(base_model.output)
        new_dropout = tf.keras.layers.Dropout(dropout)(new_dense)

        # Combine original model output with new dense layer
        combined = tf.keras.layers.Concatenate()([model.layers[-1].output, new_dropout])

        # New model output on top of concatenate layer
        new_output = tf.keras.layers.Dense(1,activation=self.activation,name=output_name)(combined)
        new_model = tf.keras.models.Model(inputs=model.inputs, outputs=new_output)

        self.params['data']['train']['dataset'] = self.build_dataset(
            self.params['data']['train']['X_df'],self.params['data']['train']['y_df'],output_name)
        self.params['data']['val']['dataset'] = self.build_dataset(
            self.params['data']['val']['X_df'],self.params['data']['val']['y_df'],output_name)

        # Compile and train model
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss=self.loss,metrics='accuracy')
        new_model = self.dynamic_training(
            model=new_model, 
            train_data=self.params['data']['train']['dataset'], 
            validation_data=self.params['data']['val']['dataset'], 
            lock=True)
        
        return new_model
    
    def dropout_scheduler(self,model:tf.keras.Model) -> Tuple[tf.keras.Model, float, float]:
        """
        Finds the optimal dropout rate for the base model.
        Args:
            model: base model
        Returns:
            best_model: model with the best dropout rate
            best_val_loss: validation loss of the best model
            best_dropout: best dropout rate
        """
        best_val_loss = float('inf')
        for dropout_rate in [0.0625, 0.125, 0.1875,0.25,0.375,0.5]:
            print("Testing Dropout:",str(dropout_rate))

            # Set dropout rate for all dropout layers and compile model
            for layer in model.layers:
                if isinstance(layer,tf.keras.layers.Dropout):
                    layer.rate = dropout_rate

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss=self.loss, metrics='accuracy')

            # Grow and train for first run, then retrain for additonal runs
            if dropout_rate == 0.0625:
                model = self.grow_model(model,dropout=dropout_rate,output_name=self.target+'_1')
            else:
                model = self.dynamic_training(
                    model=model, train_data=self.params['data']['train']['dataset'],
                    validation_data=self.params['data']['val']['dataset'], lock=True)
                
            # Evaluate model and update if necessary
            loss, _ = model.evaluate(
                self.params['data']['val']['dataset'].batch(self.params['max_batch_size']))
            
            if loss < best_val_loss:
                best_val_loss = loss
                best_dropout = dropout_rate
                best_weights = model.get_weights()
            else:
                model.set_weights(best_weights)
                break

        print("Best Dropout:",str(best_dropout),"Loss:",best_val_loss)
        return model, best_val_loss, best_dropout
    
    def evolve_model(self):
        """
        Evolves the EENN model.
        Args:
            None
        Returns:
            None
        """
        # Find optimal dropout rate for base model
        #output_name = self.target+'_1'
        best_model, best_val_loss, best_dropout = self.dropout_scheduler(
            self.params['models']['EENN']['model'])

        # Grow model until validation loss increases
        for layers in range(2,10):
            print("Current Layers:",str(layers))

            # Clone a new model to preserve best model
            new_model = tf.keras.models.clone_model(best_model)
            new_model.set_weights(best_model.get_weights())

            # Grow model and evaluate
            new_model = self.grow_model(new_model, dropout=best_dropout,output_name=self.target+'_'+str(layers))
            loss, _ = new_model.evaluate(
                self.params['data']['val']['dataset'].batch(self.params['max_batch_size']))
            
            # Update best model if necessary
            if loss < best_val_loss:
                best_val_loss = loss
                best_model = new_model
            else:
                break

        self.params['models']['EENN']['model'] = best_model
        #return self.params
    
    def train_ResMem(self):
        """
        Trains the ResMem model.
        Args:
            None
        Returns:
            None
        """
        # Calculate residuals using EENN
        yhat = self.params['models']['EENN']['model'].predict(self.params['data']['train']['dataset'].batch(4096))
        residule = self.params['data']['train']['y_df'] - yhat

        # Train ResMem model on residules
        X_df = self.params['data']['train']['X_df'].copy()
        _ = X_df.pop(self.target)
        resmem = KNeighborsRegressor(n_neighbors=5).fit(X_df, residule)
        self.params['models']['ResMem'] = resmem

    def prune_scalers(self):
        """
        Prune fitted scalers by removing the scaling parameters corresponding to dropped features.
        Args:
            None
        Returns:
            None
        """
        # Prune features from the normalizer
        all_features = self.params['models']['normalize']['features']
        kept_features = [feature for feature in all_features if feature in self.params['models']['EENN']['features']]

        # Create a mask that selects only the kept features
        mask = np.isin(all_features, kept_features)

        # Prune the scalers
        robust_scaler = self.params['models']['normalize']['robust']
        robust_scaler.center_ = robust_scaler.center_[mask]
        robust_scaler.scale_ = robust_scaler.scale_[mask]
        robust_scaler.n_features_in_ = len(kept_features)

        minmax_scaler = self.params['models']['normalize']['minmax']
        minmax_scaler.min_ = minmax_scaler.min_[mask]
        minmax_scaler.scale_ = minmax_scaler.scale_[mask]
        minmax_scaler.data_range_ = minmax_scaler.data_range_[mask]
        minmax_scaler.n_features_in_ = len(kept_features)

        # Update the 'features' list
        self.params['models']['normalize']['features'] = kept_features
        self.params['models']['normalize']['robust'] = robust_scaler
        self.params['models']['normalize']['minmax'] = minmax_scaler

    def training_pipeline(self) -> dict:
        """
        Trains the EENN model with ResMem.
        Args:
            None
        Returns:
            params: dictionary of final trained parameters
        """
        self.build_feature_models()
        self.build_base_model()
        self.evolve_model()
        self.train_ResMem()
        self.prune_scalers()

        models = {
            'robust':self.params['models']['normalize']['robust'],
            'minmax':self.params['models']['normalize']['minmax'],
            'EENN':self.params['models']['EENN']['model'],
            'ResMem':self.params['models']['ResMem'],
            'features':{
                'model_features':self.params['models']['EENN']['features'],
                'norm_features':self.params['models']['normalize']['features'],
                'target':self.target
            },
        }
        return models