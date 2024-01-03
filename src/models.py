import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from typing import Tuple, Union

class EENN:
    """ Class for training an EENN model. """
    def __init__(self,params:dict,shape:dict={'size':'auto'}):
        """ Initialize EENN class. """
        self.params = params
        self.target = list(params['target'].keys())[0]
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
        # Set initial batch size and learning rate
        min_batch_size = self.params['min_batch_size']
        max_batch_size = self.params['max_batch_size']
        batch_size = min_batch_size
        best_val_loss = float('inf')
        best_weights = None
        cascade = False
        fails = 0
        success = 0

        # Lock layers for first training round if pruning
        if lock:
            for layer in model.layers[:-3]:
                layer.trainable = False

        # Train model until patience is reached
        while fails < patience:
            print("current batch size:",batch_size)
            # Train model for one epoch (pd.DataFrame)
            if isinstance(train_data, tuple):
                history = model.fit(x=train_data[0], y=train_data[1],
                                    epochs=1, batch_size=batch_size,
                                    validation_data=validation_data, 
                                    validation_batch_size=max_batch_size,verbose=1)
                
            # Train model for one epoch (tf.data.Dataset)
            else:
                history = model.fit(train_data.batch(batch_size).shuffle(
                    self.params['data']['train']['X_df'].shape[0]).prefetch(tf.data.experimental.AUTOTUNE),
                    validation_data=validation_data.batch(max_batch_size), 
                    epochs=1, verbose=1)
                
            # Unlock layers after first training round
            for layer in model.layers:
                layer.trainable = True
            
            # Get the validation accuracy for this epoch
            current_val_loss = history.history['val_loss'][-1]

            # Save weights if validation accuracy has improved
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_weights = model.get_weights()
                fails = 0
                success += 1
                lr_double = False

                # If cascade is False, double batch size & learning rate
                if cascade == False:
                    if batch_size * 2 <= max_batch_size:
                        batch_size *= 2
                        lr_double = True
                        success = 0
                    elif success == 3:
                        success = 0
                        lr_double = True
                else:
                    if success == 4:
                        success = 0
                        lr_double = True

                if lr_double:
                    lr_double = False
                    if model.optimizer.lr * 2 > 0.25:
                        model.optimizer.lr.assign(0.25)
                    else:
                        model.optimizer.lr.assign(model.optimizer.lr * 2)
            else:
                # If validation accuracy has not improved, restore best weights and halve learning rate
                model.set_weights(best_weights)
                model.optimizer.lr.assign(model.optimizer.lr / 2)
                fails += 1
                success = 0

                # If validation accuracy has not improved for 3 epochs, activate cascade
                if fails == 3 and cascade == False:
                    cascade = True
                    print("cascade activated")
                    fails = 0

                # If cascade is True, halve batch size
                if cascade and batch_size > min_batch_size:
                    batch_size = int(batch_size / 2)
                    model.optimizer.lr.assign(model.optimizer.lr / 2)
        return model
    
    def tree_model(
            self,
            train_data: Union[Tuple[pd.DataFrame, pd.DataFrame], tf.data.Dataset],
            validation_data: Union[Tuple[pd.DataFrame, pd.DataFrame], tf.data.Dataset],
            grow:bool=False,activation:str='linear') -> tf.keras.models.Sequential:
        """
        Builds a regression model.
        Args:
            train_data: input variables to train the model
            validation_data: input variables to validate the model
            grow: boolean to determine if model should be grown
            activation: activation function for the output layer
        Returns:
            model: trained regression model
        """
        # Build model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(len(train_data[0].columns),)))
    
        # add dropout and dense layers if second iteration
        if grow:
            model.add(tf.keras.layers.Dropout(0.125))
            model.add(tf.keras.layers.Dense(
                self.shape['aux'],activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
            model.add(tf.keras.layers.Dropout(0.125))

        model.add(tf.keras.layers.Dense(1,activation=activation))

        if activation == 'linear':
            loss = 'mean_squared_error'
        else:
            loss = 'binary_crossentropy'

        # Compile and train model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss=loss,metrics='accuracy')
        print(model.summary())
        model = self.dynamic_training(
            model=model, train_data=train_data, validation_data=validation_data)
        
        return model
    
    def grow_model(
            self, model:tf.keras.models.Model,
            train_data: Union[Tuple[pd.DataFrame, pd.DataFrame], tf.data.Dataset],
            validation_data: Union[Tuple[pd.DataFrame, pd.DataFrame], tf.data.Dataset],
            units:int, activation:str='linear') -> tf.keras.models.Model:
        """
        Builds a model with an additional output concatinated Dense layer with dropout.
        Args:
            model: pre-trained TensorFlow model
            train_data: input variables to train the model
            validation_data: input variables to validate the model
            units: number of nodes in the new layer
            activation: activation function for the output layer
        Returns:
            new_model: new model
        """
        # Extract input and output layers from original model
        input_layer = model.layers[0].input
        output_layer = model.layers[-1].output

        # Add new layer and dropout on top of original model inputs
        new_dense = tf.keras.layers.Dense(
            units,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input_layer)
        new_dropout = tf.keras.layers.Dropout(0.125)(new_dense)

        # Combine original model outputs with new layers
        combined = tf.keras.layers.Concatenate()([output_layer,new_dropout])

        # New model output on top of concatenate layer
        new_output = tf.keras.layers.Dense(1,activation=activation)(combined)
        new_model = tf.keras.models.Model(inputs=input_layer, outputs=new_output)

        if activation == 'linear':
            loss = 'mean_squared_error'
        else:
            loss = 'binary_crossentropy'

        # Compile and train model
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss=loss,metrics='accuracy')
        new_model = self.dynamic_training(model=new_model, train_data=train_data, validation_data=validation_data, lock=True)
        return new_model
    
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
            train_data=(
                self.params['data']['train']['X_df'][input_features],
                self.params['data']['train']['X_df'][self.target]),
            validation_data=(
                self.params['data']['val']['X_df'][input_features],
                self.params['data']['val']['X_df'][self.target]),
            activation=self.params['target'][self.target])
        
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
            train_data=(
                self.params['data']['train']['X_df'][self.params['models']['trees'][self.target]['features']],
                self.params['data']['train']['X_df'][self.target]),
            validation_data=(
                self.params['data']['val']['X_df'][self.params['models']['trees'][self.target]['features']],
                self.params['data']['val']['X_df'][self.target]),
            grow=True,
            activation=self.params['target'][self.target])
        
        # Build and prune feature models for each tree
        for tree in tree_features:
            print('Running tree regression:',tree)
            self.params['models']['trees'][tree] = {}
            input_tree = [feature for feature in input_features if feature != tree]

            # Determine loss function based on feature characteristics
            if ((self.params['data']['train']['X_df'][tree].dtype == int) &
                (self.params['data']['train']['X_df'][tree].between(0,1,inclusive='both').all())):
                activation = 'sigmoid'
            else:
                activation = 'linear'

            model = self.tree_model(
                train_data=(
                    self.params['data']['train']['X_df'][input_tree],
                    self.params['data']['train']['X_df'][tree]),
                validation_data=(
                    self.params['data']['val']['X_df'][input_tree],
                    self.params['data']['val']['X_df'][tree]),
                activation=activation)
            
            # Extract top features from tree model for pruning
            self.params['models']['trees'][tree]['features'] = self.top_weights(
                model,input_tree,self.shape['inputs'])
            
            print('Growing Tree',tree)
            self.params['models']['trees'][tree]['model'] = self.tree_model(
                train_data=(
                    self.params['data']['train']['X_df'][self.params['models']['trees'][tree]['features']],
                    self.params['data']['train']['X_df'][tree]),
                validation_data=(
                    self.params['data']['val']['X_df'][self.params['models']['trees'][tree]['features']],
                    self.params['data']['val']['X_df'][tree]),
                grow=True,
                activation=activation)
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

        return model_inputs, emb_outputs, feature_outputs
    
    def base_model(self):
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
            concat = tf.keras.layers.Concatenate()([v for k,v in model_inputs.items() if k in model['features']])
            tree_inputs.append(model['model'](concat))

            cloned_layer = tf.keras.models.clone_model(model['model'].layers[-3])
            cloned_layer.build(model['model'].layers[-3].input_shape)
            cloned_layer.set_weights(model['model'].layers[-3].get_weights())
            tree_outputs.append(cloned_layer(concat))

        # Combine tree models with concatenate layer
        combined = tf.keras.layers.Concatenate()(tree_inputs + tree_outputs + [emb_outputs, feature_outputs])

        # Build on top of concatenate layer
        layer = tf.keras.layers.Dropout(0.125)(combined)
        output = tf.keras.layers.Dense(1,name="CatalogPrice")(layer)

        # Build and compile model
        base_model = tf.keras.Model(inputs=model_inputs, outputs=output)
        base_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss='mean_squared_error',metrics='accuracy')

        # Convert dataframes to tensorflow datasets
        self.params['data']['train']['dataset'] = self.build_dataset(
            self.params['data']['train']['X_df'],self.params['data']['train']['y_df']).cache()
        self.params['data']['val']['dataset'] = self.build_dataset(
            self.params['data']['val']['X_df'],self.params['data']['val']['y_df']).cache()
        
        # Train model
        base_model = self.dynamic_training(
            model=base_model, train_data=self.params['data']['train']['dataset'], 
            validation_data=self.params['data']['val']['dataset'],
            min_batch_size=128,max_batch_size=4096)
        
        self.params['models']['EENN'] = base_model
    
    def grow_modelX(self,base_model:tf.keras.Model,dropout:float=0.125) -> tf.keras.Model:
        """
        Builds a new model with an additional Dense layer, and copies the weights from the base model.
        Args:
            base_model: base model
            dropout: dropout rate
        Returns:
            new_model: new model
        """
        self.params['models']['weights'], self.params['models']['biases'] = base_model.layers[-1].get_weights()

        # Extract base model inputs and outputs
        model_inputs = base_model.inputs
        combined = base_model.layers[-2].output

        ################# add auto actiation #################
        # Add new layer and dropout on top of base model
        layer = LSR_Dense(self.shape['layers'],first_node_activation='linear',pretrained_weights=self.params['models']['weights'],pretrained_biases=self.params['models']['biases'])(combined)
        layer = tf.keras.layers.Dropout(dropout)(layer)
        output = tf.keras.layers.Dense(1, name=self.params['target'])(layer)
        new_model = tf.keras.Model(inputs=model_inputs, outputs=output)

        # Copy weights from base model
        for new_layer, old_layer in zip(new_model.layers[:-2], base_model.layers[:-1]):
            new_layer.set_weights(old_layer.get_weights())

        # Compile and train model
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss='mean_squared_error', metrics='accuracy')
        new_model = self.dynamic_training(
            model=new_model, train_data=self.params['data']['train']['dataset'],
            validation_data=self.params['data']['val']['dataset'],
            min_batch_size=128,max_batch_size=4096,lock=True)
        
        return new_model
    
    def dropout_scheduler(self,base_model:tf.keras.Model) -> Tuple[tf.keras.Model, float, float]:
        """
        Finds the optimal dropout rate for the base model.
        Args:
            base_model: base model
        Returns:
            best_model: model with the best dropout rate
            best_val_loss: validation loss of the best model
            best_dropout: best dropout rate
        """
        best_dropout = 0.0625
        print("Testing Dropout:",str(best_dropout))
        # Build initial model with minimum dropout rate
        new_model = self.grow_model(base_model,dropout=best_dropout)
        best_weights = new_model.get_weights()
        best_val_loss, _ = new_model.evaluate(self.params['data']['val']['dataset'].batch(4096))
        
        # Test increasing dropout rates until validation loss increases
        for dropout_rate in [0.125, 0.1875,0.25,0.375,0.5]:
            print("Testing Dropout:",str(dropout_rate))
            for layer in new_model.layers:
                if isinstance(layer,tf.keras.layers.Dropout):
                    layer.rate = dropout_rate

            new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss='mean_squared_error', metrics='accuracy')
            new_model = self.dynamic_training(
                model=new_model, train_data=self.params['data']['train']['dataset'],
                validation_data=self.params['data']['val']['dataset'],
                min_batch_size=128,max_batch_size=4096,lock=True)
            
            loss, _ = new_model.evaluate(self.params['data']['val']['dataset'].batch(4096))
            if loss < best_val_loss:
                best_val_loss = loss
                best_dropout = dropout_rate
                best_weights = new_model.get_weights()
            else:
                new_model.set_weights(best_weights)
                break

        print("Best Dropout:",str(best_dropout),"Loss:",best_val_loss)
        return new_model, best_val_loss, best_dropout
    
    def evolve_model(self):
        """
        Evolves the EENN model.
        Args:
            None
        Returns:
            None
        """
        # Find optimal dropout rate for base model
        best_model, best_val_loss, best_dropout = self.dropout_scheduler(self.params['models']['EENN'])
        grow = True
        layers = 1

        # Grow model until validation loss increases
        while grow:
            layers += 1
            print("Current Layers:",str(layers))

            new_model = tf.keras.models.clone_model(best_model)
            new_model.set_weights(best_model.get_weights())

            new_model = self.grow_model(new_model, dropout=best_dropout)
            loss, _ = new_model.evaluate(self.params['data']['val']['dataset'].batch(4096))
            if loss < best_val_loss:
                best_val_loss = loss
                best_model = new_model
            else:
                grow = False

        self.params['models']['EENN'] = best_model
    
    def train_ResMem(self):
        """
        Trains the ResMem model.
        Args:
            None
        Returns:
            None
        """
        # Calculate residuals using EENN
        yhat = self.params['models']['EENN'].predict(self.params['data']['train']['dataset'].batch(4096))
        residule = self.params['data']['train']['y_df'] - yhat

        # Train ResMem model on residules
        X_df = self.params['data']['train']['X_df'].copy()
        _ = X_df.pop(self.params['target'])
        resmem = KNeighborsRegressor(n_neighbors=5).fit(X_df, residule)
        self.params['models']['ResMem'] = resmem

    def training_pipeline(self) -> dict:
        """
        Trains the EENN model with ResMem.
        Args:
            None
        Returns:
            params: dictionary of model parameters
        """
        self.build_feature_models()
        self.base_model()
        self.evolve_model()
        self.train_ResMem()
        return self.params