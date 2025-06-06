import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, random_split

from xgboost import XGBClassifier
from sklearn.svm import SVC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  
num_cpus = torch.get_num_threads()
print(f"Number of CPUs available: {num_cpus}\n")

print(device)
torch.cuda.empty_cache()

print(torch.cuda.is_available())
num_channels=63

n_splits=3

print(f"Splits: {n_splits}")

batch_size=1600

classes= 3 # Define number of classes 

out= 10 # Define here

# support for surrogate gradients:

fraction_firing=0.1 # 10% neurons should fire
adjustment_rate= torch.tensor(0.1)
random_spike_rate=0.1 # induce random spiking/suppression in view of all/no neurons spiking

perturbation_strength= 0.05 # to avoid rank 1 matrices

# Select the top m eigenvectors (let's say m = 1)
m = 10
hidden_size= m # top features to consider

# Create a random forest feature selector here-

from collections import Counter

#- inbuilt RF package--

class RandomForestWrapper:
    def __init__(self, num_trees=10, max_depth=None, max_features='sqrt', n_jobs=-1):
        """
        Initializes the RandomForestWrapper.

        Parameters:
        - num_trees (int): Number of trees in the forest.
        - max_depth (int or None): Maximum depth of the trees.
        - max_features (int, float, str, or None): Number of features to consider when looking for the best split.
        - n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.
        """
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_jobs = n_jobs
        # self.model = RandomForestClassifier(
        #     n_estimators=self.num_trees,
        #     max_depth=self.max_depth,
        #     max_features=self.max_features,
        #     n_jobs=self.n_jobs,
        #     random_state=42  # For reproducibility
        # )
        self.model= XGBClassifier(n_estimators=100, max_depth=5, random_state=42)

    def fit(self, X, y):
        """
        Fits the Random Forest model.

        Parameters:
        - X (torch.Tensor or np.ndarray): Feature matrix.
        - y (torch.Tensor or np.ndarray): Target vector.
        """
        # Convert torch tensors to numpy arrays for sklearn compatibility
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts using the Random Forest model.

        Parameters:
        - X (torch.Tensor or np.ndarray): Feature matrix.

        Returns:
        - torch.Tensor: Predicted classes.
        """
        # Convert torch tensors to numpy arrays
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        predictions = self.model.predict(X)
        return torch.tensor(predictions)

    def feature_importances(self):
        """
        Returns the feature importances.

        Returns:
        - torch.Tensor: Feature importances.
        """
        return torch.tensor(self.model.feature_importances_)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X.cpu().numpy())
    
##--

def surrogate_grad(x):
    #return torch.clamp(alpha * (1.0 - torch.abs(x)), min=-1.0, max=1.0)
    #x = x - x.max(dim=0, keepdim=True)[0]
    x = x - torch.max(x)
    x=torch.clamp(x, min=-20.0, max=20.0)
    x= torch.clamp(torch.sigmoid(x), min= 0.2, max=0.5) 
    x += torch.randn_like(x) * perturbation_strength
    return x

def grad(x):
    #return torch.clamp(torch.sigmoid(x), min= 0.2, max=0.8) 
    return (torch.sigmoid(x)) 
    
def stable_softmax(x):
    z = x - x.max(dim=-1, keepdim=True)[0]  # Subtract max value for stability
    return torch.exp(z) / torch.exp(z).sum(dim=-1, keepdim=True)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
    def forward(self, x, prev, alpha, weights, bias):

        output = torch.mm(x, weights)

        # Precompute common terms
        prev_spike_inv = 1 - prev.spike  # Inverse of previous spike
        residual_term = (1 - alpha) * prev_spike_inv * prev.present_layer

        # Add residual term to the output
        output += residual_term

        return output
           
class layer(nn.Module): # to create a layer containing specified number of neurons
    def __init__(self):
        super(layer, self).__init__()
        self.neurons=[]
        
    def create(self, size):
        
        self.neurons=neuron_layer(size).to(device)
    
        return self.neurons

class fire_first_layer(nn.Module):
    def __init__(self):
        super(fire_first_layer, self).__init__()
    
    def forward(self,input):
        
        # Create a mask for neurons that fire (present_layer >= threshold)
        spikes = (input.present_layer >= input.threshold)

        b,c = spikes.shape  # or tensor.size()

        # Update the time for neurons that spiked
        #input.time = torch.where(spikes.bool(), input.curr_time, input.time)
        spikes=spikes.int()

        # Set the spike layer to 1 where the threshold is crossed, else 0

        # Step 3: Apply surrogate gradient during the backward pass
        
        # Adaptive threshold logic with random spiking/suppression
        
        if torch.all(spikes == 1):  # If all neurons spike
            #print("All neurons are spiking, reducing threshold and randomly suppressing some spikes.")
            input.threshold += adjustment_rate

            # Randomly suppress some spikes to maintain a balanced activity
            random_suppression = torch.bernoulli((1 - random_spike_rate) * torch.ones_like(spikes)).int()
            spikes = spikes * random_suppression

        elif torch.all(spikes == 0):  # If no neurons spike
            #print("No neurons are spiking, increasing threshold and randomly spiking some neurons.")
            input.threshold -= adjustment_rate

            # Randomly spike some neurons to maintain a desired activity
            random_spiking = torch.bernoulli(random_spike_rate * torch.ones_like(spikes)).int()
            spikes = spikes + random_spiking
   
        input.time = torch.where(spikes.bool(), input.curr_time, input.time)

        spikes=spikes.float()

        input.present_layer = spikes

        # Precompute exponential decay terms
        x= input.curr_time - input.time
        x = torch.clamp(x, max=40,min=0) 

        decay_pre = torch.exp(-(x) / input.tpre)
        decay_post = torch.exp(-(x) / input.tpost)

        # Efficiently update apre and apost using masks
        input.apre = input.present_layer + (1 - input.present_layer) * decay_pre
        input.apost = input.present_layer + (1 - input.present_layer) * decay_post

        return input.present_layer

class fire_neuron(nn.Module): # activation function
    def __init__(self):
        super(fire_neuron, self).__init__()

    def forward(self,input,weights):
        
        # Create a mask for neurons that fire (present_layer >= threshold)
        spikes = (input.present_layer >= input.threshold)
        
        b,c = spikes.shape  # or tensor.size()

        # Update the time for neurons that spiked
        #input.time = torch.where(spikes.bool(), input.curr_time, input.time)
        spikes=spikes.int()
        # Set the spike layer to 1 where the threshold is crossed, else 0

       # Adaptive threshold logic with random spiking/suppression
        if torch.all(spikes == 1):  # If all neurons spike
            #print("All neurons are spiking, reducing threshold and randomly suppressing some spikes.")
            input.threshold += adjustment_rate

            # Randomly suppress some spikes to maintain a balanced activity
            random_suppression = torch.bernoulli((1 - random_spike_rate) * torch.ones_like(spikes)).int()
            spikes = spikes * random_suppression

        elif torch.all(spikes == 0):  # If no neurons spike
            #print("No neurons are spiking, increasing threshold and randomly spiking some neurons.")
            input.threshold -= adjustment_rate

            # Randomly spike some neurons to maintain a desired activity
            random_spiking = torch.bernoulli(random_spike_rate * torch.ones_like(spikes)).int()
            spikes = spikes + random_spiking
        
        else:
            fraction_fired = torch.mean(spikes.float())

            # Adjust threshold based on firing rate relative to target
            if fraction_fired > fraction_firing:
                # Too many neurons fired, increase threshold
                input.threshold += adjustment_rate
            elif fraction_fired < fraction_firing:
                # Too few neurons fired, decrease threshold
                input.threshold -= adjustment_rate
        
        # Update the time for neurons that spiked
        input.time = torch.where(spikes.bool(), input.curr_time, input.time)
        
        # Clip threshold to ensure it stays within a reasonable range
        #input.threshold = torch.clamp(input.threshold, 0, 1)

        # Step 3: Apply surrogate gradient during the backward pass
        spikes=spikes.float()

        input.spike= spikes

        # STDP weight update with clamping for stability
        x= input.curr_time - input.time
        x = torch.clamp(x, max=40,min=0) 

        input.apre = input.spike + (1 - input.spike) * torch.exp(-(x) / input.tpre)
        input.apost = input.spike + (1 - input.spike) * torch.exp(-(x) / input.tpost)

        # Weight updates
        wt_plus= input.post * torch.mm(input.prev_apre.T, input.spike) * (1 - weights)
        wt_minus= input.pre * torch.mm(weights, torch.mm(input.apost.T, 1 - input.spike))

        weights += grad(wt_plus)- grad(wt_minus)

        weights = torch.clamp(weights, -1, 1)  # Ensure weights are bounded
        #print(f"weights: {weights}")

        return input.present_layer

class neuron_layer(nn.Module):
    def __init__(self,size):

        super(neuron_layer, self).__init__()
        
        self.present_layer=torch.zeros(size,device=device).to(device)
        self.spike=torch.zeros(size,device=device).to(device)
        self.prev_apre=torch.zeros(size,device=device).to(device)
        self.post=torch.tensor(0)
        self.pre=torch.tensor(0)
        self.apost=torch.zeros(size,device=device).to(device)
        self.curr_time=torch.tensor(0)
        self.time=torch.zeros(size,device=device).to(device)
        self.tpre=torch.tensor(0)
        self.tpost=torch.tensor(0)
        self.apre=torch.zeros(size,device=device).to(device) 
        self.threshold=torch.tensor(0)

        # flag to check if base threshold has been set
        self.set= False 

# Main module-
        
class NN(nn.Module):
    def __init__(self, learning, decay, threshold, pre, post, posttau, pretau,bias):
        super(NN, self).__init__()
        self.layers=nn.ModuleList([]).to(device)
        
        self.num_layers=0
        self.matrix= nn.ParameterList([]).to(device) # synaptic weights
        
        self.alpha=torch.tensor(learning)
        self.decay=decay
        self.threshold=torch.tensor(threshold)
        self.time=0
        self.pre=torch.tensor(pre)
        self.post=torch.tensor(post)
        self.tpre=torch.tensor(pretau)
        self.tpost=torch.tensor(posttau)

        self.forward_propagation=NeuralNet().to(device)
        self.activation=fire_neuron().to(device)
        self.first_layer=fire_first_layer().to(device)

        self.state=None
        self.bias=torch.tensor(bias)
        
        # Make changes here- 

        #self.normalise= nn.Softmax(dim=-1)
        self.normalise= lambda x: surrogate_grad(x)
        
    def create_connections(self, i, j):
        # update:

        # # Generate a matrix of random values between 0 and 1
        random_matrix = torch.rand(i, j, device=device)

        wt= nn.Parameter(random_matrix)
        nn.init.xavier_uniform_(wt)
        
        wt=wt.to(device)
        self.matrix.append((wt))
    
    def initialise_layer(self, neurons):
        create=layer().to(device)
        create=create.create(neurons)
        self.layers.append(create)
        self.num_layers=self.num_layers+1
    
    def forward(self, id, voltage):
        
        self.layers[id].post=self.post
        self.layers[id].pre=self.pre

        self.layers[id].tpre=self.tpre
        self.layers[id].tpost=self.tpost

        if not(self.layers[id].set):
            self.layers[id].threshold=self.threshold
            self.layers[id].set=True

        self.layers[id].curr_time=self.time

        # Parameters of a layer: 
        
        '''
        self.present_layer=torch.zeros(size)
        self.prev_apre=torch.zeros(size)
        self.spike=torch.zeros(size)
        self.aplus=torch.zeros(size)
        self.time=torch.zeros(size)
        self.apre=torch.zeros(size) 
        
        '''

        if id==0:
            #Update:

            desired_shape= self.layers[id].present_layer.shape
            original_tensor=voltage.clone().detach()

            padded_tensor = torch.zeros(desired_shape, dtype=original_tensor.dtype)

            # Copy the original tensor into the new tensor
            padded_tensor[:original_tensor.size(0), :original_tensor.size(1)] = original_tensor
            self.layers[id].present_layer=padded_tensor.to(device)

            self.state=self.first_layer.forward(self.layers[id])  

        else:
            self.layers[id].prev_apre=self.layers[id-1].apre
            
            self.layers[id].present_layer=self.forward_propagation.forward(self.state,self.layers[id],
                                                                           self.alpha,self.matrix[id-1].detach(),self.bias)
         
            self.layers[id].present_layer=self.normalise(self.layers[id].present_layer)

            if torch.isnan(self.layers[id].present_layer).any():
                print(f"present layer {id} post norm contains NaN")

            self.state=self.activation.forward(self.layers[id],self.matrix[id-1].detach())
        
        self.time+=10 # in milliseconds


class SNN_Model(nn.Module):
    
    def __init__(self,b,eh,out,hidden_layer_neurons, learning, decay, threshold, pre, post, posttau, pretau,bias):
        
        super(SNN_Model, self).__init__()
        
        # Update: Add a RNN layer too to learn residual connections

        self.input_size=eh

        self.hidden_size= hidden_size # top features to consider
        self.num_layers=2
        
        # Random Forest Parameters
        n_trees = 10  # Number of decision trees in the forest
    
        n_features = eh  # Number of features in the dataset
        max_features_per_tree = int(np.sqrt(n_features))  # Features to consider at each split

        self.snn=train(b,self.hidden_size,out,hidden_layer_neurons, learning, decay, threshold, pre, post, posttau, pretau,bias)

        self.pool= lambda x: surrogate_grad(x)
        
        # initialise random forest:
        #self.rnn = RandomForest(num_trees=n_trees,max_depth=5, max_features=max_features_per_tree)
        self.rnn=RandomForestWrapper(num_trees=100, max_depth=10, max_features="sqrt", n_jobs=-1)

        # Train a meta-classifier (Logistic Regression) on the stacked predictions
        #self.meta_clf = LogisticRegression(multi_class='multinomial', max_iter=200, solver='lbfgs')
        self.meta_clf= SVC(kernel='rbf', C=1.0, gamma='scale')

        self.epoch=30

    def forward(self,x):
        
        # y= x.contiguous().view(x.size(0), x.size(1)*x.size(2)).to(device)
        y=self.pool(x)
        # Get feature importances
        importances = self.rnn.feature_importances()

        # Select the most important features
        top_features = torch.topk(importances, k=self.hidden_size).indices
        y= y.clone().detach()
        y=y[:, top_features]
    
        for time in range(y.size(1)*self.epoch):

            for layer in range(self.snn.num_layers):
                
                # if time==0:
                #     print(f"Spike info: {self.snn.state}")

                self.snn.forward(layer,y)
            
        spike_train=self.snn.layers[-1].spike  # Main output of the network 
        
        #print(spike_train)
        return spike_train
    
def train(b,eh,out,hidden_layer_neurons, learning, decay, threshold, pre, post, posttau, pretau,bias): # eeg = samples X channels X time series
    
    # No of hidden layers
    hidden_layer_neurons=[eh]+hidden_layer_neurons+[out]
    number_of_hidden_layers=len(hidden_layer_neurons)
    
    SNN=NN(learning, decay, threshold, pre, post, posttau, pretau,bias).to(device)

    print("No of GPUs", torch.cuda.device_count())
    
    for i in range(number_of_hidden_layers-1):
        
        SNN.create_connections(hidden_layer_neurons[i],hidden_layer_neurons[i+1])
      
    for i in range(number_of_hidden_layers):

        SNN.initialise_layer([b,hidden_layer_neurons[i]])
    
    return SNN
        
def load2(folder):
    d=[]
    with open(folder, 'rb') as f:
        d=pickle.load(f)
    return d

def load():
    d=[]
    with open('data', 'rb') as f:
        d=pickle.load(f)
    return d

def store(folder, data):
    with open(folder,'wb') as f:
        pickle.dump(data,f)

## Model to learn in initial stages:

class learn_initial(nn.Module):

    def __init__(self):
        
        super(learn_initial, self).__init__()

        self.linear=nn.Sequential(
        nn.Dropout(0.4)
        #,nn.Softmax(dim=-1)
        )

    def forward(self,x):
        
        x=self.linear(x)
        #x=torch.mean(x,1,True) # Mean across rows
        
        return x

## Model to learn class labels:
        
class learn_labels(nn.Module):

    def __init__(self):
        
        super(learn_labels, self).__init__()

        self.linear=nn.Sequential(
            nn.Linear(out, 10),
            nn.Linear(10, 51),
            nn.Linear(51, 12),
            nn.Linear(12, classes))
        #self.normalise=nn.Softmax(dim=1)

    def forward(self,x):
        
        x=self.linear(x)
        #x=self.normalise(x)
        #x=torch.logit(x, eps=1e-6)
        return x

class max_pool(nn.Module):
    def __init__(self):
        
        super(max_pool, self).__init__()
        self.pool=nn.MaxPool1d(3, stride=2)

    def forward(self,x):

        x=self.pool(x)
        return x

# Replace StratifiedGroupKFold 

def stratified_k_fold(y, k):
    unique_classes, class_counts = np.unique(y, return_counts=True)

    print(f"No of classes: {len(unique_classes)}")

    #unique_classes= torch.tensor(unique_classes,device=device, dtype=torch.int64)
    indices_per_class = {cls : np.where(y == cls)[0] for cls in unique_classes}
    folds = []

    for i in range(k):
        fold_indices = []
        for cls in unique_classes:
            class_indices = indices_per_class[cls]
            fold_size = len(class_indices) // k
            fold_indices.extend(class_indices[i * fold_size:(i + 1) * fold_size])
        folds.append(fold_indices)

    return torch.tensor(folds).to(device=device)

# Main function that makes the calls:

def main(dataset,label,hidden_layers,time_stamps_train,time_stamps_test,learning,decay,threshold,pre,post,posttau,pretau,bias):

    ## Loading data
    
    data=dataset[:,:,[2, 3, 5, 6, 7, 13, 14,18, 19,24,27,29,35, 36, 38, 39, 40,43, 44,50, 51,55, 56, 57, 58, 59, 62],:]
    
    # apply average pooling

    m = nn.AvgPool2d((3, 2), stride=(2, 1)).to(device)

    data=m(torch.Tensor(np.asarray(data)))

    b,e,h,w=data.shape

    eh=e*h
    #eh=1

    b=batch_size
    print(b)

    epochs=30
    #epochs=2

    conformer=SNN_Model(b,eh,out,hidden_layers,learning,decay,threshold,pre,post,posttau,pretau,bias).to(device)

    label_learner=learn_labels().to(device)
    initial_learner=learn_initial().to(device)

    optimizer = optim.Adam(label_learner.parameters(), lr = 0.001)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1, patience=3)

    # Define a loss function (e.g., Mean Squared Error loss) for RNN
    #criterion2 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()

    folds = stratified_k_fold(label,n_splits)

    label=torch.tensor(label,dtype=torch.int64, device=device)

    accuracy=dict()
    accuracy_rf=dict()
    accuracy_snn=dict()

    epoch_loss=dict()

    for fold, fold_indices in enumerate(folds):

        # Create train and test subsets
        if n_splits==1:
            train_indices = (folds[i] for i in range(n_splits) )

        else:
            train_indices = (folds[i] for i in range(n_splits) if i != fold)

        train_indices = torch.cat(list(train_indices), dim=0)

        test_indices = fold_indices

        #train_features = Subset(data, train_indices)
        train_features= data[train_indices.to('cpu')]

        #test_features = Subset(data, test_indices)
        test_features= data[test_indices.to('cpu')]

        #train_labels = Subset(label, train_indices)
        train_labels= label[train_indices]

        #test_labels = Subset(label, test_indices)
        test_labels= label[test_indices]
        
        # split for train and validation - torch enabled 

        # Create a dataset
        compiled_dataset = TensorDataset(train_features, train_labels)

        # Define the split ratio
        train_size = int(0.8 * len(compiled_dataset))
        val_size = len(compiled_dataset) - train_size
        
        # Split the dataset
        train_dataset, val_dataset = random_split(compiled_dataset, [train_size, val_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_data = torch.utils.data.TensorDataset(test_features, test_labels)
       
        print("Start")

        test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = False)

        loss=batch_train(epoch_loss,conformer,initial_learner,label_learner,criterion,criterion2,optimizer,train_loader,time_stamps_train,epochs)
        
        # Validation
        validation_loss=batch_validation(conformer,initial_learner,label_learner,criterion,val_loader,time_stamps_train)
        scheduler.step(validation_loss)
        
        batch_test(accuracy,accuracy_rf, accuracy_snn,conformer,initial_learner,label_learner,test_loader,time_stamps_test)

        # debug purposes:
        print(f"spikes: {conformer.snn.layers[-1].spike}")
 
    for keys in epoch_loss:

        epoch_loss[keys]=sum(epoch_loss[keys])/len(epoch_loss[keys])

    return accuracy[time_stamps_test], accuracy_rf[time_stamps_test], accuracy_snn[time_stamps_test]

def store(folder, data):
    with open(folder,'wb') as f:
        pickle.dump(data,f) 

def batch_train(epoch_loss, conformer: SNN_Model,initial_learner: learn_initial,label_learner: learn_labels,criterion,criterion2,optimizer, dataloader,time_stamps,epochs:int):
    
    train_running_loss=0
    train_loss=0
    
    label_learner.train()
    initial_learner.train()

    print("Training:")

    # Select the top m eigenvectors (let's say m = 1)
    #m = 10

    for batch_index, (x, y) in enumerate(dataloader):

        optimizer.zero_grad()

        x=x.to(device)
        #y=y.to(device)

        b,e,h,w = x.shape

        x=x.view(b,e*h,w)

        y = y.to(torch.int64)
    
        #print(f"Time Stamp {t+1} of {time_stamps}")

        eeg_temp=x[:,:,:time_stamps]

        eeg=initial_learner(eeg_temp)

        eeg= torch.abs(torch.fft.fft2(eeg))

        #eeg,_ =torch.topk(eeg.view(eeg.shape[0], -1), k=2*m, dim=1)

        # rnn training ---
        #create one hot encoding:
        num_classes=classes 

        # Create one-hot encoding using torch.nn.functional.one_hot

        labels= y.clone().detach()

        y_= eeg.contiguous().view(eeg.size(0),eeg.size(2)*conformer.input_size).to(device)
       
        cov=y_

        cov = torch.cov(cov.T) # if y of n,m, torch.cov returns n,n
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # Sort eigenvalues in descending order and get corresponding indices
        #sorted_indices = torch.argsort(eigenvalues, descending=True)

        #eigenvectors=eigenvectors.to(torch.complex64)
        #print(eigenvalues)

        y_ = y_@eigenvectors[:, -m:].real
     
        # train rf:
        conformer.rnn.fit(y_, labels)

        rf_preds_train = conformer.rnn.predict_proba(y_)

        # SNN: 
        outputs=conformer.forward(y_)
        output=outputs[:len(y)] 

        ## learning labels

        for epoch in range(epochs):
        #   
            optimizer.zero_grad()

            outputs=label_learner(output)

            predicted=outputs

            loss = criterion(predicted, y)

            if epoch not in epoch_loss:
                epoch_loss[epoch]=[loss.item()]
            
            else:
                epoch_loss[epoch].append(loss.item())

            # backpropagation
            loss.backward()
            # update optimizer parameters
            optimizer.step()
        
        train_running_loss += loss.item()
            
        #print(loss)

        cnn_preds_train = torch.softmax(predicted, dim=1)
        stacked_train = torch.cat((torch.tensor(rf_preds_train).cuda(),cnn_preds_train), dim=1)
        stacked_train_np = stacked_train.detach().cpu().numpy()
        conformer.meta_clf.fit(stacked_train_np, labels.cpu().numpy()) 
        
    train_loss = train_running_loss / (len(dataloader)*epochs)
    
    return train_loss

def batch_validation(conformer: SNN_Model,initial_learner,label_learner, criterion,dataloader,time_stamps):
    

    train_running_loss=0
    train_loss=0

    label_learner.eval()
    initial_learner.eval()

    print("Validating")

    # picking number of features
    #m=10

    with torch.no_grad():

        for batch_index, (x, y) in enumerate(dataloader):

            b,e,h,w = x.shape
            x=x.to(device)

            x=x.view(b,e*h,w)

            y = y.to(torch.int64)

            eeg=x[:,:,:time_stamps]

            eeg=initial_learner(eeg)
            eeg= torch.abs(torch.fft.fft2(eeg))

            y_= eeg.contiguous().view(eeg.size(0),eeg.size(2)*conformer.input_size).to(device)
            cov=y_

            cov = torch.cov(cov.T) # if y of n,m, torch.cov returns n,n
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            # Sort eigenvalues in descending order and get corresponding indices
            #sorted_indices = torch.argsort(eigenvalues, descending=True)
            
            #eigenvectors=eigenvectors.to(torch.complex64)
            #y_ = eigenvectors[:, :m].real
            y_ = y_@eigenvectors[:, -m:].real
            # y_= conformer.ica.fit_transform(y_.cpu().numpy())
            # y_=torch.tensor(y_,device=device)

            outputs=conformer.forward(y_)
                  
            outputs=outputs[:len(y)]

            #print(outputs)
                
            ## learning labels

            outputs=label_learner(outputs)

            predicted=outputs

            loss = criterion(predicted, y)
            train_running_loss += loss.item()

            #print(y)
            #print(predicted)
            
    train_loss = train_running_loss / len(dataloader)
    
    return train_loss

def batch_test(accuracy, accuracy_rf, accuracy_snn, conformer:SNN_Model,initial_learner,label_learner,dataloader,time_stamps):
    

    y_target = []
    collection=dict()

    rf_collection=dict()
    snn_collection=dict()

    label_learner.eval()
    initial_learner.eval()

    acc=0

    print("Testing:")

    # picking number of features:
    #m=10

    with torch.no_grad():

        for batch_index, (x, y) in enumerate(dataloader):

            #x=x.to(device)
            #y=y.to(device)

            b,e,h,w = x.shape
            x=x.to(device)

            x=x.view(b,e*h,w)

            y = y.to(torch.int64)

            y_target.extend(y.cpu().data.numpy().tolist())

            eeg=x[:,:,:time_stamps]

            eeg=initial_learner(eeg)
            eeg= torch.abs(torch.fft.fft2(eeg))

            y_= eeg.contiguous().view(eeg.size(0),eeg.size(2)*conformer.input_size).to(device)
          
            cov=y_

            cov = torch.cov(cov.T) # if y of n,m, torch.cov returns n,n
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            # Sort eigenvalues in descending order and get corresponding indices
            #sorted_indices = torch.argsort(eigenvalues, descending=True)

            #eigenvectors=eigenvectors.to(torch.complex64)

            #y_ = eigenvectors[:, :m].real
            y_ = y_@eigenvectors[:, -m:].real
            # y_= conformer.ica.fit_transform(y_.cpu().numpy())
            # y_=torch.tensor(y_,device=device)

            rf_preds_train = conformer.rnn.predict_proba(y_)

            y_pred = conformer.rnn.predict(y_)

            outputs=conformer.forward(y_)

            outputs=outputs[:len(y)]

            ## learning labels

            outputs=label_learner(outputs)

            cnn_preds_train = torch.softmax(outputs, dim=1)
            stacked_train = torch.cat((torch.tensor(rf_preds_train).cuda(),cnn_preds_train), dim=1)
            stacked_train_np = stacked_train.detach().cpu().numpy()
            
            # Make predictions using the meta-classifier
            prediction = conformer.meta_clf.predict(stacked_train_np)
            
            _,prediction_snn=torch.max(outputs.data,1)

            if time_stamps not in collection:
                collection[time_stamps]=[]
                rf_collection[time_stamps]=[]
                snn_collection[time_stamps]=[]
            
            collection[time_stamps].extend(prediction.tolist())
            rf_collection[time_stamps].extend(y_pred.cpu().data.numpy().tolist())
            snn_collection[time_stamps].extend(prediction_snn.tolist())


    target_vec = torch.tensor(y_target)

    for keys in collection: 
       
        actual_vec = torch.tensor(collection[keys])

        actual_vec_rf = torch.tensor(rf_collection[keys])

        actual_vec_snn = torch.tensor(snn_collection[keys])

        # Element-wise comparison
        comparison = (target_vec == actual_vec)

        comparison_rf=(target_vec== actual_vec_rf)

        comparison_snn= (target_vec== actual_vec_snn)

        # Count the number of True values (matching elements)
        comparison = comparison.sum().item()
        comparison=comparison/target_vec.size(0)
                
        if keys not in accuracy:

            accuracy[keys]=[]
        
        accuracy[keys].append(comparison)

        # Count the number of True values (matching elements)
        comparison_rf = comparison_rf.sum().item()
        comparison_rf=comparison_rf/target_vec.size(0)
                
        if keys not in accuracy_rf:

            accuracy_rf[keys]=[]
        
        accuracy_rf[keys].append(comparison_rf)

        # Count the number of True values (matching elements)
        comparison_snn = comparison_snn.sum().item()
        comparison_snn=comparison_snn/target_vec.size(0)
                
        if keys not in accuracy_snn:

            accuracy_snn[keys]=[]
        
        accuracy_snn[keys].append(comparison_snn)
