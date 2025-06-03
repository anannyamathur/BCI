# Implementing State of the Art- Symmetric Convolutional and Adversarial Neural Network

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

#from sklearn.model_selection import StratifiedGroupKFold,GroupKFold,LeaveOneGroupOut,GroupShuffleSplit

#import sklearn.metrics as metrics

#from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  
num_cpus = torch.get_num_threads()
print(f"Number of CPUs available: {num_cpus}\n")
#device='cpu'

print(device)
torch.cuda.empty_cache()

print(torch.cuda.is_available())
num_channels=63

n_splits=3

print(f"Splits: {n_splits}")

batch_size=1600

classes= 3 # Number of classes 

out= 10 # Define here

# support for surrogate gradients:

fraction_firing=0.1 # 10% neurons should fire
adjustment_rate= torch.tensor(0.1)
random_spike_rate=0.1 # induce random spiking/suppression in view of all/no neurons spiking

perturbation_strength= 0.05 # to avoid rank 1 matrices

# Select the top m eigenvectors (let's say m = 1)
m = 10
hidden_size=m # top features to consider

#---
# Create a random forest feature selector here-

# SOTA Model-

class Generator(nn.Module):
    """Generator Network using PCA features."""
    def __init__(self, pca_features, img_channels):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(pca_features, 256*8*8)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, features):
        x = self.fc1(features)
        x = x.view(-1, 256, 8, 8)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        img = self.tanh(self.conv3(x))
        return img

class Discriminator(nn.Module):
    """Discriminator Network using PCA features."""
    def __init__(self, img_channels,num_classes):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(256*8*8, 256)
       
        self.fc_adv = nn.Linear(256, 1)
        self.fc_class = nn.Linear(256, num_classes)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        
        #print(features.size())
        x = self.leaky_relu(self.conv1(features))

        #print(features.size())

        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        validity = self.sigmoid(self.fc_adv(x))
        class_logits = self.fc_class(x)
        return validity, class_logits

class SCANN(nn.Module):
    """Symmetric Convolutional and Adversarial Neural Network (SCANN) with PCA features."""
    def __init__(self, pca_features, img_channels, num_classes, lr=0.0002, beta1=0.5):

        super(SCANN, self).__init__()
        self.channels= pca_features
        self.generator = Generator(pca_features, img_channels)
        self.discriminator = Discriminator(img_channels, num_classes)
        self.optim_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.adversarial_loss = nn.BCELoss().to(device)
        self.classification_loss = nn.CrossEntropyLoss().to(device)

    def train_step(self, pca_features, real_labels):

        # generate fake data:
        batch_size = pca_features.size(0)
        fake_data= torch.randn(batch_size, self.channels).to(device)
        # hidden_size= number of pca features extracted

        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # Train Generator
        self.optim_G.zero_grad()

        gen_imgs = self.generator(fake_data)
        valid_imgs= self.generator(pca_features)

        validity_fake, _ = self.discriminator(gen_imgs)
        validity_real, _ = self.discriminator(valid_imgs)

        g_loss = self.adversarial_loss(validity_fake, valid)
        g_loss_real = self.adversarial_loss(validity_real, valid)

        g_loss= (g_loss+g_loss_real)/2

        g_loss.backward()
        self.optim_G.step()

        # Train Discriminator
        self.optim_D.zero_grad()
        
        validity_real, class_logits_real = self.discriminator(valid_imgs.detach())

        # Adversarial Loss
        real_loss = self.adversarial_loss(validity_real, valid)

        validity_fake, _ = self.discriminator(gen_imgs.detach())
        fake_loss = self.adversarial_loss(validity_fake, fake)
        adv_loss = (real_loss + fake_loss) / 2

        # Classification Loss
        class_loss = self.classification_loss(class_logits_real, real_labels)

        # Total Discriminator Loss
        d_loss = adv_loss + class_loss
        d_loss.backward()
        self.optim_D.step()

        return g_loss.item(), d_loss.item(), class_loss.item()

    def predict(self, pca_features):
        """Predict class labels for given PCA features."""
        real_imgs= self.generator(pca_features)

        _, class_logits = self.discriminator(real_imgs)
        class_probs = torch.softmax(class_logits, dim=1)
        predicted_labels = torch.argmax(class_probs, dim=1)
        return predicted_labels
    
##############

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

def main(dataset,label,time_stamps_train,time_stamps_test):

    ## Loading data
    
    data=dataset[:,:,[2, 3, 5, 6, 7, 13, 14,18, 19,24,27,29,35, 36, 38, 39, 40,43, 44,50, 51,55, 56, 57, 58, 59, 62],:]
    
    # apply average pooling

    m = nn.AvgPool2d((3, 2), stride=(2, 1)).to(device)

    data=m(torch.Tensor(np.asarray(data)))

    b,e,h,w=data.shape

    eh=e*h
    b=batch_size
    print(b)

    epochs=1
   
    conformer= SCANN(pca_features=hidden_size, img_channels=hidden_size//2,
                     num_classes=classes).to(device)

   
    initial_learner=learn_initial().to(device)

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

        train_features= data[train_indices.to('cpu')]
        test_features= data[test_indices.to('cpu')]

        train_labels= label[train_indices]
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

        loss=batch_train(epoch_loss,conformer,initial_learner,train_loader,time_stamps_train,epochs)
        
        # Validation
        validation_loss=batch_validation(conformer,initial_learner,val_loader,time_stamps_train)
        
        batch_test(accuracy, conformer,initial_learner,test_loader,time_stamps_test)

    for keys in epoch_loss:

        epoch_loss[keys]=sum(epoch_loss[keys])/len(epoch_loss[keys])

    return accuracy[time_stamps_test]

def store(folder, data):
    with open(folder,'wb') as f:
        pickle.dump(data,f) 


def batch_train(epoch_loss, conformer: SCANN,initial_learner: learn_initial,dataloader,time_stamps,epochs:int):
    
    train_running_loss=0
    train_loss=0

    initial_learner.train()

    print("Training:")

    # Select the top m eigenvectors (let's say m = 1)
    #m = 10
    for epoch in range(epochs):
        for batch_index, (x, y) in enumerate(dataloader):

            x=x.to(device)
            #y=y.to(device)

            b,e,h,w = x.shape

            x=x.view(b,e*h,w)

            y = y.to(device)
        
            #print(f"Time Stamp {t+1} of {time_stamps}")

            eeg_temp=x[:,:,:time_stamps]

            eeg=initial_learner(eeg_temp)

            eeg= torch.abs(torch.fft.fft2(eeg))

            #create one hot encoding:
            num_classes=classes 

            # Create one-hot encoding using torch.nn.functional.one_hot

            labels= y.clone().detach()
            
            y_= eeg.contiguous().view(eeg.size(0),eeg.size(2)*e*h).to(device)
       
            cov=y_

            cov = torch.cov(cov.T) # if y of n,m, torch.cov returns n,n
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            # Sort eigenvalues in descending order and get corresponding indices
          
            y_ = y_@eigenvectors[:, -m:].real
        
            g_loss, d_loss, loss = conformer.train_step(y_,labels)

            ## learning labels

            if epoch not in epoch_loss:
                epoch_loss[epoch]=[loss]
            
            else:
                epoch_loss[epoch].append(loss)

            train_running_loss += loss
                
    train_loss = train_running_loss / (len(dataloader)*epochs)
    
    return train_loss

def batch_validation(conformer: SCANN,initial_learner,dataloader,time_stamps):
    

    train_running_loss=0
    train_loss=0

    initial_learner.eval()

    print("Validating")

    # picking number of features
    #m=10

    with torch.no_grad():

        for batch_index, (x, y) in enumerate(dataloader):

            b,e,h,w = x.shape
            x=x.to(device)

            x=x.view(b,e*h,w)

            y = y.to(device)

            eeg=x[:,:,:time_stamps]

            eeg=initial_learner(eeg)
            eeg= torch.abs(torch.fft.fft2(eeg))

            y_= eeg.contiguous().view(eeg.size(0),eeg.size(2)*e*h).to(device)
            
            cov=y_

            cov = torch.cov(cov.T) # if y of n,m, torch.cov returns n,n
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            # Sort eigenvalues in descending order and get corresponding indices
            #sorted_indices = torch.argsort(eigenvalues, descending=True)
            
            y_ = y_@eigenvectors[:, -m:].real
           
            predicted_labels= conformer.predict(y_)
            accuracy = (predicted_labels == y).float().mean().item()
            train_running_loss += accuracy

    train_loss = train_running_loss / len(dataloader)
    
    return train_loss

def batch_test(accuracy, conformer:SCANN,initial_learner,dataloader,time_stamps):
    

    y_target = []
    collection=dict()
    initial_learner.eval()

    acc=0

    print("Testing:")

    # picking number of features:
    #m=10

    with torch.no_grad():

        for batch_index, (x, y) in enumerate(dataloader):

            b,e,h,w = x.shape
            x=x.to(device)

            x=x.view(b,e*h,w)

            y = y.to(device)

            y_target.extend(y.cpu().data.numpy().tolist())

            eeg=x[:,:,:time_stamps]

            eeg=initial_learner(eeg)
            eeg= torch.abs(torch.fft.fft2(eeg))

            y_= eeg.contiguous().view(eeg.size(0),eeg.size(2)*e*h).to(device)
            
            cov=y_

            cov = torch.cov(cov.T) # if y of n,m, torch.cov returns n,n
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            # Sort eigenvalues in descending order and get corresponding indices
          
            y_ = y_@eigenvectors[:, -m:].real
          
            ## learning labels
            predicted_labels= conformer.predict(y_)
            
            if time_stamps not in collection:
                collection[time_stamps]=[]
                
            collection[time_stamps].extend(predicted_labels.tolist())
           
    target_vec = torch.tensor(y_target)

    for keys in collection:

        actual_vec = torch.tensor(collection[keys])

        # Element-wise comparison
        comparison = (target_vec == actual_vec)

        # Count the number of True values (matching elements)
        comparison = comparison.sum().item()
        comparison=comparison/target_vec.size(0)
                
        if keys not in accuracy:

            accuracy[keys]=[]
        
        accuracy[keys].append(comparison)
