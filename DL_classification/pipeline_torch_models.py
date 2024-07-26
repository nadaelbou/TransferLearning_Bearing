
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from datetime import datetime
import copy

class PipelineTorch():
    def __init__(self, model, config):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=model.to(self.device)
        self.best_model=self.model
        self.batch_size = config['model_config']['batch_size']
        self.n_epochs = config['model_config']['n_epochs']
        self.learning_rate = config['model_config']['learning_rate']
        self.t_0=config['model_config']['t_0']
        self.t_mult=config['model_config']['t_mult']
        self.eta_min=config['model_config']['eta_min']
        self.optimizer = self.__set_optimizer_transfer()
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, 
                                        T_0 = self.t_0,# Number of iterations for the first restart
                                        T_mult = self.t_mult, # A factor increases TiTi​ after a restart
                                        eta_min = self.eta_min) # Minimum learning rate
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.early_stop_patience=config['model_config']['early_stop_patience']
        self.history = dict(train=[], val=[])
        # initialize tracker for minimum validation loss:
        self.valid_loss_min=np.Inf 
        self.train_loss_min=np.Inf
        self.epochs_since_improvement = 0
        self.start_epoch=0
        self.best_epoch=0
        self.artifact_path=config['model_config']['artifact_path']
        self.__create_artifact_path(self.artifact_path)
        self.correct = 0
        self.total = 0

    def __create_artifact_path(self, artifact_path):  
        if not os.path.exists(artifact_path):
            os.makedirs(artifact_path)
  
    def train(self, train_dataset, val_dataset, version):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,shuffle=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size,shuffle=False)
        self.model=self.model.to(self.device)
        for epoch in range(1, self.n_epochs + 1):
            epoch_start_time = datetime.now()
            train_loss = 0.0
            valid_loss = 0.0
            ###################
            # train the model #
            ###################
            # set the module to training mode
            self.model = self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                # move to GPU if available
                data = data.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(epoch + batch_idx / len(train_loader))
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            print(f"Epoch [{epoch}/{self.n_epochs}], Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            epoch_end_time = datetime.now()
            print('Duration of training at epoch {} is : {} seconds.'.format(epoch, (epoch_end_time - epoch_start_time))) 
            # set the module to validation mode
            self.model = self.model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    # move to GPU if available
                    
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            self.history['train'].append(train_loss)
            self.history['val'].append(valid_loss)       
            # print training/validation statistics
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                        epoch, 
                        train_loss,
                        valid_loss
            ))
            #if the validation loss has decreased, save the model, if not, check early-stopping:
            if (valid_loss < self.valid_loss_min):
                if (self.valid_loss_min - valid_loss > 0.0001*self.valid_loss_min or epoch==1):
                    self.epochs_since_improvement = 0
                else:
                    self.epochs_since_improvement += 1 
                    torch.manual_seed(epoch*10)
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,shuffle=True)
                    print(f'seed has been changed. The new torch seed is {torch.initial_seed()}')
                print('Validation loss has descreased ({:.6f}-->{:.6f}). Saving model...'.format(self.valid_loss_min, valid_loss))
                self.best_model=copy.deepcopy(self.model)
                self.best_epoch = epoch 
                state = {'epoch': epoch , 'state_dict': self.best_model.state_dict(),'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),'history':self.history}
                torch.save(state, os.path.join(self.artifact_path, f"model_{version}.pth"))
                self.valid_loss_min=valid_loss
                if (train_loss < self.train_loss_min):
                    self.train_loss_min=train_loss
            else:
                self.epochs_since_improvement += 1  
                torch.manual_seed(epoch*10)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,shuffle=True)
                print(f'seed has been changed. The new torch seed is {torch.initial_seed()}')     
            if self.epochs_since_improvement >= self.early_stop_patience:
                print(f"No improvement in training! Training stopped.")
                print(f'Last improved epoch is #{self.best_epoch}')
                break
        return self

    def predict(self, dataset):
        t_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,shuffle=False)
        self.model=self.model.to(self.device)
        # monitor test loss and accuracy
        average_loss = 0.
        losses =list()
        predictions=list()
        real_labels=list()
        self.model = self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(t_loader):      
                # move to GPU if available
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                average_loss = average_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - average_loss))
                losses.append(loss.item()) 
                # convert output probabilities to predicted class
                pred = output.data.max(1, keepdim=True)[1]
                predictions.extend(pred.cpu().numpy())
                real_labels.extend(target.cpu().numpy())
                 # compare predictions to true label
                self.correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
                self.total += data.size(0)
                #predictions=np.vstack([predictions,output.detach().cpu().numpy()])
                acc = (100. * self.correct / self.total, self.correct, self.total)
        #print('Test Loss: {:.6f}\n'.format(average_loss))  
        #print('\Accuracy: %2d%% (%2d/%2d)' % (100. * self.correct / self.total, self.correct, self.total))       
        return losses, average_loss, predictions, real_labels, acc
      
    def load_checkpoint(self, version):
        # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
            print("=> loading checkpoint '{}'".format(os.path.join(self.artifact_path, f"model_{version}.pth")))
            checkpoint=torch.load(os.path.join(self.artifact_path, f"model_{version}.pth"),map_location=self.device)
            self.start_epoch = checkpoint['epoch']
            self.history=checkpoint['history']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                      .format(os.path.join(self.artifact_path, f"model_{version}.pth"), checkpoint['epoch']))
         
    def __set_optimizer_transfer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(params=params, lr=self.learning_rate)
                    
                    
                    
      