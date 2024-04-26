import os
import sys
import constants.constants as constants
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.logger import MyLogger
import torch
import torch.nn as nn
from utils.model_parts import DownDiscr
import pytorch_lightning as pl
from utils.labels import one_hot_to_index, get_labels_to_index
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

class OutputRedirector:
    def __init__(self, file_path):
        self.file_path = file_path
        self.stdout = sys.stdout

    def __enter__(self):
        self.file = open(self.file_path, "a")
        sys.stdout = self.file
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        self.file.close()

class Classify():
    def __init__(self, label, epoch=None):
        super(Classify, self).__init__()
        print("*"*10, "classification of", label, "*"*10)
        saved_path = "/gpfsdswork/projects/rech/urk/uln35en/Scripts/non-verbal-behaviours-generation-hubert/generation/saved_models/classifier/"+label+"/"
        self.checkpoint_path = saved_path + "checkpoint/"
        self.train_file = saved_path + "training.txt"
        self.result_file = saved_path +"results.txt"
        checkpoint_callback = ModelCheckpoint(dirpath=self.checkpoint_path, every_n_epochs=10, save_top_k = -1, filename='{epoch}')

        constants.number_of_gpu = int(os.environ['SLURM_GPUS_ON_NODE']) * int(os.environ['SLURM_NNODES'])
        print("number of gpu", constants.number_of_gpu)
        trainer_args = {'accelerator': 'gpu', 
                "max_epochs": 400, 
                "check_val_every_n_epoch": 1, 
                "log_every_n_steps":10,
                "enable_progress_bar": False, 
                "callbacks": [checkpoint_callback],
                }
        trainer = pl.Trainer(**trainer_args)

        dm = self.load_data()
        if epoch == None:
            model = Model(label)
            with OutputRedirector(self.train_file):
                trainer.fit(model, dm)
        else:
            model = Model.load_from_checkpoint(self.checkpoint_path + "epoch="+str(epoch)+".ckpt", label=label)
            with OutputRedirector(self.result_file):
                predictions = trainer.predict(model, dm)
                self.evaluate(predictions, label)

    def load_data(self):
        dm = constants.customDataModule()
        dm.prepare_data()
        dm.setup(stage="fit")
        dm.is_prepared = True
        return dm
    
    def conf_matrix(self, y_test, y_pred):
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Matrice de confusion :\n")
        print(str(conf_matrix))

        # Calcul du nombre d'erreurs par classe
        errors_per_class = []
        for i in range(len(conf_matrix)):
            errors_per_class.append(sum(conf_matrix[i]) - conf_matrix[i][i])
        print("\nNombre d'erreurs par classe :\n")
        for i in range(len(errors_per_class)):
            print(f"Classe {i}: {errors_per_class[i]} erreurs")
    
    def evaluate(self, predictions, label):
        all_predictions = []
        all_labels = []
        for prediction in predictions:
            print(len(prediction))
            outputs = prediction[0]
            item, predicted = torch.max(outputs, 1)
            if(label == "gender"):
                labels = prediction[1]
            if(label == "dialog_act"):
                labels = prediction[2]
            elif(label == "valence"):
                labels = prediction[3]
            elif(label == "arousal"):
                labels = prediction[4]
            elif(label == "certainty"):
                labels = prediction[5]
            elif(label == "dominance"):
                labels = prediction[6]
            all_predictions.extend(predicted)
            all_labels.extend(labels)
        all_labels = [one_hot_to_index(ele, label) for ele in all_labels]
        accuracy = accuracy_score(all_labels, all_predictions)
        print("label en num: "+str(get_labels_to_index(label))+"\n")
        print(f'Accuracy on test set: {accuracy:.4f}')
        self.conf_matrix(all_labels, all_predictions)
    

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class ConvClassifier(nn.Module):

    def __init__(self, latent_dim, nb_classes):
        super().__init__()

        self.conv1 = Conv(latent_dim, 32)
        self.conv2 = Conv(32, 64)
        self.fc1 = torch.nn.Linear(64 * 25, 64)
        self.fc2 = torch.nn.Linear(64, nb_classes)

    def forward(self, x): #(512,6)
        x = x.swapaxes(1, 2)
        x = self.conv1(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class Model(pl.LightningModule):
    def __init__(self, label):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.label = label
        if(self.label == "dialog_act"):
            self.classifier = ConvClassifier(28, 11)
        elif(self.label == "valence"):
            self.classifier = ConvClassifier(28, 4)
        elif(self.label == "arousal"):
            self.classifier = ConvClassifier(28, 4)
        elif(self.label == "certainty"):
            self.classifier = ConvClassifier(28, 4)
        elif(self.label == "dominance"):
            self.classifier = ConvClassifier(28, 4)
        elif(self.label == "gender"):
            self.classifier = ConvClassifier(28, 2)

        self.create_loss()

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=0.001)


    def training_step(self, batch, batch_idx):
        inputs_audio, targets, gender, dialog_act, valence, arousal, certainty, dominance = batch
        
        opt = self.optimizers(self.classifier)
        if(self.label == "dialog_act"):
            labels = dialog_act
        elif(self.label == "valence"):
            labels = valence
        elif(self.label == "arousal"):
            labels = arousal
        elif(self.label == "certainty"):
            labels = certainty
        elif(self.label == "dominance"):
            labels = dominance
        elif(self.label == "gender"):
            labels = gender

        opt.zero_grad()
        outputs = self.classifier(targets)
        loss = self.criterion(outputs, labels)
        loss.backward()
        opt.step()
        self.loss.append(loss)


    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.loss).mean()
        avg_val_loss = torch.stack(self.val_loss).mean()
        if((self.current_epoch+1) % 10 == 0):
            print(f'Epoch [{self.current_epoch+1}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        self.clear_loss()

    def validation_step(self, batch, batch_idx):
        inputs_audio, targets, gender, dialog_act, valence, arousal, certainty, dominance = batch

        if(self.label == "dialog_act"):
            labels = dialog_act
        elif(self.label == "valence"):
            labels = valence
        elif(self.label == "arousal"):
            labels = arousal
        elif(self.label == "certainty"):
            labels = certainty
        elif(self.label == "dominance"):
            labels = dominance
        elif(self.label == "gender"):
            labels = gender
        
        with torch.no_grad():
            outputs = self.classifier(targets)
            val_loss = self.criterion(outputs, labels)
        
        self.val_loss.append(val_loss)
        ##si la valeur est plus basse que la plus basse, on sauvegarde et on génère et on calcul l'accuracy 

        return val_loss
    
    def predict_step(self, batch, batch_idx):
        inputs_audio, targets, gender, dialog_act, valence, arousal, certainty, dominance = batch
        with torch.no_grad():
            outputs = self.classifier(targets).squeeze(2)
        
        return outputs, gender, dialog_act, valence, arousal, certainty, dominance


    ################ Loss processing #########################
    def create_loss(self):
        self.loss = []
        self.val_loss = []

    def clear_loss(self):
        self.loss.clear()
        self.val_loss.clear()
    

    
        