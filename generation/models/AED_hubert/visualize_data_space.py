import os
from os.path import isdir, join
import pickle
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import constants.constants as constants
from utils.create_final_file import createFinalFile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages
from utils.labels import one_hot_to_label, get_color, get_labels
from utils.data_utils import separate_openface_features
import torch
import umap


class VisualizeData():
    def __init__(self):
        super(VisualizeData, self).__init__()


    def draw_umap(self, saved_path, epoch, predict_values_list, all_labels_list, labels_to_test, create_umap_list = True, list_of_umap=None):
        if create_umap_list:
            list_of_umap = {}

        #creation of the path to save the data
        path = saved_path + "data_visualization/" 
        if not os.path.exists(path):
            os.makedirs(path)
        for label_type in labels_to_test:
            path_1 = path + label_type + "/"
            if not os.path.exists(path_1):
                    os.makedirs(path_1)
            path_2 = path_1 + epoch + "/"
            if not os.path.exists(path_2):
                os.makedirs(path_2)

        #reshape the data to be able to use UMAP
        all_behaviours = torch.cat(predict_values_list, dim=0)
        eye, pose_r, au = separate_openface_features(all_behaviours)
        for behaviour in [["all", all_behaviours], ["eye", eye], ["pose", pose_r], ["au", au]]:
            behaviour_name = behaviour[0]
            path_3 = path_2 + behaviour_name + "/"
            if not os.path.exists(path_3):
                os.makedirs(path_3)

            stacked_values_list = behaviour[1]
            stacked_values_list = stacked_values_list.reshape(stacked_values_list.shape[0], -1)
            print("shape of data in UMAP", stacked_values_list.shape)
    
            for n_components in [3]:
                for n in (5, 10, 20, 50, 100):
                    for d in (0.0, 0.1, 0.25):
                        if create_umap_list:
                            umap_model = umap.UMAP(n_neighbors=n, min_dist=d, n_components=n_components, random_state=2855)
                            umap_fit = umap_model.fit(stacked_values_list)
                            list_of_umap[behaviour_name + "_n_" + str(n) + "_d_" + str(d)] = umap_fit
                        else:
                            umap_fit = list_of_umap[behaviour_name + "_n_" + str(n) + "_d_" + str(d)]
                        result = umap_fit.transform(stacked_values_list)
                        for label_type in labels_to_test:
                            print("*"*10,"UMAP",label_type,"*"*10)
                            labels_list = all_labels_list[label_type]
                            #print(labels_list)
                            stacked_labels_list = torch.cat(labels_list, dim=0)
                            str_stacked_labels_list = [one_hot_to_label(label, label_type) for label in stacked_labels_list]
                            str_stacked_labels_list = np.array(str_stacked_labels_list)
                            # print("stacked_values_list", stacked_values_list.shape, "str_stacked_labels_list", str_stacked_labels_list.shape)
                            # print("stacked_labels_list", str_stacked_labels_list)
                            
                            colors = get_color(label_type)
                            print(colors)
                            fig = plt.figure()
                            if n_components == 2:
                                ax = fig.add_subplot(111)
                            if n_components == 3:
                                ax = fig.add_subplot(111, projection='3d')
                            with PdfPages(path_3+"dim_"+str(n_components)+"_n_"+str(n)+"_d_"+str(d)+".pdf") as pdf:
                                for label in get_labels(label_type):
                                    print("*"*5,label,"*"*5)
                                    indices = np.where(str_stacked_labels_list == label) # get the indices of the current label in the dataset
                                    print("number of frames", len(result), "-- number of", label, "frames", len(result[indices]))
                                    if n_components == 2:
                                        ax.scatter(result[indices, 0], result[indices, 1], label=label, c=colors[label])
                                    if n_components == 3:
                                        ax.scatter(result[indices, 0], result[indices, 1], result[indices, 2], label=label, c=colors[label])
                                plt.title('UMAP Visualization of generated data for '+label_type+ "\\" + behaviour_name + "_n_" + str(n) + "_d_" + str(d))
                                ax.view_init(elev=20, azim=30)  # Perspective 1
                                plt.legend()
                                pdf.savefig()
                                ax.view_init(elev=30, azim=60)  # Perspective 2
                                pdf.savefig()
                                ax.view_init(elev=40, azim=90)  # Perspective 3
                                pdf.savefig()
                                ax.view_init(elev=90, azim=-90)  # XY
                                pdf.savefig()
                                ax.view_init(elev=0, azim=-90)  # XZ
                                pdf.savefig()
                                ax.view_init(elev=0, azim=0)  # YZ
                                pdf.savefig()
                                ax.view_init(elev=-90, azim=90)  # -XY
                                pdf.savefig()
                                ax.view_init(elev=0, azim=90)  # -XZ
                                pdf.savefig()
                                ax.view_init(elev=0, azim=180)  # -YZ
                                pdf.savefig()                                                                                                                                                                       
                                pdf.savefig()
                                plt.close()
        return list_of_umap
    
    def draw_latent_umap(self, saved_path, epoch, predict_values_list, all_labels_list, labels_to_test):

        #creation of the path to save the data
        path = saved_path + "data_visualization/latent_values/" 
        if not os.path.exists(path):
            os.makedirs(path)
        for label_type in labels_to_test:
            path_1 = path + label_type + "/"
            if not os.path.exists(path_1):
                    os.makedirs(path_1)
            path_2 = path_1 + epoch + "/"
            if not os.path.exists(path_2):
                os.makedirs(path_2)

        #reshape the data to be able to use UMAP
        stacked_values_list = torch.cat(predict_values_list, dim=0)
        stacked_values_list = stacked_values_list.reshape(stacked_values_list.shape[0], -1)
        print("shape of data in UMAP", stacked_values_list.shape)

        for n_components in [3]:
            for n in (5, 10, 20, 50, 100):
                for d in (0.0, 0.1, 0.25):

                    umap_model = umap.UMAP(n_neighbors=n, min_dist=d, n_components=n_components, random_state=2855)
                    umap_fit = umap_model.fit(stacked_values_list)
                    result = umap_fit.transform(stacked_values_list)
                    for label_type in labels_to_test:
                        print("*"*10,"UMAP",label_type,"*"*10)
                        labels_list = all_labels_list[label_type]
                        #print(labels_list)
                        stacked_labels_list = torch.cat(labels_list, dim=0)
                        str_stacked_labels_list = [one_hot_to_label(label, label_type) for label in stacked_labels_list]
                        str_stacked_labels_list = np.array(str_stacked_labels_list)
                        # print("stacked_values_list", stacked_values_list.shape, "str_stacked_labels_list", str_stacked_labels_list.shape)
                        # print("stacked_labels_list", str_stacked_labels_list)
                        colors = get_color(label_type)
                        print(colors)
                        fig = plt.figure()
                        if n_components == 2:
                            ax = fig.add_subplot(111)
                        if n_components == 3:
                            ax = fig.add_subplot(111, projection='3d')
                        with PdfPages(path_2+"dim_"+str(n_components)+"_n_"+str(n)+"_d_"+str(d)+".pdf") as pdf:
                            for label in get_labels(label_type):
                                print("*"*5,label,"*"*5)
                                indices = np.where(str_stacked_labels_list == label) # get the indices of the current label in the dataset
                                print("number of frames", len(result), "-- number of", label, "frames", len(result[indices]))
                                if n_components == 2:
                                    ax.scatter(result[indices, 0], result[indices, 1], label=label, c=colors[label])
                                if n_components == 3:
                                    ax.scatter(result[indices, 0], result[indices, 1], result[indices, 2], label=label, c=colors[label])
                            plt.title('UMAP Visualization of generated data for '+label_type+ "\\" + "_n_" + str(n) + "_d_" + str(d))
                            ax.view_init(elev=20, azim=30)  # Perspective 1
                            plt.legend()
                            pdf.savefig()
                            ax.view_init(elev=30, azim=60)  # Perspective 2
                            pdf.savefig()
                            ax.view_init(elev=40, azim=90)  # Perspective 3
                            pdf.savefig()
                            ax.view_init(elev=90, azim=-90)  # XY
                            pdf.savefig()
                            ax.view_init(elev=0, azim=-90)  # XZ
                            pdf.savefig()
                            ax.view_init(elev=0, azim=0)  # YZ
                            pdf.savefig()
                            ax.view_init(elev=-90, azim=90)  # -XY
                            pdf.savefig()
                            ax.view_init(elev=0, azim=90)  # -XZ
                            pdf.savefig()
                            ax.view_init(elev=0, azim=180)  # -YZ
                            pdf.savefig()                                                                                                                                                                       
                            pdf.savefig()
                            plt.close()


    def visualize_generated_sequences_data(self, epoch, trainer_args, dm):
        checkpoint_epoch = int(epoch) - 1
        model_path = constants.saved_path + "epoch="+str(checkpoint_epoch)+".ckpt"
        model = constants.model.load_from_checkpoint(model_path)
        model.pose_scaler=dm.y_scaler

        trainer = pl.Trainer(**trainer_args)
        predictions = trainer.predict(model, dm)
        predict_values_list = []
        predict_latent_list = []
        labels_list_dialog_act = []
        labels_list_valence = []
        labels_list_arousal = []
        labels_list_certainty = []
        labels_list_dominance = []
        labels_list_genders = []

        for batch in predictions:
            predict_values_list.append(batch[1])
            predict_latent_list.append(batch[3])
            labels_list_genders.append(batch[4])
            labels_list_dialog_act.append(batch[5])
            labels_list_valence.append(batch[6])
            labels_list_arousal.append(batch[7])
            labels_list_certainty.append(batch[8])
            labels_list_dominance.append(batch[9])

        all_list = {"gender": labels_list_genders, "dialog_act": labels_list_dialog_act, "valence": labels_list_valence, "arousal": labels_list_arousal, "certainty": labels_list_certainty, "dominance": labels_list_dominance}
        labels_to_test = ["gender"]
        #list_of_umap = pickle.load(open(join("generation","saved_models","15-04-2024_trueness_1_CGAN_12",'list_of_umap.pkl'), 'rb')) #TODO à paramétrer
        self.draw_umap(constants.saved_path, epoch, predict_values_list, all_list, labels_to_test, create_umap_list=True, list_of_umap=None)
        #self.draw_latent_umap(constants.saved_path, epoch, predict_latent_list, all_list, labels_to_test)
        

        



            


