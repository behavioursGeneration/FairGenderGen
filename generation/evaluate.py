import os
from os.path import isdir, isfile
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dtaidistance import dtw
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import constants.constants as constants

all_features = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry", "pose_Rz", 
                "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]

motion_group_features = {
                "init" : [["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "pose_Rx", "pose_Ry", "pose_Rz"], 3],
                "first_eye" : [["gaze_0_x", "gaze_0_y", "gaze_0_z"], 3],
                "second_eye" : [["gaze_1_x", "gaze_1_y", "gaze_1_z"], 3],
                "gaze_angle" : [["gaze_angle_x", "gaze_angle_y"], 2],
                "pose" : [["pose_Rx", "pose_Ry", "pose_Rz"], 3],
                "AU01" : [["AU01_r"], 1],
                "AU02" : [["AU02_r"], 1],
                "AU04" : [["AU04_r"], 1],
                "AU05" : [["AU05_r"], 1],
                "AU06" : [["AU06_r"], 1],
                "AU07" : [["AU07_r"], 1],
                "AU09" : [["AU09_r"], 1],
                "AU10" : [["AU10_r"], 1],
                "AU12" : [["AU12_r"], 1],
                "AU14" : [["AU14_r"], 1],
                "AU15" : [["AU15_r"], 1],
                "AU17" : [["AU17_r"], 1],
                "AU20" : [["AU20_r"], 1],
                "AU23" : [["AU23_r"], 1],
                "AU25" : [["AU25_r"], 1],
                "AU26" : [["AU26_r"], 1],
                "AU45" : [["AU45_r"], 1],
                }

#creation of different mesures for each type of output
types_output = {"eyes" : ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y"],
                "pose" : ["pose_Rx", "pose_Ry", "pose_Rz"],
                "sourcils" : ["AU01_r", "AU02_r", "AU04_r"],
                "yeux" : ["AU05_r","AU06_r", "AU07_r", "AU45_r"],
                "bouche_sup" : ["AU09_r", "AU10_r", "AU12_r", "AU14_r"],
                "ouv_bouche": ["AU23_r", "AU25_r", "AU26_r"],
                "bouche_inf" : ["AU15_r", "AU17_r", "AU20_r"]
                }

class Evaluate():
    def __init__(self, datasets, epoch, dtw_arg, pca_arg, curve_arg, curveVideo_arg, motion_arg, dm=None):
        super(Evaluate, self).__init__()
        if(datasets != ""):
            constants.datasets = datasets.split(",")

        if(dm == None):
            dm = constants.customDataModule(fake_examples=False, predict=True)
            dm.prepare_data()
        dm.is_prepared = False
        dm.setup(stage="evaluate")
        self.y_final_keys, self.y_final_data = dm.test_dataset.get_final_test_videos()

        self.path_data_out = constants.output_path + constants.model_path + "/epoch_" + str(epoch) + "/"
        if(not isdir(self.path_data_out)):
            raise Exception(self.path_data_out + "is not a directory. Data are not generated.")

        self.path_evaluation = constants.evaluation_path + constants.model_path + "/epoch_" + str(epoch) + "/"
        if(not isdir(self.path_evaluation)):
            os.makedirs(self.path_evaluation, exist_ok=True)

        if(motion_arg):
            self.create_motion_measures()

        if(curve_arg):
            self.create_curve()
        
        if(curveVideo_arg):
            self.create_curves_video()

        if(dtw_arg):
            self.create_dtw()

        if(pca_arg):
            self.create_pca()

    def getData(self, all_features, features):
        gened_seqs = []
        real_seqs = []
        for index, key in enumerate(self.y_final_keys):
            #real data
            final_data = self.y_final_data[index]
            if features != None:
                final_data = final_data[features]
            real_seqs.append(final_data)

            #generated data
            pd_file = pd.read_csv(self.path_data_out+key+".csv")
            pd_file = pd_file[all_features]
            if features != None:
                pd_file = pd_file[features]
            gened_seqs.append(pd_file)

        self.gened_frames = np.concatenate(gened_seqs, axis=0)
        self.real_frames = np.concatenate(real_seqs, axis=0)
        
        return self.real_frames, self.gened_frames


    def getVideoData(self, key, index, all_features, features = None):
        #generated data
        pd_file = pd.read_csv(self.path_data_out+key+".csv")[all_features]
        if features != None:
            pd_file = pd_file[features]
        gened_frames = pd_file
        #real data
        real_frames = self.y_final_data[index]
        if features != None:
            real_frames = real_frames[features]
        
        #print("*"*10, key, len(real_frames), len(gened_frames))
        return real_frames, gened_frames
    
    def getRealVideoData(self, index, features = None):
        #real data
        real_frames = self.y_final_data[index]
        if features != None:
            real_frames = real_frames[features]
        return real_frames
    
    def getGenVideoData(self, key, all_features, features = None):
        #generated data
        pd_file = pd.read_csv(self.path_data_out+key+".csv")[all_features]
        if features != None:
            pd_file = pd_file[features]
        gened_frames = pd_file
        return gened_frames


    def compute_motion(self, data, derivative, dim=3):
        # velocity is the first dervative of position (La vitesse d'un objet est une mesure de la rapidité avec laquelle il change de position par rapport au temps.)
        # Second derivative of position is acceleration (L'accélération est une mesure de la rapidité avec laquelle la vitesse d'un objet change par rapport au temps.)
        # Third derivative of position is jerk (la mesure de la rapidité avec laquelle l'accélération change par rapport au temps.)
        if(len(data.shape) == 1):
            data = data.unsqueeze(-1)
        motion_data = np.diff(data, n=derivative, axis=0) #une approximation de la Xieme dérivée par rapport au temps. Cela donne les velocity/acc/jerks pour chaque frame.

        num_motion = motion_data.shape[0]
        num_joints = motion_data.shape[1] // dim
        motion_norms = np.zeros((num_motion, num_joints))

        for i in range(num_motion):
            for j in range(num_joints):
                x1 = j * dim + 0
                x2 = j * dim + dim
                motion_norms[i, j] = np.linalg.norm(motion_data[i, x1:x2]) #la norme le long d'un axe spécifique dans le cas de matrices multidimensionnelles
        average = np.mean(motion_norms, axis=0)

        # Take into account that frame rate was 25 fps
        # obtain a coherente unity in secondes
        scaled_av = average * (25 ** derivative)

        return scaled_av.item(0)
    
    def create_motion_measures(self):
        print("*"*10, "MOTION DATA", "*"*10)
        self.create_real_motion_measures()
        
        motion_tab = {}
        for i, features in enumerate(motion_group_features.keys()):
            motion_tab[features] = {"velocity": [], "acceleration" : [], "jerk": []}
            for index, key in enumerate(self.y_final_keys):
                
                dim = motion_group_features[features][1]
                gened_frames = self.getGenVideoData(key, all_features, motion_group_features[features][0])
        
                motion_tab[features]["velocity"].append(self.compute_motion(gened_frames, derivative=1, dim=dim))
                motion_tab[features]["acceleration"].append(self.compute_motion(gened_frames, derivative=2, dim=dim))
                motion_tab[features]["jerk"].append(self.compute_motion(gened_frames, derivative=3, dim=dim))
                
            motion_tab[features]["velocity"] = np.mean(motion_tab[features]["velocity"])
            motion_tab[features]["acceleration"] = np.mean(motion_tab[features]["acceleration"])
            motion_tab[features]["jerk"] = np.mean(motion_tab[features]["jerk"])

        df_acc = pd.DataFrame(motion_tab)
        df_acc['mean'] = df_acc.mean(axis=1)
        df_acc.to_excel(self.path_evaluation + "gen_acc_and_jerk.xlsx")
        #delete old file
        if(isfile(self.path_evaluation + "/acceleration_and_jerk.xlsx")):
            os.remove(self.path_evaluation + "/acceleration_and_jerk.xlsx")


    def create_real_motion_measures(self):
        final_file = constants.evaluation_path + "/real_acc_and_jerk.xlsx"
        if not isfile(final_file):
            print("*"*10, "REAL MOTION DATA", "*"*10)
            motion_tab = {}
            for i, features in enumerate(motion_group_features.keys()):
                motion_tab[features] = {"velocity": [], "acceleration" : [], "jerk": []}
                for index, key in enumerate(self.y_final_keys):
                    
                    dim = motion_group_features[features][1]
                    gened_frames = self.getRealVideoData(index, motion_group_features[features][0])
            
                    motion_tab[features]["velocity"].append(self.compute_motion(gened_frames, derivative=1, dim=dim))
                    motion_tab[features]["acceleration"].append(self.compute_motion(gened_frames, derivative=2, dim=dim))
                    motion_tab[features]["jerk"].append(self.compute_motion(gened_frames, derivative=3, dim=dim))
                    
                motion_tab[features]["velocity"] = np.mean(motion_tab[features]["velocity"])
                motion_tab[features]["acceleration"] = np.mean(motion_tab[features]["acceleration"])
                motion_tab[features]["jerk"] = np.mean(motion_tab[features]["jerk"])
                
            df_acc = pd.DataFrame(motion_tab)
            df_acc['mean'] = df_acc.mean(axis=1)
            df_acc.to_excel(final_file)

    def create_curve(self):
        print("*"*10, "GENERAL CURVE", "*"*10)
        with PdfPages(self.path_evaluation + "curve.pdf") as pdf:
            #print(len(all_features))
            for feature in all_features : 
                #print("*"*5,feature, "*"*5)
                real_frames, gened_frames = self.getData(all_features, feature)
                self.plot_figure(real_frames, gened_frames, pdf, feature)
    
    def create_curves_video(self):
        print("*"*10,"VIDEO CURVE", "*"*10)
        for index, key in enumerate(self.y_final_keys):
            #print("*"*5, key, "*"*5)
            with PdfPages(self.path_evaluation + key + "_curve.pdf") as pdf:
                for feature in all_features :
                    #print("*"*2,feature, "*"*2)
                    real_frames, gened_frames = self.getVideoData(key, index, all_features, feature)
                    plot = self.plot_figure(real_frames, gened_frames, pdf, feature)


    def create_dtw(self):
        print("*"*10,"DTW", "*"*10)
        dist_tab = {}
        for index, key in enumerate(self.y_final_keys):
            #print("*"*5, key, "*"*5)
            dist_tab[key] = {}
            for feature in all_features:
                #print("*"*2,feature, "*"*2)
                real_frames, gened_frames = self.getVideoData(key, index, all_features, feature)
                distance = dtw.distance_fast(real_frames.values, gened_frames.values, use_pruning=True)
                dist_tab[key][feature] = distance

        df_global_dtw = pd.DataFrame(dist_tab)
        df_global_dtw['mean'] = df_global_dtw.mean(axis=1)
        mean_per_file = df_global_dtw.mean()
        mean_per_file.name = 'mean_per_file'
        df_global_dtw = df_global_dtw.append(mean_per_file)
        df_global_dtw.to_csv(self.path_evaluation + "global_dtw.csv", sep=";")


    def create_pca(self):
        print("*"*10,"PCA", "*"*10)
        with PdfPages(self.path_evaluation + "PCA.pdf") as pdf:
            for cle, features in types_output.items():
                real_frames, gened_frames = self.getData(all_features, features)
                #print("*"*5,cle, "*"*5)
                if(cle != "clignement"):
                    self.compute_pca(real_frames, gened_frames, pdf, cle)


    def compute_pca(self, real_frames, gened_frames, pdf, features_name = ""):
        #first scaling the data
        scaler = StandardScaler()
        scale_real = scaler.fit(real_frames)
        X_real = scale_real.transform(real_frames)
        X_gened = scale_real.transform(gened_frames)

        mypca = PCA(n_components=2, random_state=1) # calculate the three major components

        #pca in graphs
        pca_real = mypca.fit(X_real)
        data_real = pca_real.transform(X_real)
        data_generated = pca_real.transform(X_gened)
        col_list = ['principal component 1', 'principal component 2']
        df_real = pd.DataFrame(data = data_real, columns=col_list)
        df_generated = pd.DataFrame(data = data_generated, columns=col_list)
        indicesToKeep = df_generated.index

        plt.figure(figsize=(3, 3), dpi=100)
        plt.title('pca_'+features_name)
        plt.scatter(df_real.loc[indicesToKeep, 'principal component 1'], df_real.loc[indicesToKeep, 'principal component 2'], label='Real data', rasterized=True)
        plt.scatter(df_generated.loc[indicesToKeep, 'principal component 1'], df_generated.loc[indicesToKeep, 'principal component 2'], label='Generated data', alpha=0.7, rasterized=True)
        plt.xlabel('Principal Component - 1')
        plt.ylabel('Principal Component - 2')
        plt.legend()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
        

    def plot_figure(self, real_signal, generated_signal, pdf, features_name):
        x_real = range(len(real_signal))
        x_gen = range(len(generated_signal))
        plt.figure(figsize=(3, 3), dpi=100)
        plt.title(features_name)
        plt.plot(x_gen, generated_signal, label="generated", alpha=0.5, rasterized=True)
        plt.plot(x_real, real_signal, label="real", alpha=0.8, rasterized=True)
        plt.legend()
        pdf.savefig()
        plt.close()