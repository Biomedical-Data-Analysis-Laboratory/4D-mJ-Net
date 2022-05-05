from Model.constants import *
from Utils import general_utils, dataset_utils, model_utils

import os, glob, cv2, time
import numpy as np
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K


################################################################################
class DisplayCallback(callbacks.Callback):
    def __init__(self, ds_seq,text_fold_path):
        self.ds_seq = ds_seq
        self.text_fold_path = text_fold_path
        self.patients = ["01_031","02_036"]
        self.slice_idx = "03"

    def on_epoch_end(self, epoch, logs=None):
        for p in self.patients:
            dir_path = self.text_fold_path+p+os.path.sep
            general_utils.create_dir(dir_path)
            f = "/bhome/lucat/DATASET/SUS2020/COMBINED_Najm_v21-0.5/"+DATASET_PREFIX+p+general_utils.get_suffix()+".pkl"
            df = dataset_utils.read_pickle_or_hickle(f,flagHickle=False)
            x,y = 0,0
            s = time.time()
            X, coords = None, []
            img = np.empty((get_img_height(),get_img_height()))

            while True:
                coords.append((x,y))
                # if we reach the end of the image, break the while loop.
                if x>=get_img_width()-get_m() and y>=get_img_height()-get_n(): break
                if get_m()==get_img_width() and get_n()==get_img_height(): break  # check for M == WIDTH & N == HEIGHT
                if y<(get_img_height()-get_n()): y+=get_n()  # going to the next slicingWindow
                else:
                    if x<get_img_width():
                        y = 0
                        x += get_m()

            for i, coord in enumerate(coords):
                row = df[df.x_y == coord]
                row = row[row.sliceIndex == self.slice_idx]
                row = row[row.data_aug_idx == 0]
                assert len(row) == 1, "The length of the row to analyze should be 1."
                X = model_utils.get_correct_X_for_input_model(ds_seq=self.ds_seq, current_folder=row["pixels"].iloc[0],
                                                              row=row, batch_idx=i, batch_len=len(coords), X=X)
            tmp_img_arr = self.model.predict(X)
            general_utils.pickle_save(tmp_img_arr, "/bhome/lucat/tmp_img_arr.pkl")
            for i,tmp_img in enumerate(tmp_img_arr):
                x,y = coords[i]
                if is_TO_CATEG(): img[x:x+get_m(),y:y+get_n()] = (np.argmax(tmp_img,axis=-1) * 255) / (get_n_classes() - 1)
                else: img[x:x+get_m(),y:y+get_n()] = tmp_img*255

            cv2.imwrite(dir_path+self.slice_idx+"_"+str(epoch+1)+".png", img)
            print('\nSample Prediction ({0}s) after epoch {1}\n'.format(round(time.time()-s, 3), epoch + 1))


################################################################################
# Return information about the loss and accuracy
class CollectBatchStats(callbacks.Callback):
    def __init__(self, root_path, saved_mdl_name, txt_fold_path, acc="acc"):
        self.batch_losses = []
        self.batch_acc = []
        self.root_path = root_path
        self.saved_mdl_name = saved_mdl_name
        self.txt_fold_path = txt_fold_path
        self.model_name = self.saved_mdl_name[self.saved_mdl_name.rfind(os.path.sep):]
        self.acc = acc

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs[self.acc])

    # when an epoch is finished, it removes the old saved weights (from previous epochs)
    def on_epoch_end(self, epoch, logs=None):
        # save loss and accuracy in files
        textToSave = "epoch: {}".format(str(epoch))
        for k, v in logs.items():
            textToSave += ", {}: {}".format(k, round(v, 6))
        textToSave += '\n'
        with open(self.txt_fold_path + self.model_name + "_logs.txt", "a+") as loss_file:
            loss_file.write(textToSave)

        tmpSavedModels = glob.glob(self.saved_mdl_name + suffix_partial_weights + "*.h5")
        if len(tmpSavedModels) > 1:  # just to be sure and not delete everything
            for file in tmpSavedModels:
                if self.saved_mdl_name+ suffix_partial_weights in file:
                    tmpEpoch = general_utils.get_epoch_from_partial_weights_path(file)
                    if tmpEpoch < epoch: os.remove(file)  # Remove the old saved weights


################################################################################
# Save the best model every "period" number of epochs
def modelCheckpoint(filename, monitor, mode, period):
    return callbacks.ModelCheckpoint(
        filename + suffix_partial_weights + "{epoch:02d}.h5",
        monitor=monitor,
        verbose=is_verbose(),
        save_best_only=True,
        mode=mode,
        period=period
    )


################################################################################
# Stop the training if the "monitor" quantity does NOT change of a "min_delta"
# after a number of "patience" epochs
def earlyStopping(monitor, min_delta, patience):
    return callbacks.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=is_verbose(),
            mode="auto"
    )


################################################################################
# Reduce learning rate when a metric has stopped improving.
def reduceLROnPlateau(monitor, factor, patience, min_delta, cooldown, min_lr):
    return callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=is_verbose(),
            mode='auto',
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr
    )


################################################################################
# Reduce the learning rate every decay_step of a certain decay_rate
def LearningRateScheduler(decay_step, decay_rate):
    def lr_scheduler(epoch, lr):
        if epoch % decay_step == 0 and epoch: return lr * decay_rate
        return lr
    return callbacks.LearningRateScheduler(lr_scheduler, verbose=is_verbose())


################################################################################
# If you have installed TensorFlow with pip, you should be able to launch TensorBoard from the command line:
# tensorboard --logdir="/home/stud/lucat/PhD_Project/Stroke_segmentation/SAVE/"
def TensorBoard(log_dir, histogram_freq, update_freq):
    return callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=histogram_freq,
        update_freq=update_freq
    )


################################################################################
# Callback that terminates training when a NaN loss is encountered.
def TerminateOnNaN():
    return callbacks.TerminateOnNaN()


################################################################################
# Callback that streams epoch results to a CSV file.
def CSVLogger(textFolderPath, nn_id, filename, separator):
    return callbacks.CSVLogger(textFolderPath + nn_id + general_utils.get_suffix() + filename, separator=separator, append=True)
