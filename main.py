import argparse
import cv2
from sklearn import svm
from sklearn import metrics
from src.data_load import load_images, binarize_labels, load_feature_maps
from src.data_preprocessing import batch_data_preprocessing
from src.evaluation import model_predictions, printWrongPredictions
from src.feature_maps import corf_feature_maps, temp_feature_maps, feature_fusion, \
    feature_stack, feature_normalization
from src.models import Models
from src.train import train_test_split, train_model, five_cross_validation, leave_one_day_out
from src.visualizations import plot_data_graph
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.engine import Model
import os
import numpy as np

if __name__ == '__main__':

    # For boolean input from the command line
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessing', type=str2bool, default=False,
                        help='Whether to use the preprocessing step')
    parser.add_argument('--method', type=str, choices=["5_fold", "leave_one_day_out"],
                        default="5_fold", help='Select "5_fold", or "leave_one_day_out" method')
    parser.add_argument('--mode', type=str, choices=["single", "fusion"],
                        default="fusion",
                        help='Select "single", or "fusion" mode')
    parser.add_argument('--feature_map_1', type=str, choices=["RGB", "MSX"],
                        required=False, default="",
                        help='Which Feature map 1 you are using.')
    parser.add_argument('--feature_map_2', type=str, choices=["CORF3D", "TEMP3D", "MSX"],
                        default="", required=False,
                        help='Which Feature map 2 you are using.')
    parser.add_argument('--dataset_1', type=str, default="",
                        required=False,
                        help='Dataset 1 you are using.')
    parser.add_argument('--dataset_2', type=str,
                        default="",
                        required=False,
                        help='Dataset 2 you are using.')
    parser.add_argument('--preprocessed_dataset', type=str,
                        default="./data/preprocessed",
                        required=False,
                        help='Dataset 2 you are using.')
    parser.add_argument('--timestamp', type=str, default="./data/timestamp.xlsx",
                        help='Timestamp file')
    parser.add_argument('--resize', type=int, default=224,
                        help='Height of cropped input image to network')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of images in each batch')
    parser.add_argument('--classes', type=int, default=383,
                        help='Number of images in each batch')
    parser.add_argument('--trainable', type=str2bool, default=True,
                        help='Whether to train all the layers')
    parser.add_argument('--pretrained_dataset', type=str, default='imagenet',
                        help='specify which pre-trained dataset to use')
    parser.add_argument('--include_top', type=str2bool, default=False,
                        help='specify if to use the top layer of pre-trained model')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate for the optimizer')
    parser.add_argument('--model', type=str, default="densenet121",
                        help='Your pre-trained classification model of choice')
    args = parser.parse_args()

    np.random.seed(7)
    if args.preprocessing:

        if args.feature_map_1 == "RGB" and args.feature_map_2 == "CORF3D" and args.mode == "fusion":

            dataset_1, labels, labels_list = batch_data_preprocessing(args.dataset_1,
                                                                      args.dataset_2,
                                                                      args.preprocessed_dataset,
                                                                      args.feature_map_1)
            corf_feature_set_1 = corf_feature_maps(args.preprocessed_dataset, 2.2, 4, 0.0, 0.005)
            corf_feature_set_norm_1 = feature_normalization(corf_feature_set_1)
            corf_feature_set_2 = corf_feature_maps(args.preprocessed_dataset, 2.2, 4, 1.8, 0.005)
            corf_feature_set_norm_2 = feature_normalization(corf_feature_set_2)
            corf_feature_set_3 = corf_feature_maps(args.preprocessed_dataset, 2.2, 4, 3.6, 0.005)
            corf_feature_set_norm_3 = feature_normalization(corf_feature_set_3)
            dataset_2 = feature_stack(corf_feature_set_norm_1, corf_feature_set_norm_2,
                                      corf_feature_set_norm_3)

        elif args.feature_map_1 == "RGB" and args.feature_map_2 == "TEMP3D" and args.mode == "fusion":

            dataset_1, labels, labels_list = batch_data_preprocessing(args.dataset_1,
                                                                      args.dataset_2,
                                                                      args.preprocessed_dataset,
                                                                      args.feature_map_1)
            temp_feature_set_1 = temp_feature_maps(args.dataset_2)
            temp_feature_set_norm_1 = feature_normalization(temp_feature_set_1)
            dataset_2 = feature_stack(temp_feature_set_norm_1, temp_feature_set_norm_1,
                                      temp_feature_set_norm_1)

        elif args.feature_map_1 == "RGB" and args.feature_map_2 == "MSX" and args.mode == "fusion":

            dataset_1, labels, labels_list = batch_data_preprocessing(args.dataset_1,
                                                                      args.dataset_2,
                                                                      args.preprocessed_dataset,
                                                                      args.feature_map_1)
            dataset_2, labels, labels_list = batch_data_preprocessing(args.dataset_1,
                                                                      args.dataset_2,
                                                                      args.preprocessed_dataset,
                                                                      args.feature_map_2)

        elif args.feature_map_1 == "RGB" or args.feature_map_1 == "MSX" and args.mode == "single":

            dataset_1, labels, labels_list = batch_data_preprocessing(args.dataset_1,
                                                                      args.dataset_2,
                                                                      args.preprocessed_dataset,
                                                                      args.feature_map_1)

        elif args.feature_map_2 == "TEMP3D" and args.mode == "single":

            temp_feature_set_1, labels_list, labels = temp_feature_maps(args.dataset_2)
            temp_feature_set_norm_1 = feature_normalization(temp_feature_set_1)
            dataset_1 = feature_stack(temp_feature_set_norm_1, temp_feature_set_norm_1,
                                      temp_feature_set_norm_1)

        elif args.feature_map_2 == "CORF3D" and args.mode == "single":

            corf_feature_set_1 = corf_feature_maps(args.preprocessed_dataset, 2.2, 4, 0, 0.005)
            corf_feature_set_norm_1 = feature_normalization(corf_feature_set_1)
            corf_feature_set_2 = corf_feature_maps(args.preprocessed_dataset, 2.2, 4, 1.8, 0.005)
            corf_feature_set_norm_2 = feature_normalization(corf_feature_set_2)
            corf_feature_set_3 = corf_feature_maps(args.preprocessed_dataset, 2.2, 4, 3.6, 0.005)
            corf_feature_set_norm_3 = feature_normalization(corf_feature_set_3)
            dataset_1 = feature_stack(corf_feature_set_norm_1, corf_feature_set_norm_2,
                                             corf_feature_set_norm_3)

        binarizelabels = binarize_labels(labels)

    else:
        print("No Preprocessing")

        if args.feature_map_1 == "RGB" and args.feature_map_2 == "CORF3D" or args.feature_map_2 == "TEMP3D" and \
                args.mode == "fusion":

            dataset_1, labels, labels_list = load_images(args.dataset_1, 224)
            dataset_2, labels, labels_list = load_feature_maps(args.dataset_2, 224)

        elif args.feature_map_1 == "RGB" or args.feature_map_1 == "MSX" and args.mode == "single":

            dataset_1, labels, labels_list = load_images(args.dataset_1, 224)

        elif args.feature_map_2 == "CORF3D" or args.feature_map_2 == "TEMP3D" and args.mode == "single":

            dataset_1, labels, labels_list = load_feature_maps(args.dataset_2, 224)

        elif args.feature_map_1 == "RGB" and args.feature_map_2 == "MSX" and args.mode == "fusion":

            dataset_1, labels, labels_list = load_images(args.dataset_1, 224)
            dataset_2, labels, labels_list = load_images(args.dataset_2, 224)

        binarizelabels = binarize_labels(labels)

    skf = train_test_split(5)
    all_fold_accuracy = []
    all_fold_loss = []
    all_fold_accuracy_1 = []
    all_fold_loss_1 = []
    all_fold_accuracy_2 = []
    all_fold_loss_2 = []
    all_fold_svm_accuracy = []
    counter = 1
    model = Models(args.trainable, args.classes, args.pretrained_dataset, args.include_top,
                   args.learning_rate)

    if args.model == 'densenet121':
        model.densenet121()
    elif args.model == 'mobilenet':
        model.mobilenet()
    elif args.model == 'xception':
        model.xception()

    model.model_trainable()
    compiled_model = model.model_compile(args.learning_rate)

    if args.method == "5_fold":

        X_train_index, X_test_index, y_train_index, y_test_index = five_cross_validation(dataset_1,
                                                                                         binarizelabels,
                                                                                         skf)

        if args.mode == "single":

            for i in range(0, len(X_train_index)):
                X_train, X_test = dataset_1[X_train_index[i]], dataset_1[X_test_index[i]]
                y_train, y_test = binarizelabels[y_train_index[i]], binarizelabels[y_test_index[i]]

                model, hist, loss, accuracy = train_model(compiled_model, X_train,
                                                          y_train, args.batch_size,
                                                          args.num_epochs, X_test, y_test,
                                                          args.model, counter, args.feature_map_1)

                plot_data_graph(hist, args.num_epochs, counter, args.model, args.feature_map_1)
                counter = counter + 1
                all_fold_accuracy.append(accuracy * 100)
                all_fold_loss.append(loss)

            print("Average accuracy of Model", np.mean(all_fold_accuracy))
            print("Std accuracy of Model", np.std(all_fold_accuracy))

        if args.mode == "fusion":

            for i in range(0, len(X_train_index)):
                X_train_1, X_test_1 = dataset_1[X_train_index[i]], dataset_1[X_test_index[i]]
                y_train_1, y_test_1 = binarizelabels[y_train_index[i]], binarizelabels[
                    y_test_index[i]]

                X_train_2, X_test_2 = dataset_2[X_train_index[i]], dataset_2[X_test_index[i]]
                y_train_2, y_test_2 = binarizelabels[y_train_index[i]], binarizelabels[
                    y_test_index[i]]

                model_1, hist_1, loss_1, accuracy_1 = train_model(compiled_model, X_train_1,
                                                                  y_train_1, args.batch_size,
                                                                  args.num_epochs, X_test_1,
                                                                  y_test_1, args.model, counter,
                                                                  args.feature_map_1)

                model_2, hist_2, loss_2, accuracy_2 = train_model(compiled_model, X_train_2,
                                                                  y_train_2, args.batch_size,
                                                                  args.num_epochs, X_test_2,
                                                                  y_test_2, args.model, counter,
                                                                  args.feature_map_2)

                plot_data_graph(hist_1, args.num_epochs, counter, args.model, args.feature_map_1)
                all_fold_accuracy_1.append(accuracy_1 * 100)
                all_fold_loss_1.append(loss_1)

                plot_data_graph(hist_2, args.num_epochs, counter, args.model, args.feature_map_2)
                all_fold_accuracy_2.append(accuracy_2 * 100)
                all_fold_loss_2.append(loss_2)

                # Convert to single column
                y_train = np.argmax(y_train_1, axis=1)
                y_test = np.argmax(y_test_1, axis=1)

                # Load models
                filepath_1 = "./models/" + str(args.model) + "_" + str(counter) + "_" + \
                             str(args.feature_map_1) + "_model.h5"
                filepath_2 = "./models/" + str(args.model) + "_" + str(counter) + "_" + \
                             str(args.feature_map_2) + "_model.h5"

                trained_model_1 = load_model(filepath_1)
                trained_model_2 = load_model(filepath_2)

                # Extract features
                model_1_feature_map = Model(trained_model_1.input,
                                            trained_model_1.layers[-2].output)
                model_2_feature_map = Model(trained_model_2.input,
                                            trained_model_2.layers[-2].output)

                features_train_1 = model_1_feature_map.predict(X_train_1)
                features_test_1 = model_1_feature_map.predict(X_test_1)

                features_train_2 = model_2_feature_map.predict(X_train_2)
                features_test_2 = model_2_feature_map.predict(X_test_2)

                combined_feature_train = feature_fusion(features_train_1, features_train_2)
                combined_feature_test = feature_fusion(features_test_1, features_test_2)

                clf = svm.SVC(kernel='linear')

                clf.fit(combined_feature_train, y_train)

                # Test the model
                y_pred = clf.predict(combined_feature_test)

                svm_accuracy = metrics.accuracy_score(y_test, y_pred)

                print("Model + SVM accuracy:", svm_accuracy)

                all_fold_svm_accuracy.append(svm_accuracy * 100)

                counter = counter + 1

            print("Average accuracy of Model + SVM:", np.mean(all_fold_svm_accuracy))
            print("Std accuracy of Model + SVM:", np.std(all_fold_svm_accuracy))

    elif args.method == "leave_one_day_out":

        X_train_index, X_test_index, y_train_index, y_test_index = leave_one_day_out(
            args.timestamp,
            dataset_1, binarizelabels)

        if args.mode == "single":

            for i in range(0, len(X_train_index)):
                X_train, X_test = dataset_1[X_train_index[i]], dataset_1[X_test_index[i]]
                y_train, y_test = binarizelabels[y_train_index[i]], binarizelabels[y_test_index[i]]

                model, hist, loss, accuracy = train_model(compiled_model, X_train,
                                                          y_train, args.batch_size,
                                                          args.num_epochs, X_test, y_test,
                                                          args.model, counter, args.feature_map_1)

                plot_data_graph(hist, args.num_epochs, counter, args.model, args.feature_map_1)
                counter = counter + 1
                all_fold_accuracy.append(accuracy * 100)
                all_fold_loss.append(loss)

            print("Average accuracy of Model", np.mean(all_fold_accuracy))
            print("Std accuracy of Model", np.std(all_fold_accuracy))

        if args.mode == "fusion":

            for i in range(0, len(X_train_index)):
                X_train_1, X_test_1 = dataset_1[X_train_index[i]], dataset_1[X_test_index[i]]
                y_train_1, y_test_1 = binarizelabels[y_train_index[i]], binarizelabels[
                    y_test_index[i]]

                X_train_2, X_test_2 = dataset_2[X_train_index[i]], dataset_2[X_test_index[i]]
                y_train_2, y_test_2 = binarizelabels[y_train_index[i]], binarizelabels[
                    y_test_index[i]]

                model_1, hist_1, loss_1, accuracy_1 = train_model(compiled_model, X_train_1,
                                                                  y_train_1, args.batch_size,
                                                                  args.num_epochs, X_test_1,
                                                                  y_test_1, args.model, counter,
                                                                  args.feature_map_1)

                model_2, hist_2, loss_2, accuracy_2 = train_model(compiled_model, X_train_2,
                                                                  y_train_2, args.batch_size,
                                                                  args.num_epochs, X_test_2,
                                                                  y_test_2, args.model, counter,
                                                                  args.feature_map_2)

                plot_data_graph(hist_1, args.num_epochs, counter, args.model, args.feature_map_1)
                all_fold_accuracy_1.append(accuracy_1 * 100)
                all_fold_loss_1.append(loss_1)

                plot_data_graph(hist_2, args.num_epochs, counter, args.model, args.feature_map_2)
                all_fold_accuracy_2.append(accuracy_2 * 100)
                all_fold_loss_2.append(loss_2)

                # Convert to single column
                y_train = np.argmax(y_train_1, axis=1)
                y_test = np.argmax(y_test_1, axis=1)

                # Load models
                filepath_1 = "./models/" + str(args.model) + "_" + str(counter) + "_" + \
                             str(args.feature_map_1) + "_model.h5"
                filepath_2 = "./models/" + str(args.model) + "_" + str(counter) + "_" + \
                             str(args.feature_map_2) + "_model.h5"

                trained_model_1 = load_model(filepath_1)
                trained_model_2 = load_model(filepath_2)

                # Extract features
                model_1_feature_map = Model(trained_model_1.input,
                                            trained_model_1.layers[-2].output)
                model_2_feature_map = Model(trained_model_2.input,
                                            trained_model_2.layers[-2].output)

                features_train_1 = model_1_feature_map.predict(X_train_1)
                features_test_1 = model_1_feature_map.predict(X_test_1)

                features_train_2 = model_2.predict(X_train_2)
                features_test_2 = model_2.predict(X_test_2)

                combined_feature_train = feature_fusion(features_train_1, features_train_2)
                combined_feature_test = feature_fusion(features_test_1, features_test_2)

                clf = svm.SVC(kernel='linear')

                clf.fit(combined_feature_train, y_train)

                # Test the model
                y_pred = clf.predict(combined_feature_test)

                svm_accuracy = metrics.accuracy_score(y_test, y_pred)

                print("Model + SVM accuracy:", svm_accuracy)

                all_fold_svm_accuracy.append(svm_accuracy * 100)

                counter = counter + 1

            print("Average accuracy of Model + SVM:", np.mean(all_fold_svm_accuracy))
            print("Std accuracy of Model + SVM:", np.std(all_fold_svm_accuracy))
