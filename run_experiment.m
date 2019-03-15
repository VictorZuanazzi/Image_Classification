function [MAP, average_precisions, label, score] = run_experiment(sift_type, sampling_mode, vocab_size,train_subset, split_rate, feature_type, clust_type)
%runs all experiments for us.
%inputs:
    %sift_type: str, "gray", "RGB" or "opponent"
    %sampling_mode: str, "dense", "key_points"
    %vocab_size: int, number of image words in the vocabulary.
    %train_subset: int or "all", define the size of the training set.
    %split_rate: float between 0 and 1. Defines the portion of the training
    %data that is used for building the vocabulary.
    %feature_type: NOT IMPLEMENTED YET. "sift" or ___, choses the method 
    %for feature extraction.
    %clust_type: NOT IMPLEMENTED YET. "kmeans" or ___, choses the 
    %clustering method for feature extraction.
%outputs:
    %MAP: mean average precision of all classifiers.
    %average_precisions: average precision of each classifier.
    %label: predicted labels for the test set for each classifier.
    %score: certanty level of each classification of the test set for each
    %classifer.

%Experiments:
%obligatory -> find optimal parameters from that to run the other tests.
%different svm kernels (linear, rbf)
%ammount of training data
%change the ratio between vocab and train data
%change the clustering gaussian mixture models vs k-means
%different classification

%path to the training and test data
train_path = "./Data/train.mat";
test_path = "./Data/test.mat";

%only keep the classes of interest
class_name = ["airplane", "bird", "ship", "horse", "car"];
classes = [1, 2, 9, 7, 3];

%final dimensions of the images
im_dim = [96,96,3];

%magic function that loads the images and reshapes them
[x_train, y_train] = load_n_reshape(train_path, classes, im_dim);
[x_test, y_test] = load_n_reshape(test_path, classes, im_dim);

% Use a subset of the train data.
if train_subset ~= "all"
    rate = train_subset/length(y_train);
    [x_train, ~, y_train, ~] = split_data(x_train, y_train, rate, classes);
    train_subset = numwstr(train_subset);
end

% Split training data into a part for the vocabulary and a part for the SVM
[x_vocab, x_svm, ~, y_svm] = split_data(x_train, y_train, split_rate, classes);

% Create vocabulary
vocabulary = create_vocabulary(x_vocab, sampling_mode, sift_type, vocab_size);

% Create the BoW for all images
x_svm_BoW = BoW_representation_2(x_svm, sampling_mode, sift_type, vocabulary, false);
x_test_BoW = BoW_representation_2(x_test, sampling_mode, sift_type, vocabulary, false);

% Strings necessary to save the images with unique names.
path = "./Results/";
experiment_name = sift_type + "_" + sampling_mode + "_" + num2str(vocab_size) + "_" + train_subset + "_" + num2str(split_rate) + "_" + feature_type + "_" + clust_type;
    
%initialize variables for lower run time.
SVMModels = cell(1,5);
label = cell(1,5);
score = cell(1,5);
sorted_imgs = cell(1,5);
sorted_labels = cell(1,5);
average_precisions = zeros(1,5);

%train one model for each class
for i =1:length(classes)
    
    %binary label for the class
    y = y_svm == classes(i);
    
    %train the SVM
    SVMModels{i} = fitcsvm(x_svm_BoW, y, 'KernelFunction', 'rbf', 'Cost', [0,1;4,0]);
    
    %get results of the model on the first 100 images of the test set
    [label{i}, score{i}] = predict(SVMModels{i}, x_test_BoW);
    
    %sorts images and labels by their scores to this class
    [~, idx] = sort(score{i}(:,1));
    sorted_imgs{i} = x_test(idx);
    sorted_labels{i} = y_test(idx);
    
    %calculates Average Precision for the classifier
    binary_labels = sorted_labels{i} == classes(i);
    cumulative = cumsum(binary_labels);
    precisions = cumulative .* binary_labels ./ (1:length(sorted_labels{i}))';
    average_precisions(i) = sum(precisions)/sum(binary_labels);
    
    %display and save top 5 images
    image(i)
    top_im = [sorted_imgs{i}{1}, sorted_imgs{i}{2}, sorted_imgs{i}{3}, sorted_imgs{i}{4}, sorted_imgs{i}{5}];
    imshow(top_im)
    name = path + "top5_class_" + num2str(classe_name(i)) + experiment_name + ".png";
    saveas(gcf, name);
    
    %display and save bottom 5 images.
    image(i*2)
    bottom_im = [sorted_imgs{i}{end}, sorted_imgs{i}{end - 1}, sorted_imgs{i}{end - 2}, sorted_imgs{i}{end - 3}, sorted_imgs{i}{end - 4}];
    imshow(bottom_im)
    name = path + "bottom5_class_" + num2str(classe_name(i)) + experiment_name + ".png";
    saveas(gcf, name);
end

%MAP over all classifiers
MAP = mean(average_precisions);


end