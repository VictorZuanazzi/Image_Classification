function [MAP, average_precisions, label, score] = run_experiment(x_train, y_train, x_test, y_test, classes, class_name, sift_type, sampling_mode, vocab_size,train_subset, split_rate, feature_type, clust_type)
%runs all experiments for us.
%inputs:
    %sift_type: str, "gray", "RGB" or "opponent"
    %sampling_mode: str, "dense", "key_points"
    %vocab_size: int, number of image words in the vocabulary.
    %train_subset: int or "all", define the size of the training set.
    %split_rate: float between 0 and 1. Defines the portion of the training
    %data that is used for building the vocabulary.
    %feature_type:  "sift" or "liop, choses the method 
    %for feature extraction.
    %clust_type: "kmeans" or "kmedoids", choses the 
    %clustering method for feature extraction.
%outputs:
    %MAP: mean average precision of all classifiers.
    %average_precisions: average precision of each classifier.
    %label: predicted labels for the test set for each classifier.
    %score: certanty level of each classification of the test set for each
    %classifer.
feature_type = convertStringsToChars(feature_type);

%create progress bar
p_bar = waitbar(0, 'Initializing...', 'Name', 'Running experiment');
    
% Use a subset of the train data.
if isnumeric(train_subset)
    waitbar(.05, p_bar, sprintf('selecting training subset: %.2f', train_subset))
    rate = train_subset/length(y_train);
    [x_train, ~, y_train, ~] = split_data(x_train, y_train, rate, classes);
    train_subset = num2str(train_subset);
end

% Split training data into a part for the vocabulary and a part for the SVM
waitbar(.1, p_bar, sprintf('spliting data for training, split rate: %.2f', split_rate));
[x_vocab, x_svm, ~, y_svm] = split_data(x_train, y_train, split_rate, classes);

% Create vocabulary
waitbar(.2, p_bar, sprintf('creating vocabulary, sift type: %s vocab size: %.2f', sift_type, vocab_size));
sprintf("sampling mode: %s, sift type: %s, vocab size: %d, x vocab: %d %d", sampling_mode, sift_type, vocab_size, size(x_vocab))
vocabulary = create_vocabulary(x_vocab, sampling_mode, sift_type, vocab_size, feature_type, clust_type);

% Create the BoW for all images
waitbar(.25, p_bar, sprintf('BoW representation of training data, sift type: %s sampling mode: %.2f', sift_type, sampling_mode));
x_svm_BoW = BoW_representation_2(x_svm, sampling_mode, sift_type, feature_type, vocabulary, false);

waitbar(.3, p_bar, sprintf('BoW representation of test data, sift type: %s sampling mode: %.2f', sift_type, sampling_mode));
x_test_BoW = BoW_representation_2(x_test, sampling_mode, sift_type, feature_type, vocabulary, false);

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
    
    waitbar((.3+i/(length(classes)+5)), p_bar, sprintf('Training model: %s', class_name(i)));
    
    %binary label for the class
    y = y_svm == classes(i);
    
    %train the SVM
    SVMModels{i} = fitcsvm(x_svm_BoW, y, 'KernelFunction', 'rbf', 'Cost', [0,1;4,0]);
    
    %get results of the model on the test set
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
    figure(i)
    top_im = [sorted_imgs{i}{1}, sorted_imgs{i}{2}, sorted_imgs{i}{3}, sorted_imgs{i}{4}, sorted_imgs{i}{5}];
    imshow(top_im)
    name = path + "top5_class_" + num2str(class_name(i)) + "_"  + experiment_name + ".png";
    saveas(gcf, name);
    
    %display and save bottom 5 images.
    figure(i*2)
    bottom_im = [sorted_imgs{i}{end}, sorted_imgs{i}{end - 1}, sorted_imgs{i}{end - 2}, sorted_imgs{i}{end - 3}, sorted_imgs{i}{end - 4}];
    imshow(bottom_im)
    name = path + "bottom5_class_" + num2str(class_name(i))  + "_" + experiment_name + ".png";
    saveas(gcf, name);
end

%MAP over all classifiers
MAP = mean(average_precisions);

waitbar(1, p_bar, 'Experiment finished');

close(p_bar)
end