clear all;

%Experiments:
%obligatory -> find optimal parameters from that to run the other tests.
%different svm kernels (linear, rbf)
%ammount of training data
%change the ratio between vocab and train data
%test on rotated images
%change the clustering gaussian mixture models vs k-means
%different classification
tic;

%path to the training and test data
train_path = "./Data/train.mat";
test_path = "./Data/test.mat";

%only keep the classes of interest
keep = [1, 2, 9, 7, 3];

%final dimensions of the images
im_dim = [96,96,3];

% Set parameters
image_type = "gray";
sampling_strategy = "key point";
descriptor_type = 'sift';
cluster_type = "kmeans";
vocabulary_size = 400;

%magic function that loads the images and reshapes them
[x_train, y_train] = load_n_reshape(train_path, keep, im_dim);
[x_test, y_test] = load_n_reshape(test_path, keep, im_dim);

%is it a airplane? is it a bird? No, it is sup... it is a bird.
figure(75);
imshow(x_train{1})

% Split training data into a part for the vocabulary and a part for the SVM
[x_vocab, x_svm, ~, y_svm] = split_data(x_train, y_train, 0.5, keep);

% Create vocabulary
vocabulary = create_vocabulary(x_vocab, sampling_strategy, image_type, vocabulary_size, descriptor_type, cluster_type);

% Create the BoW for all images
x_svm_BoW = BoW_representation_2(x_svm, sampling_strategy, image_type, descriptor_type, vocabulary, false);
x_test_BoW = BoW_representation_2(x_test, sampling_strategy, image_type, descriptor_type, vocabulary, false);
%train one model for each class
SVMModels = cell(5,1);

for i =1:length(keep)
    
    %binary label for the class
    y = y_svm == keep(i);
    
    %train the SVM
    SVMModels{i} = fitcsvm(x_svm_BoW, y, 'KernelFunction', 'rbf', 'Cost', [0,1;4,0]);
    
    %get results of the model on the test set
    [label{i}, score{i}] = predict(SVMModels{i}, x_test_BoW);
    
    %sorts images and labels by their scores to this class
    [~, idx] = sort(score{i}(:,1));
    sorted_imgs{i} = x_test(idx);
    sorted_labels{i} = y_test(idx);
    
    %calculates Average Precision for the classifier
    binary_labels = sorted_labels{i} == keep(i);
    cumulative = cumsum(binary_labels);
    precisions = cumulative .* binary_labels ./ (1:length(sorted_labels{i}))';
    average_precisions(i) = sum(precisions)/sum(binary_labels);
    
    %display and save top 5 images
    figure(i)
    top_im = [sorted_imgs{i}{1}, sorted_imgs{i}{2}, sorted_imgs{i}{3}, sorted_imgs{i}{4}, sorted_imgs{i}{5}];
    imshow(top_im)
    path = "./Results/";
    name = path + "top5_class_" + num2str(keep(i)) + ".png";
    saveas(gcf, name);
    
    %display and save bottom 5 images.
    figure(i*2)
    bottom_im = [sorted_imgs{i}{end}, sorted_imgs{i}{end - 1}, sorted_imgs{i}{end - 2}, sorted_imgs{i}{end - 3}, sorted_imgs{i}{end - 4}];
    imshow(bottom_im)
    name = path + "bottom5_class_" + num2str(keep(i)) + ".png";
    export_fig(name);
end

%MAP over all classifiers
MAP = mean(average_precisions);

csvwrite("MAP.csv",MAP);
csvwrite("average_precisions.csv", average_precisions);
csvwrite("label.csv", label);
csvwrite("score.csv", score);

toc %output runtime