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
vocabulary_size = 400;

%magic function that loads the images and reshapes them
[x_train, y_train] = load_n_reshape(train_path, keep, im_dim);
[x_test, y_test] = load_n_reshape(test_path, keep, im_dim);

%is it a airplane? is it a bird? No, it is sup... it is a bird.
imshow(x_train{1})

% Split training data into a part for the vocabulary and a part for the SVM
[x_vocab, x_svm, ~, y_svm] = split_data(x_train, y_train, 0.5, keep);

% Create vocabulary
vocabulary = create_vocabulary(x_vocab, sampling_strategy, image_type, vocabulary_size);

% Create the BoW for all images
x_svm_BoW = BoW_representation_2(x_svm, sampling_strategy, image_type, vocabulary, false);
x_test_BoW = BoW_representation_2(x_test, sampling_strategy, image_type, vocabulary, false);

SVMModels = cell(5,1);
%creates binary label for the class of interest.

for i =1:length(keep)
    
    %binary label for the class
    y = y_svm == keep(i);
    
    %train the SVM
    SVMModels{i} = fitcsvm(x_svm_BoW, y', 'KernelFunction', 'rbf' );
    
    %get results of the model on the first 100 images of the test set
    [label{i}, score{i}] = predict(SVMModels{i}, x_test_BoW(1:100,:));
    
end

%make one prediction






