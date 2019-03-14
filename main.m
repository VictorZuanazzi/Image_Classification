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
vocabulary_size = 100;

%magic function that loads the images and reshapes them
[x_train, y_train] = load_n_reshape(train_path, keep, im_dim);
[x_test, y_test] = load_n_reshape(test_path, keep, im_dim);

%is it a airplane? is it a bird? No, it is sup... it is a bird.
imshow(x_train{1})

% Split training data into a part for the vocabulary and a part for the SVM
[x_vocab, x_svm, ~, y_svm] = split_data(x_train, y_train, 0.5, keep);

% Create vocabulary
vocabulary = create_vocabulary(x1, sampling_strategy, image_type, vocabulary_size);
   
