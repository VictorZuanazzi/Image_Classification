
%path to the training and test data
train_path = "./Data/train.mat";
test_path = "./Data/test.mat";

%only keep the classes of interest
keep = [1, 2, 9, 7, 3];

%final dimensions of the images
im_dim = [96,96,3];

%magic function that loads the images and reshapes them
[x_train, y_train] = load_n_reshape(train_path, keep, im_dim);
[x_test, y_test] = load_n_reshape(test_path, keep, im_dim);

%is it a airplane? is it a bird? No, it is sup... it is a bird.
imshow(x_train{1})



