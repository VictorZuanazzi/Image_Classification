clear all

%Variations we want to experiment with:

%sift_type: str, "gray", "RGB" or "opponent"
sift_type = ["gray", "RGB", "opponent"];
%sampling_mode: str, "dense", "key_points"
sampling_mode = ["key point", "dense"];
%vocab_size: int, number of image words in the vocabulary.
vocab_size = [400, 1000, 4000];
%train_subset: int or "all", define the size of the training set.
train_subset = ["all"];
%split_rate: float between 0 and 1. Defines the portion of the training
%data that is used for building the vocabulary.
split_rate = [0.5];
%feature_type: "sift" or "liop", choses the method 
%for feature extraction.
feature_type = ["sift"];
%clust_type: NOT IMPLEMENTED YET. "kmeans" or ___, choses the 
%clustering method for feature extraction.
clust_type = ["kmeans"];

%load data
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

num_exp = length(sift_type)*length(sampling_mode)*length(vocab_size)*length(train_subset)*length(split_rate)*length(feature_type)*length(clust_type);
exp_bar = waitbar(0, sprintf('Will run %d experiments', num_exp), 'Name', 'All experiments');

%initialize variables for better runtime
MAP = cell(num_exp, 1);
average_precisions = cell(num_exp, 1);
label = cell(num_exp, 1);
score = cell(num_exp, 1);

%run all experiments in grid manner.
c = 1;
for st = 1:length(sift_type) 
    for sm = 1:length(sampling_mode)
        for vs = 1:length(vocab_size)
            for ts = 1:length(train_subset)
                for sr = 1:length(split_rate)
                    for ft = 1:length(feature_type)
                        for ct = 1:length(clust_type)
                            tic;
                            sprintf('%s %s %d', sift_type(st), sampling_mode(sm), vocab_size(vs))
                            waitbar((c/num_exp), exp_bar,  ...
                                sprintf('%s %s %d', sift_type(st), sampling_mode(sm), vocab_size(vs)));
                            
                            [MAP{c}, average_precisions{c}, label{c}, score{c}] = run_experiment(x_train, y_train, x_test, y_test, classes, class_name, sift_type(st), sampling_mode(sm), vocab_size(vs), train_subset(ts), split_rate(sr), feature_type(ft), clust_type(ct));
                            c = c +1;
                            toc
                        end
                    end
                end
            end
        end
    end
end

csvwrite("MAP.csv",MAP);
csvwrite("average_precisions.csv", average_precisions);
csvwrite("label.csv", label);
csvwrite("score.csv", score);

close(exp_bar);
                            
                            
                            
                            