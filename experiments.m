clear all

%Variations we want to experiment with:

%sift_type: str, "gray", "RGB" or "opponent"
sift_type = ["gray"];
%sampling_mode: str, "dense", "key_points"
sampling_mode = ["dense"];
%vocab_size: int, number of image words in the vocabulary.
vocab_size = [100];
%train_subset: int or "all", define the size of the training set.
train_subset = ["all"];
%split_rate: float between 0 and 1. Defines the portion of the training
%data that is used for building the vocabulary.
split_rate = [0.5];
%feature_type: NOT IMPLEMENTED YET. "sift" or ___, choses the method 
%for feature extraction.
feature_type = ["sift"];
%clust_type: NOT IMPLEMENTED YET. "kmeans" or ___, choses the 
%clustering method for feature extraction.
clust_type = ["kmeans"];

%run all experiments in grid manner.
for st = 1:length(sift_type) 
    for sm = 1:length(sampling_mode)
        for vs = 1:length(vocab_size)
            for ts = 1:length(train_subset)
                for sr = 1:length(split_rate)
                    for ft = 1:length(feature_type)
                        for ct = 1:length(clust_type)
                            [MAP, average_precisions, label, score] = run_experiment(sift_type, sampling_mode, vocab_size, train_subset, split_rate, feature_type, clust_type);
                        end
                    end
                end
            end
        end
    end
end
                            
                            
                            
                            
                            