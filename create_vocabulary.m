function [vocabulary] = create_vocabulary(data, sampling_strategy, image_type, vocabulary_size)
% Create a vocabulary from the given training data
descriptors = [];
for i = 1:length(data)
    d = feature_extraction(data{i}, sampling_strategy, image_type);
    descriptors = [descriptors; d];
end
[~, vocabulary] = kmeans(double(descriptors), vocabulary_size);
end

