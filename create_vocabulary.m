function [vocabulary] = create_vocabulary(data, sampling_strategy, image_type, vocabulary_size)
% Create a vocabulary from the given training data
descriptors = [];
for i = 1:length(data)
    d = feature_extraction(data{i}, sampling_strategy, image_type);
    descriptors = [descriptors; d];
end

%k means
stream = RandStream('mlfg6331_64');  % Random number stream
options = statset('UseParallel',1,'UseSubstreams',1, 'Streams',stream);
[~, vocabulary] = kmeans(double(descriptors), vocabulary_size, ...
    'Options', options, 'MaxIter', 100, 'Display', 'final');
end

