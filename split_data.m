function [x1, x2, y1, y2] = split_data(x, y, split_rate, classes)
%split data and labels into 2 sets. The returned vectors are sorted for the
%classes.
%input:
    %x: type cell containing one datapoint per cell.
    %y: array containing the labels
    %split_rate: number between 0 and 1 to indicate which percentage of the
    %data should go to x1 and y1. If 1 x1 and x2 get the sorted data from x
    %and the same for y.
    %classes: array containing the unique classes of y.
%output:
    %x1, x2: type cell with the data, length(x1) = split_rate*length(x).
    %y1, y2: array with the classes, length(y1) = split_rate*length(y)    
    
%number of classes
num_classes = length(classes);

%number of images per class
num_im = round(length(y)/num_classes);

%initialize counters
c1 = 1;
c2 = 1;

x1 = cell(1);
x2 = cell(1);

if split_rate == 1
    %if set to 1, all images are also used for x2 and y2
    sr_2 = 1;
else
    sr_2 = 1 - split_rate;
end

%split the images for each class
for i = 1:num_classes
    %selects the first split_rate*num_im images.
    idx_1 = find(y == classes(i), round(split_rate*num_im), "first"); 
    c_end = length(idx_1);
    %it is ugly and ineficient, but I could not find a better way of doing it
    for c = 1 : c_end 
        x1{c + c1 -1} = x{idx_1(c)};
    end
    y1(c1 : c1 + c_end -1) = y(idx_1);
    c1 = c_end + c1;
    
    %selects the last (1-split_rate)*num_im images.    
    idx_2 = find(y == classes(i), round(sr_2*num_im), "last");
    c_end = length(idx_2);
    for c = 1:c_end %ugly ugly ugly
        x2{c2 + c -1} = x{idx_2(c)};
    end
    y2(c2 : c2 + c_end -1) = y(idx_2); 
    c2 = c2 + c_end;

end