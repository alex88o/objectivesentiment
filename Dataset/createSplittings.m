% Create random split from the dataset
clear all
clc

% SETTINGS
RND_DATA.shuffle_no = 3;
half_test_size = 962;


%Random splitting
test_set = [];
test_label = [];
train_set = [];
train_label = [];

load finalDataset;


% Step1: shuffle both vectors
rand_idx = randperm(length(DATA.neg_images));
shuff_neg = DATA.neg_images(rand_idx);

rand_idx = randperm(length(DATA.pos_images));
shuff_pos = DATA.pos_images(rand_idx);

% Step2: create the test set by taking half_test_size images from shuff_neg and shuff_pos
test_neg = shuff_neg(1:half_test_size);
test_pos = shuff_pos(1:half_test_size);

test_set = [test_neg test_pos];
% shuffling among positive and negative examples in the test set
rand_idx = randperm(length(test_set));
test_set = test_set(rand_idx);
test_label = zeros(size(test_set));

for i = 1:length(test_set)
    id = test_set(i);
    p = find(test_pos == id);
    if isempty(p)
       p = find(test_neg == id);
       if isempty(p)
           fprintf('Error!!');
           return;
       else
           test_label(i) = -1;
        %   fprintf(strcat('\n',num2str(id),'--->',num2str(test_neg(p)),'\tin\t',num2str(p)));
       end
    else
        test_label(i) = 1;
        %fprintf(strcat('\n',num2str(id),'--->',num2str(test_pos(p)),'\tin\t',num2str(p)));
    end
end
%find(test_label ==  0)


% Step3: create the training set (all the reminder elements from the
% negative array, the same amount from the positive array
train_neg  = setdiff(shuff_neg,test_neg); %consider the elements not included in the test set

shuff_pos = setdiff(shuff_pos,test_pos);  %consider the elements not included in the test set
train_pos = shuff_pos(1:length(train_neg)); 

train_set = [train_neg train_pos];
rand_idx = randperm(length(train_set));
train_set = train_set(rand_idx);

train_label = zeros(size(train_set));

for i = 1:length(train_set)
    id = train_set(i);
    p = find(train_pos == id);
    if isempty(p)
       p = find(train_neg == id);
       if isempty(p)
           fprintf('Error!!');
           return;
       else
           train_label(i) = -1;
        %   fprintf(strcat('\n',num2str(id),'--->',num2str(train_neg(p)),'\tin\t',num2str(p)));
       end
    else
        train_label(i) = 1;
        %fprintf(strcat('\n',num2str(id),'--->',num2str(train_pos(p)),'\tin\t',num2str(p)));
    end
end

RND_DATA.train_set = train_set;
RND_DATA.train_label = train_label;
RND_DATA.test_set  = test_set;
RND_DATA.test_label = test_label;

filename = strcat('shuffleDataset_',num2str(RND_DATA.shuffle_no));
save(filename,'RND_DATA');



