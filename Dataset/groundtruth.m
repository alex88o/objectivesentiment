clear all
clc

% List of images by Katsurai
f = fopen('image_ids.txt');
txt = textscan(f,'%s','Delimiter',' ');

DS = txt{1};
DS = string(DS);

%Labels by Katsurai     (1=pos, 0=neut, -1=neg)
load Label.mat;

con = sqlite('../CrossSentiment2016/flickrCrossSentiment.db');
rows = fetch(con, 'SELECT FlickrId FROM Image');
rows = string(rows);

%All dataset data
dataset = [];
ground_truth = [];
%Positive and negative
pos_images = [];
neg_images = [];

% Consider only the images we have
cneut = 0;
j = 0;
cpos = 0;
cneg = 0;
% label = 0;
for i = 1:size(rows,1)
    id = rows(i);
    dataset(i) = id;
    idx = find(DS == id);
%     if Label(idx)>=0 && label<0
%         break;
%     end
    label = Label(idx);
%     if Label(idx)<0
%         fprintf(strcat('\n\n',id,'--->',DS(idx),'\tlabel\t',num2str(label)));
%         fprintf(strcat('\t',num2str(i),'--->',num2str(cneg)));
% 
%     end
    ground_truth(i) = label;
   % fprintf(strcat('\n',id,'--->',DS(idx),'\tlabel\t',num2str(label)));
    if label == 0
        %count the neutral images
        cneut = cneut +1;
    else
        if label == 1
            cpos = cpos + 1;
            pos_images(cpos) = id;
        else
            cneg = cneg + 1;
            neg_images(cneg) = id;     
        end
    end
end

DATA.dataset = dataset;                 %All FlickrID
DATA.ground_truth = ground_truth;       %All sentiment labels (+1 0 -1)
DATA.pos_images = pos_images;           %Only positive images
DATA.neg_images = neg_images;           %Only negative images

save('finalDataset','DATA');


