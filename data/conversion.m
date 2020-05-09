clear;
clc;
files = dir('*.jpg');
[l m] = size(files);
for(i=1:l)

        a = files(i).name;
        temp = imread(a);
        data = reshape(temp,1,2500);
        dataset(i , 1:2500) = data(1 , :); 
        q = char(a);
        if(q(2) == '(')
            dataset(i , 2501) = str2mat(q(1)) - 48;
        
        else if (q(3) == '(')
                dataset(i , 2501) = 10;
  
    
            end
        end
end

%csvwrite('dataset_beta.csv',dataset)