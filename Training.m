function [acc] = Training(dataX,X_test,Y_test)
p = zeros(1,40);

for i = 1:40
    new_data = cell(1,10);
    j = 1;
    while j<=10
        cat = randsample(1:length(dataX{j}), i*50);
        X = dataX{j}(cat,:); 
        new_data{j} = X;
        j = j+1;
    end
    p(i) = Learn(new_data,X_test,Y_test);  
    
end
acc=p;


plot(v,acc);
end
        
function accuracy = Learn(train,test,label)
p = zeros(10,784);

for i = 1:10
    
    for j= 1:784
        p(i,j)=mean(train{i}(:,j));
    end
end
y=zeros(10000);
   
for i = 1:10000
    min = 100000000;
    level = -1;
    for j = 1:10
        sum = 0;
        for k = 1:784
            sum = sum + (p(j,k)-test(i,k))*(p(j,k)-test(i,k));  
      
        end
        distance = sqrt(sum);
        if(min > distance)
                  min = distance;
                  level = j-1;
        end
    end
    y(i) = level;
end

count = 0;
for i = 1:10000
    if(y(i) == label(i))
        count = count+1;
    end
end

accuracy = count/100;

end

function = Graph(acc)

v = zeros(40);
for i = 1:40
    v(i) = i*50;
end
plot(v, acc);
end



        
  
        
    