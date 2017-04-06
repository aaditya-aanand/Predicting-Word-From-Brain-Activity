% You are only suppose to complete the "TODO" parts and the code should run
% if these parts are correctly implemented
clear all;close all;
%load('mnist_big.mat');
load('fmri_words.mat')
Data = X_train';
[U, D, V] = svd(Data, 0);
Factor = U(:, 1:250);
ND = Data'*Factor;
Y_test = Y_test(:, 1);
[N D] = size(ND);
[O D] = size(X_test);
%Y_train = Y_train + 1;
%Y_test = Y_test + 1;
W = randn(D,max(Y_train));
b = randn(1,max(Y_train));
W_avg = W;
b_avg = b;
k = 0; % number of mistakes (i.e., number of updates)

% Just do a single pass over the training data (our stopping condition will
% simply be this)
for n=1:N
    % TODO: predict the label for the n-th training example
    % as y_pred = argmax W'x_n + b
    A= zeros(1,60);
    for m=1:10
     A(m) = X_train(n,:)*W(:,m)+b(1,m);
    end
    [S K] = max (A);
    
     y_pred=K;
     
     %y_pred
    if (y_pred ~= Y_train(n) )
        k = k + 1;
        % TODO: Update W, b
        W(:,y_pred)=W(:,y_pred)-X_train(n,:)';
        W(:,Y_train(n))=W(:,Y_train(n))+X_train(n,:)';
        b(:,y_pred) = b(:,y_pred)-1;
        b(:,Y_train(n)) = b(:,Y_train(n))+1;
   
        % TODO: Update W_avg, b_avg using Ruppert Polyak Averaging
        % Important: You don't need to store all the previous W's and b's
        % to compute W_avg, b_avg
        W_avg= W_avg-(1/k)*(W_avg-W) ;
        b_avg = b_avg - (1/k)*(b_avg-b);    

 % TODO: Predict test labels using W, b
       % for m=1:10
     y_test_pred = zeros(O,1);
      A= zeros(1,60);
    for i= 1:O
        for m=1:60
          A(m) = X_test(i,:)*W(:,m)+b(:,m);
        end
     [M K] = max(A);
     y_test_pred(i,1)=K;
        end
     
    
        
        acc(k) = mean(Y_test==y_test_pred);   % test accuracy     
        
        % TODO: Now predict test labels using W_avg, b_avg
           y_test_pred = zeros(O,1);
      A= zeros(1,60);
    for i= 1:O
        for m=1:60
          A(m) = X_test(i,:)*W_avg(:,m)+b_avg(:,m);
        end
     [M K] = max(A);
     y_test_pred(i,1)=K;
        end
        acc_avg(k) = mean(Y_test==y_test_pred); % test accuracy with R-P averaging
        
        fprintf('Update number %d, accuracy = %f, accuracy (with R-P averaging) = %f\n',k,acc(k),acc_avg(k));
        plot(1:k,acc(1:k),'r');
        hold on;
        plot(1:k,acc_avg(1:k),'g');
        drawnow;        
               
    end
end