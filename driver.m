% Clearing Variables, Closing figures, Clear Screen
clear variables
close all
clc

% Loading data
load dataset2;

% Setting values of eta and the error below which the gradient should stop
% running.
eta=3;
epsilon=0.06;
cost=[];
x2=[];
n = length(X);
classification_error=[];

% Initializing theta as a random 3x1 matrix.
theta_old = rand(3,1);

% Running a single iteration of gradient descent to get a proper first
% theta value value for the while loop.
theta_new = theta_old - eta*grad(X,Y,theta_old);
count=1;

% Running gradient descent based till error is less than epsilon and
% parralely calculating logictic loss for each iteration. 
while norm(theta_new - theta_old) > epsilon
    theta_old=theta_new;
    theta_new = theta_old - eta * grad(X,Y,theta_old);
    cost=[cost;(1/n)*costFunction(X,Y,theta_old)];
    temp_class_error=0;
    % Calculating the Classification error
    for j=1:n
        temp_class_error=temp_class_error + abs(round(sigmoid(theta_new' * X(j,:)'))- Y(j,:));
    end
    classification_error =[classification_error;temp_class_error];
    count=count+1;
end

% Plotting the points on the graph based on Y
for k=1:n
    if Y(k,:)==1
        plot(X(k,1),X(k,2),'X');
        hold on
    else
        plot(X(k,1),X(k,2),'cO');
        hold on
    end
end
hold on

% Plotting the Decision boundary based on theta_new and using the whole
% dataset as testing data.
for i=1:n
   x2=[x2; (-theta_new(3,1) - theta_new(2,1)* X(i,2))/theta_new(1,1)]; 
end
plot(x2,X(:,2));
count

% Plotting Classification error
figure;
plot(classification_error);
figure;
final_classification_error=classification_error(end)
% Finding minimum cost and plotting the cost
min(cost);
final_cost=cost(end)
plot(cost);
