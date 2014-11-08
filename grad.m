function [avg] = grad(x,y,theta)
% x= dataset x
% y= dataset y
% theta= theta values
% avg= gradient of risk
n= length(x);
avg=zeros(3,1);
% Calculating the gradient using sigmoid function
% Gradient of Risk = (sum (g(theta' * x)) - y) /no. of records
for i=1:n
avg= avg + (sigmoid(theta' * x(i,:)')- y(i,:)) * x(i,:)';
end
avg=avg/n;
