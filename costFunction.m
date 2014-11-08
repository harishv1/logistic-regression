function cost = costFunction(x,y,theta)
% x= dataset x
% y= dataset y
% theta= theta values
% cost= Logictic loss
n= length(x);
cost=0;
% Calculating Logistc loss as per the equation giving in the problem
for j=1:n
cost =cost + (y(j,:)-1).*log(1-sigmoid(theta' * x(j,:)'))-(y(j,:).*log(sigmoid(theta' * x(j,:)')));
end
