
function  id = pcanumber(x,percentage)


[V,D] =  eig(full(x*x'));

D1 = sort(diag(D),'descend')./sum(diag(D));
varianceratio = cumsum(D1);


id = sum(varianceratio<=percentage);

if id<1
    id =1;
end



