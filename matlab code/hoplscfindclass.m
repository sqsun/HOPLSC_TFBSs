function assigned_class = hoplscfindclass(yc,class_thr)

% assign samples for PLSC on the basis of thresholds and calculated responses
%


nobj = size(yc,1);
nclass = size(yc,2);
for i = 1:nobj
    pred = yc(i,:);
    chk_ass = zeros(1,nclass);
    for c = 1:nclass
        if pred(c) > class_thr(c); 
            chk_ass(c) = 1; 
        end;
    end
    if length(find(chk_ass)) == 1
        assigned_class(i) = find(chk_ass);
    else
        assigned_class(i) = 0;
    end
end