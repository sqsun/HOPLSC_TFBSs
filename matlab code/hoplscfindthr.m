function res = hoplscfindthr( yc, class )
% 在训练阶段，利用已有的类标class和训练模型产生的类标yc比较确定分类阈值
% find the class thresholds for PLSC
%

rsize = 100;
for g = 1:size(yc,2)
    class_in = ones(size(class,1),1);
    class_in( find( class ~= g ) ) = 2;
    count = 0;
    y_in = yc( :, g );
    miny = min( y_in );
    thr = max( y_in );
    step = ( thr - miny )/rsize;
    spsn = [];
    while thr > miny
        count = count + 1;
        class_calc_in = ones(size(class,1),1);
        thr = thr - step; % the step of move to find the optimization thr
        sample_out_g = find(y_in < thr);
        class_calc_in( sample_out_g ) = 2;
        cp = classifyperf( class_calc_in, class_in );
        sp( count, g ) = cp.specificity( 1 );
        sn( count, g ) = cp.sensitivity( 1 );
        thr_val( count, g ) = thr;       
    end
end

% find best thr based on bayesian discrimination threshold
for g = 1:max( class )
    P_g = yc( find( class == g ), g );
    P_notg = yc( find( class ~= g ), g );
    m_g = mean( P_g ); s_g = std( P_g );% mean variance for P_g
    m_notg = mean( P_notg ); s_notg = std( P_notg );
    stp = abs(m_g - m_notg)/1000;
    where = [m_notg:stp:m_g];
    % fit normal distribution
    % npdf_g = normpdf(where,m_g,s_g);
    x_g = (where - m_g) ./ s_g;% x_g是服从正态分布的数据
    npdf_g = exp(-0.5 * x_g .^2) ./ (sqrt(2*pi) .* s_g);% 属于该类正态分布函数
    %npdf_notg = normpdf(where,m_notg,s_notg);
    x_notg = (where - m_notg) ./ s_notg;
    npdf_notg = exp(-0.5 * x_notg .^2) ./ (sqrt(2*pi) .* s_notg);% 不属于该类的正态分布函数
    minval = NaN;
    for k=1:length(where)
        diff = abs(npdf_g(k)-npdf_notg(k));
        if isnan(minval)|diff < minval
             minval = diff;
             class_thr(g) = where(k);
        end
    end
    if isnan(minval)
        class_thr(g) = mean([m_g m_notg]);
    end  
end

res.class_thr = class_thr;
res.sp = sp;
res.sn = sn;
res.thr_val = thr_val;