function tf=cloudFullyCovers_vec_(c, R_yun,rM,P,safety_ep)
% 计算点 c 与所有线段的最近距离；如果全部≤有效半径则完全遮蔽
    R_e= R_yun - safety_ep;
    AB = P - rM;                                
    AB_2= sum(AB.*AB, 2);                       
    CA = c - rM;                         
    t  = zeros(size(AB_2));
    nz = AB_2> 1e-12;  %判断是否大于0
    t(nz) = (AB(nz,:)*CA.') ./ AB_2(nz);
    t = max(0, min(1, t));                      % 最近点在线段内的参数
    Q = rM + t.*AB;                             % 最近点
    d = sqrt(sum((c - Q).^2, 2));               % 与云团中心距离
    tf = all(d <= R_e+ 1e-12);
end
