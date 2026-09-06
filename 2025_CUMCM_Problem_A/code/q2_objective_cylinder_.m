%自编辑的辅助函数，用于求圆出整体的遮蔽时长
function T_all=q2_objective_cylinder_(x,M1,v_dao,g,vd,R_yun, ...
    r_xy,r_R,r_z,n_r,nth,n_z,include_b)
    tau=x(4);
    td=x(3);
    phi=x(2);
    v=x(1);
    % 如果初始值不在约束范围内，则输出时间0s
    if v < 70 || v > 140 || td < 0 || tau < 0
        T_all=0; return;
    end
    te=td+tau;
    if te<=0
        T_all=0; return;
    end

    d_h=[cos(phi), sin(phi), 0];   % FY1的飞行方向
    FY1=[17800, 0, 1800];       %FY1的初始位置
    S=FY1+v*td*d_h;   %投掷点
    E=S+v*tau*d_h+0.5*[0,0,-g]*tau^2;   %起爆点

    % 保证在z>0的范围内爆炸，即落在地面前爆炸
    if E(3)<0
        T_all=0; 
        return;
    end

    %通过多组取样点，求出各点的有效时间集合，再求交集得出最终的时间
    [~, T_total]=cover_time_cylinder_sampling_( ...
        M1, v_dao, E, vd, te, R_yun, r_xy, r_R, r_z, ...
        n_r, nth, n_z, include_b);

    if ~isfinite(T_total), T_total=0; end
    T_all=T_total;
end
