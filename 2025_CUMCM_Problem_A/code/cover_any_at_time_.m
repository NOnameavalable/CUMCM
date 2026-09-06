function tf=cover_any_at_time_(S,t,P,M1,v_dao,uM,blow,vd,R_yun,safety_ep)
% 该自定义辅助函数时用于判断任意时刻 t 是否满足“任一弹遮蔽”
    rM = M1 + v_dao*t*uM;
    tf = false;
    for i=1:3    %使用循坏实现
        if t>=S.e(i) && t<=S.e(i)+20
            c = blow(i,:) + [0,0,-vd*(t - S.e(i))];
            if cloudFullyCovers_vec(c, R_yun, rM, P, safety_ep)
                tf = true; return;
            end
        end
    end
end
