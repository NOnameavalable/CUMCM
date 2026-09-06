function tf = cover_one_at_time_(S, gi, t, P, M1, v_dao,uM, blow, vd, R_yun, safety_ep)
% 使用二分法，用于判断第 gi 枚弹在时刻 t 时是否被遮蔽
    tf = false;
    if t>=S.e(gi) && t<=S.e(gi)+20
        rM = M1 + v_dao*t*uM;
        c  = blow(gi,:) + [0,0,-vd*(t - S.e(gi))];
        tf = cloudFullyCovers_vec_(c, R_yun, rM, P, safety_ep);
    end
end