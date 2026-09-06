function coverTime=objective_cover_time_vec_(x,deco,M1,v_dao,uM,FY1,R_yun,vd,g,tGrid,P,safety_ep)
% 需要使用的粗评估函数
    S=deco(x);
    if S.d(2) < S.d(1)+1 || S.d(3) < S.d(2)+1
        coverTime=-1e6; return;
    end
    
    [~, blow]=drop_blows_from_S_(S, FY1, g, R_yun);% 引用其他的辅助函数，投掷点和起爆点
    % 单独烟雾的有效遮蔽的布尔序列
    covered = mask_any_cover_(S,tGrid,P,M1,v_dao,uM,blow,vd,R_yun,safety_ep);%再次引用其他辅助函数
    coverTime = sum(covered) * (tGrid(2)-tGrid(1));
end

