function covered=mask_one_cover_(S,gi,tGrid,P,M1,v_dao,uM,blow,vd,R_yun,safety_ep)
% 对第gi枚导弹，求出每个烟雾弹单独的遮蔽时间
    covered=false(size(tGrid));
    for k=1:numel(tGrid)
        t=tGrid(k);
        if t>=S.e(gi) && t<=S.e(gi)+20
            rM=M1+v_dao*t*uM;
            c =blow(gi,:)+[0,0,-vd*(t - S.e(gi))];
            covered(k)=cloudFullyCovers_vec_(c,R_yun,rM,P,safety_ep);
        end
    end
end
