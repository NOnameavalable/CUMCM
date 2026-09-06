function [interval, T_total, info] = cover_time_cylinder_sampling_( ...
    M1,v_dao,E,vd,te,R_yun,r_xy,r_R,r_z, ...
    n_r,nth,n_z,include_b)
    %在圆柱体内生成若干采样点
    c_x=r_xy(1);
    c_y=r_xy(2);
    z1=r_z(1);
    z2=r_z(2);

    %在半径方向上均匀采点，主函数中说明include_b为false，即不取边界
    if include_b
        z_l=linspace(z1, z2, n_z);
        r_l=linspace(0, r_R, n_r);
    else
        u=linspace(1/(n_r+1), n_r/(n_r+1),n_r);
        r_l=r_R*sqrt(u);
        if n_z==1
            z_l=(z1+z2)/2;
        else
            z_l=linspace(z1,z2,n_z+2); 
            z_l=z_l(2:end-1);           % 去除端点
        end
    end
    the_l=linspace(0,2*pi,nth+1);
    the_l(end)=[]; %去除里面重复的值
    T_p=[];%采样点的集合，将去重后的采样点放入
    for iz=1:numel(z_l)
        z=z_l(iz);
        for ir=1:numel(r_l)
            r=r_l(ir);
            if r==0
                T_p(end+1,:)=[c_x,c_y,z];
            else
                for it=1:numel(the_l)
                    th=the_l(it);
                    x=c_x + r*cos(th);
                    y=c_y + r*sin(th);
                    T_p(end+1,:)=[x,y,z];
                end
            end
        end
    end

    % —— 2) 逐点求遮蔽区间并不断取“交集” —— 
    % 初始交集为整个有效窗
    interval=[te, te+20];
    for p=1:size(T_p,1)
        T=T_p(p,:);
        [interval_p,~]=cover_time_point_target_(T,M1,v_dao, E,vd,te,R_yun);
        interval=intersect_interval_sets_(interval,interval_p);
        if isempty(interval)   %如果是空集了，直接跳出循坏
            break;  
        end
    end
    %输出总的时间
    if isempty(interval) 
        T_total=0;
    else
        T_total=sum(interval(:,2) - interval(:,1));
    end

    %将提取采样的信息存入info中
    info.the_l=the_l;
    info.n_p=size(T_p,1);
    info.z_l=z_l;
    info.include_b=include_b;
    info.r_l=r_l;
end


%自主定义辅助函数
function C=intersect_interval_sets_(a, b)
% a, b: [Na,2], [Nb,2]，各自可能是多段互不重叠、已排序的区间集合
% 返回 C = a ∩ b（同样按起点升序，自动合并相邻/重叠的结果）
    tol_=1e-12;
    if isempty(a) || isempty(b)
        C=[];
        return;
    end
    % 确保有序
    a=sortrows(a,1);
    b=sortrows(b,1);
    ia=1;
    ib=1;
    C=zeros(0,2);
    while ia<=size(a,1) && ib<=size(b,1)
        a1=a(ia,1);
        a2=a(ia,2);
        b1=b(ib,1);
        b2=b(ib,2);
        %用两个集合求交集
        s1=max(a1,b1);
        e1=min(a2,b2);
        if e1-s1>=tol_
            C(end+1,:)=[s1,e1];
        end
        %按照时间顺序，哪一个时间短就推进哪一个
        if a2<b2-tol_
            ia=ia+1;
        else
            ib=ib+1;
        end
    end
    %将相邻的或者优重叠部分的小段合并
    if isempty(C), return; end
    C=merge_intervals_(C,tol_);
end

function M=merge_intervals_(I,tol_)
%将集合排序，排序完成后
    if isempty(I),M=I; return; end
    I=sortrows(I,1);
    M=I(1,:);
    for k=2:size(I,1)
        if I(k,1)<=M(end,2)+tol_
            M(end,2)=max(M(end,2),I(k,2));
        else
            M(end+1,:)=I(k,:);
        end
    end
end
