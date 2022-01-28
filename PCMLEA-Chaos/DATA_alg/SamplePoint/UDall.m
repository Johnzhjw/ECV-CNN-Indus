function uns = UDall(n,s)
% 均匀分布表  好格子点法
% n  水平数    s  因素数
h = 1:n;
ind = find(gcd(h,n)==1);   %寻找比n小且与n互素的数h
hm = h(ind);                   %生成向量
m = length(hm);             %生成向量维数，不小于因素数 s, length(hm) >= s
udt = mod(h'*hm,n);        %好格子点法
ind0 = find(udt==0);
udt(ind0)=n;                   %生成均匀设计表U(n^s)
%udt(end,:)=[];               %生成均匀设计表U^*((n-1)^s)
if s>m
    disp('s必须小于或等于m');
    return;
else
    mcs =nchoosek(m,s);   %从n 个元素中一次选k 个元素的所有组合数
    if mcs<1e5
        tind = nchoosek(1:m,s);   %得到组合的可能形式
        [p,q] = size(tind);cd2 = zeros(p,1);
        for k=1:p
            UT = udt(1:n,tind(k,:));
            cd2(k,1) = UDCD2(UT);
        end
        tc=tind(find(abs(cd2 - min(cd2))<1e-5),:);
        for r=1:size(tc,1);
            uns (:,:,r)= udt(:,tc(r,:));
        end
    else
        for k = 1:n
            a = k;
            UT = mod(h'*a.^(0:s-1),n);
            cd2(k,1) = UDCD2(UT);
        end
        tc = find(abs(cd2 - min(cd2))<1e-5);
        for r=1:size(tc,1);
            uns (:,:,r)=  mod(h'*tc(r).^(0:s-1),n);
        end
        ind0 = find(uns==0); uns(ind0)=n;
    end
end
Data=uns(:,:,1);
Data=(Data-1)/(n-1);
save(['SamplePoint_Dim'  num2str(size(Data,2)) 'N' num2str(size(Data,1))  '.txt'],'Data','-ASCII');

%output the array to txt
fp = fopen('udarray.txt', 'w');
[l,m,n] = size(uns);
% for k = 1: n
%     fprintf(fp, '{ ');
%     for i = 1:l
%         fprintf(fp, '{ ');
%         for j = 1:m-1
%             fprintf(fp, '%d, ', uns(i,j,k));
%         end
%         fprintf(fp, '%d} ,\n', uns(i,m,k));
%     end
%     fprintf(fp, '} \n\n\n');
% end
% fclose(fp);

for k = 1: n
    for i = 1:l
        for j = 1:m-1
            fprintf(fp, '%d  ', uns(i,j,k));
        end
        fprintf(fp, '%d \n', uns(i,m,k));
    end
    fprintf(fp, '\n\n\n');
end
fclose(fp);



