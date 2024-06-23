close all; clc; clear all;
% Número de archivos
%QPSK no codificado
n=4;
k=4;
M=2^k;


num_files = 10;
file_prefix = 'datos_AE_2_4_3dB_sk_cod_fd_';  % Prefijo del nombre del archivo
file_prefix2 = 'datos_AE_4_4_3dB_sk_cod_fd_';  % Prefijo del nombre del archivo
file_prefix3 = 'datos_AE_6_4_3dB_sk_cod_fd_';  % Prefijo del nombre del archivo
file_prefix4 = 'datos_AE_8_4_3dB_sk_cod_fd_';  % Prefijo del nombre del archivo

data = cell(num_files, 1);
data2 = cell(num_files, 1);
data3 = cell(num_files, 1);
data4 = cell(num_files, 1);

% Importar datos desde los archivos
for i = 1:num_files
    filename = sprintf('%s%d.txt', file_prefix, i);
    data{i} = readtable(filename, 'Delimiter', '\t', 'ReadVariableNames', false);
    filename2 = sprintf('%s%d.txt', file_prefix2, i);
    data2{i} = readtable(filename2, 'Delimiter', '\t', 'ReadVariableNames', false);
    filename3 = sprintf('%s%d.txt', file_prefix3, i);
    data3{i} = readtable(filename3, 'Delimiter', '\t', 'ReadVariableNames', false);
    filename4 = sprintf('%s%d.txt', file_prefix4, i);
    data4{i} = readtable(filename4, 'Delimiter', '\t', 'ReadVariableNames', false);
end

% Obtener los valores de EbNodB y BER
EbNodB = data{1}.Var1;  % Suponiendo que la primera columna es EbNodB
EbNo=10.^(EbNodB/10);
ber_matrix = zeros(length(EbNodB), num_files);
ber_matrix2 = zeros(length(EbNodB), num_files);
ber_matrix3 = zeros(length(EbNodB), num_files);
ber_matrix4 = zeros(length(EbNodB), num_files);
ber_cal = zeros(length(EbNodB), num_files);
ber_cal2 = zeros(length(EbNodB), num_files);
ber_cal3 = zeros(length(EbNodB), num_files);
ber_cal4 = zeros(length(EbNodB), num_files);

for i = 1:num_files
    ber_matrix(:, i) = data{i}.Var2;  % Suponiendo que la segunda columna es BER
    ber_matrix2(:, i) = data2{i}.Var2;  % Suponiendo que la segunda columna es BER
    ber_matrix3(:, i) = data3{i}.Var2;  % Suponiendo que la segunda columna es BER
    ber_matrix4(:, i) = data4{i}.Var2;  % Suponiendo que la segunda columna es BER
    ber_cal(:, i) = 1-(1-data{i}.Var2).^(1/2);
    ber_cal2(:, i) = 1-(1-data2{i}.Var2).^(1/4);
    ber_cal3(:, i) =1-(1-data3{i}.Var2).^(1/6);
    ber_cal4(:, i) =1-(1-data4{i}.Var2).^(1/8);
end

% Calcular la media y el intervalo de confianza del 95 Tasa (2,3)%
mean_ber = mean(ber_matrix, 2);
sem_ber = std(ber_matrix, 0, 2) / sqrt(num_files);  % Error estándar de la media
ci_95 = 1.96 * sem_ber;  % Intervalo de confianza del 95%
mean_ber_cal = mean(ber_cal, 2);
sem_ber_cal = std(ber_cal, 0, 2) / sqrt(num_files);
ci_95_cal = 1.96 * sem_ber_cal;

% Calcular la media y el intervalo de confianza del 95 Tasa (3,3)%
mean_ber2 = mean(ber_matrix2, 2);
sem_ber2 = std(ber_matrix2, 0, 2) / sqrt(num_files);  % Error estándar de la media
ci2_95 = 1.96 * sem_ber2;  % Intervalo de confianza del 95%
mean_ber_cal2 = mean(ber_cal2, 2);
sem_ber_cal2 = std(ber_cal2, 0, 2) / sqrt(num_files);
ci2_95_cal2 = 1.96 * sem_ber_cal2;

% Calcular la media y el intervalo de confianza del 95 Tasa (4,3)%
mean_ber3 = mean(ber_matrix3, 2);
sem_ber3 = std(ber_matrix3, 0, 2) / sqrt(num_files);  % Error estándar de la media
ci3_95 = 1.96 * sem_ber3;  % Intervalo de confianza del 95%
mean_ber_cal3 = mean(ber_cal3, 2);
sem_ber_cal3 = std(ber_cal3, 0, 2) / sqrt(num_files);
ci3_95_cal3 = 1.96 * sem_ber_cal3;

% Calcular la media y el intervalo de confianza del 95 Tasa (6,3)%
mean_ber4 = mean(ber_matrix4, 2);
sem_ber4 = std(ber_matrix4, 0, 2) / sqrt(num_files);  % Error estándar de la media
ci4_95 = 1.96 * sem_ber4;  % Intervalo de confianza del 95%
mean_ber_cal4 = mean(ber_cal4, 2);
sem_ber_cal4 = std(ber_cal4, 0, 2) / sqrt(num_files);
ci4_95_cal4 = 1.96 * sem_ber_cal4;

% limit_sup2=abs(mean_ber2-ci2_95);
% limit_inf2=abs(mean_ber2+ci2_95);
% limit_sup3=abs(mean_ber3-ci3_95);
% limit_inf3=abs(mean_ber3+ci3_95);
% limit_sup4=abs(mean_ber4-ci4_95);
% limit_inf4=abs(mean_ber4+ci4_95);


limit_sup2=mean_ber2-ci2_95;
limit_inf2=mean_ber2+ci2_95;
limit_sup3=mean_ber3-ci3_95;
limit_inf3=mean_ber3+ci3_95;
limit_sup4=mean_ber4-ci4_95;
limit_inf4=mean_ber4+ci4_95;


% Graficar BLER
figure;
sgtitle('BLER vs Eb/No (16QAM)');
subplot(1,2,1);
hold on;
errorbar(EbNo, mean_ber, ci_95, '-x','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNo, mean_ber2, ci2_95, '-kx','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
%errorbar(EbNo, mean_ber2, limit_sup2, limit_inf2, '-kx','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNo, mean_ber3, ci3_95, '--x','color',[0.9290 0.6940 0.1250],'MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNo, mean_ber4, ci4_95, '--x','color',[ 0.10 0.65 0.25],'MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );

hold off;
%set(gca, 'YScale', 'log');
xlabel('Eb/No (lineal)');
xlim([0 26]);
%ylim([1e-7 1])
ylabel('BLER');
title('Lineal');
legend(sprintf('AE (%d,%d)',2,4),sprintf('AE (%d,%d)',4,4),sprintf('AE (%d,%d)',6,4),sprintf('AE (%d,%d)',8,4))
grid on;

subplot(1,2,2);
hold on
set(gca, 'YScale', 'log');
%semilogy(EbNodB,mean_ber,'LineWidth',2)
%semilogy(EbNodB,mean_ber2,'k','LineWidth',2)
%semilogy(EbNodB,mean_ber3,'--','color',[0.9290 0.6940 0.1250],'LineWidth',2)
%errorbar(EbNo, mean_ber2, limit_sup2, limit_inf2, '-kx','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNodB, mean_ber, ci_95, '','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNodB, mean_ber2,  limit_sup2, limit_inf2, 'k','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNodB, mean_ber3,  limit_sup3, limit_inf3, '--','color',[0.9290 0.6940 0.1250],'MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNodB, mean_ber4,  limit_sup4, limit_inf4, '--','color',[0.10 0.65 0.25],'MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );

xlabel('Eb/No (dB)');
xlim([0 15]);
ylim([1e-7 1])
ylabel('BLER');
title('Semilogarítmica');
grid on;

% Calculate uncoded block error rate (R=k/n=1)
pskBLER = 1-(1-berawgn(EbNodB,'qam',2^k,'nondiff')).^n;
semilogy(EbNodB,pskBLER,'r--','LineWidth',2)
hold off
legend(sprintf('AE (%d,%d)',2,4),sprintf('AE (%d,%d)',4,4),sprintf('AE (%d,%d)',6,4),sprintf('AE (%d,%d)',8,4),sprintf('16QAM (%d,%d)',n,k),'Location','southwest')


%%%% Grafica BER
figure;
sgtitle('BER vs Eb/No (16QAM)');
subplot(1,2,1);
hold on;
errorbar(EbNo, mean_ber_cal, ci_95_cal, '-x','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNo, mean_ber_cal2, ci2_95_cal2, '-kx','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNo, mean_ber_cal3, ci3_95_cal3, '--x','color',[0.9290 0.6940 0.1250],'MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNo, mean_ber_cal4, ci4_95_cal4, '--x','color',[ 0.10 0.65 0.25],'MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );

hold off;
%set(gca, 'YScale', 'log');
xlabel('Eb/No (lineal)');
xlim([0 26]);
%ylim([1e-7 1])
ylabel('BER');
title('Lineal');
legend(sprintf('AE (%d,%d)',2,4),sprintf('AE (%d,%d)',4,4),sprintf('AE (%d,%d)',6,4),sprintf('AE (%d,%d)',8,4))
grid on;

subplot(1,2,2);
hold on
set(gca, 'YScale', 'log');
%semilogy(EbNodB,mean_ber,'LineWidth',2)
%semilogy(EbNodB,mean_ber2,'k','LineWidth',2)
%semilogy(EbNodB,mean_ber3,'--','color',[0.9290 0.6940 0.1250],'LineWidth',2)

errorbar(EbNodB, mean_ber_cal, ci_95_cal, '','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNodB, mean_ber_cal2, ci2_95_cal2, 'k','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNodB, mean_ber_cal3, ci3_95_cal3, '--','color',[0.9290 0.6940 0.1250],'MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNodB, mean_ber_cal4, ci4_95_cal4, '--','color',[0.10 0.65 0.25],'MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );

xlabel('Eb/No (dB)');
xlim([0 15]);
ylim([1e-8 1])
ylabel('BER');
title('Semilogarítmica');
grid on;

% Calculate uncoded block error rate (R=k/n=1)
pskBER = berawgn(EbNodB,'qam',16,'nondiff');
semilogy(EbNodB,pskBER,'r--','LineWidth',2)
hold off
legend(sprintf('AE (%d,%d)',2,4),sprintf('AE (%d,%d)',4,4),sprintf('AE (%d,%d)',6,4),sprintf('AE (%d,%d)',8,4),sprintf('16QAM (%d,%d)',n,k),'Location','southwest')
