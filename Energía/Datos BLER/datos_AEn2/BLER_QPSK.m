close all; clc; clear all;
% Número de archivos
%QPSK no codificado
n=2;
k=2;
M=2^k;

num_files = 10;
file_prefix = 'datos_AE_22_cod_fd_';  % Prefijo del nombre del archivo
file_prefix2 = 'datos_AE_32_cod_fd_';  % Prefijo del nombre del archivo
file_prefix3 = 'datos_AE_42_cod_fd_';  % Prefijo del nombre del archivo

data = cell(num_files, 1);
data2 = cell(num_files, 1);
data3 = cell(num_files, 1);

% Importar datos desde los archivos
for i = 1:num_files
    filename = sprintf('%s%d.txt', file_prefix, i);
    data{i} = readtable(filename, 'Delimiter', '\t', 'ReadVariableNames', false);
    filename2 = sprintf('%s%d.txt', file_prefix2, i);
    data2{i} = readtable(filename2, 'Delimiter', '\t', 'ReadVariableNames', false);
    filename3 = sprintf('%s%d.txt', file_prefix3, i);
    data3{i} = readtable(filename3, 'Delimiter', '\t', 'ReadVariableNames', false);
end

% Obtener los valores de EbNodB y BER
EbNodB = data{1}.Var1;  % Suponiendo que la primera columna es EbNodB
EbNo=10.^(EbNodB/10);
ber_matrix = zeros(length(EbNodB), num_files);
ber_matrix2 = zeros(length(EbNodB), num_files);
ber_matrix3 = zeros(length(EbNodB), num_files);
ber_cal = zeros(length(EbNodB), num_files);
ber_cal2 = zeros(length(EbNodB), num_files);
ber_cal3 = zeros(length(EbNodB), num_files);

for i = 1:num_files
    ber_matrix(:, i) = data{i}.Var2;  % Suponiendo que la segunda columna es BLER
    ber_cal(:, i) = 1-(1-data{i}.Var2).^(1/2);
    
    ber_matrix2(:, i) = data2{i}.Var2;  % Suponiendo que la segunda columna es BLER
    ber_cal2(:, i) = 1-(1-data2{i}.Var2).^(1/3);
    
    ber_matrix3(:, i) = data3{i}.Var2;  % Suponiendo que la segunda columna es BLER
    ber_cal3(:, i) =1-(1-data3{i}.Var2).^(1/4);
end

% Calcular la media y el intervalo de confianza del 95 Tasa (2,2)%
mean_ber = mean(ber_matrix, 2);
sem_ber = std(ber_matrix, 0, 2) / sqrt(num_files);  % Error estándar de la media
ci_95 = 1.96 * sem_ber;  % Intervalo de confianza del 95%
mean_ber_cal = mean(ber_cal, 2);
sem_ber_cal = std(ber_cal, 0, 2) / sqrt(num_files);
ci_95_cal = 1.96 * sem_ber_cal;

% Calcular los límites inferior y superior
lower_bound = abs(mean_ber - ci_95);
upper_bound = mean_ber + ci_95;
lower_bound2 = abs(mean_ber_cal - ci_95_cal);
upper_bound2 = mean_ber_cal + ci_95_cal;


% Calcular la media y el intervalo de confianza del 95 Tasa (3,2)%
mean_ber2 = mean(ber_matrix2, 2);
sem_ber2 = std(ber_matrix2, 0, 2) / sqrt(num_files);  % Error estándar de la media
ci2_95 = 1.96 * sem_ber2;  % Intervalo de confianza del 95%
mean_ber_cal2 = mean(ber_cal2, 2);
sem_ber_cal2 = std(ber_cal2, 0, 2) / sqrt(num_files);
ci2_95_cal2 = 1.96 * sem_ber_cal2;

% Calcular la media y el intervalo de confianza del 95 Tasa (4,2)%
mean_ber3 = mean(ber_matrix3, 2);
sem_ber3 = std(ber_matrix3, 0, 2) / sqrt(num_files);  % Error estándar de la media
ci3_95 = 1.96 * sem_ber3;  % Intervalo de confianza del 95%
mean_ber_cal3 = mean(ber_cal3, 2);
sem_ber_cal3 = std(ber_cal3, 0, 2) / sqrt(num_files);
ci3_95_cal3 = 1.96 * sem_ber_cal3;

% Graficar BLER en escala lineal
figure;
sgtitle('BLER vs Eb/No (QPSK)');
subplot(1,2,1);
hold on;
errorbar(EbNo, mean_ber, ci_95, '-x','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNo, mean_ber2, ci2_95, '-kx','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNo, mean_ber3, ci3_95, '--x','color',[0.9290 0.6940 0.1250],'MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
hold off;
%set(gca, 'YScale', 'log');
xlabel('Eb/No (lineal)');
xlim([0 26]);
%ylim([1e-7 1])
ylabel('BLER');
title('Lineal');
legend(sprintf('AE (%d,%d)',2,2),sprintf('AE (%d,%d)',3,2),sprintf('AE (%d,%d)',4,2))
grid on;

subplot(1,2,2);
hold on
set(gca, 'YScale', 'log');
errorbar(EbNodB, mean_ber, ci_95, '','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
%errorbar(EbNodB, mean_ber, lower_bound, upper_bound, '','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNodB, mean_ber2, ci2_95, 'k','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNodB, mean_ber3, ci3_95, '--','color',[0.9290 0.6940 0.1250],'MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
xlabel('Eb/No (dB)');
xlim([0 15]);
ylim([1e-8 1])
ylabel('BLER');
title('semilogarítmica');
grid on;
% Calculate uncoded block error rate (R=k/n=1)
pskBLER = 1-(1-berawgn(EbNodB,'psk',2^k,'nondiff')).^n;
semilogy(EbNodB,pskBLER,'r--','LineWidth',2)
hold off
legend(sprintf('AE (%d,%d)',2,2),sprintf('AE (%d,%d)',3,2),sprintf('AE (%d,%d)',4,2),sprintf('QPSK (%d,%d)',n,k))


% Graficar BER en escala lineal
figure;
sgtitle('BER vs Eb/No (QPSK)');
subplot(1,2,1);
hold on;
errorbar(EbNo, mean_ber_cal, ci_95_cal, '-x','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNo, mean_ber_cal2, ci2_95_cal2, '-kx','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNo, mean_ber_cal3, ci3_95_cal3, '--x','color',[0.9290 0.6940 0.1250],'MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
hold off;
%set(gca, 'YScale', 'log');
xlabel('Eb/No (lineal)');
xlim([0 26]);
%ylim([1e-7 1])
ylabel('BER');
title('Lineal');
legend(sprintf('AE (%d,%d)',2,2),sprintf('AE (%d,%d)',3,2),sprintf('AE (%d,%d)',4,2))
grid on;

subplot(1,2,2);
hold on
set(gca, 'YScale', 'log');
errorbar(EbNodB, mean_ber_cal, ci_95_cal, '','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
%errorbar(EbNo, mean_ber_cal,lower_bound2, upper_bound2, '','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNodB, mean_ber_cal2, ci2_95_cal2, 'k','MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
errorbar(EbNodB, mean_ber_cal3, ci3_95_cal3, '--','color',[0.9290 0.6940 0.1250],'MarkerSize',10 ,'CapSize',25,'LineWidth',1.5 );
xlabel('Eb/No (dB)');
xlim([0 15]);
ylim([1e-8 1])
ylabel('BER');
title('semilogarítmica');
grid on;
% Calculate uncoded block error rate (R=k/n=1)
pskBER = berawgn(EbNodB,'psk',2^k,'nondiff');
semilogy(EbNodB,pskBER,'r--','LineWidth',2)
hold off
legend(sprintf('AE (%d,%d)',2,2),sprintf('AE (%d,%d)',3,2),sprintf('AE (%d,%d)',4,2),sprintf('QPSK (%d,%d)',n,k))
