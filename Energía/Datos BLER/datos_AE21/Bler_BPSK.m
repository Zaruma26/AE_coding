close all; clc; clear all;
% Número de archivos
n=2;
k=1;
M=2^k;
num_files = 10;
file_prefix = 'datos_AE_31_2cod_fdsk_';  % Prefijo del nombre del archivo
data = cell(num_files, 1);

% Importar datos desde los archivos
for i = 1:num_files
    filename = sprintf('%s%d.txt', file_prefix, i);
    data{i} = readtable(filename, 'Delimiter', '\t', 'ReadVariableNames', false);
end

% Obtener los valores de EbNodB y BER
EbNodB = data{1}.Var1;  % Suponiendo que la primera columna es EbNodB
EbNo=10.^(EbNodB/10);
ber_matrix = zeros(length(EbNodB), num_files);
ber_cal = zeros(length(EbNodB), num_files);

for i = 1:num_files
    ber_matrix(:, i) = data{i}.Var2;  % Suponiendo que la segunda columna es BER
    ber_cal(:, i) = 1-(1-data{i}.Var2).^(1/2);
end

% Calcular la media y el intervalo de confianza del 95%
mean_ber = mean(ber_matrix, 2);
dst=std(ber_matrix, 0, 2);
sem_ber = std(ber_matrix, 0, 2) / sqrt(num_files);  % Error estándar de la media
ci_95 = 2.58 * sem_ber;  % Intervalo de confianza del 95%
mean_ber_cal = mean(ber_cal, 2);
sem_ber_cal = std(ber_cal, 0, 2) / sqrt(num_files);
ci_95_cal = 1.96 * sem_ber_cal;

% Graficar BLER
figure;
sgtitle('BLER vs Eb/No (BPSK)');
subplot(1,2,1);
errorbar(EbNo, mean_ber, ci_95, '-x','MarkerSize',10 ,'CapSize',25,'LineWidth',1 );
%set(gca, 'YScale', 'log');
xlabel('Eb/No (lineal)');
xlim([0 26]);
%ylim([1e-7 1])
ylabel('BLER');
title('lineal');
grid on;
subplot(1,2,2);
errorbar(EbNodB, mean_ber, ci_95, '-x','MarkerSize',10 ,'CapSize',25,'LineWidth',1 );
set(gca, 'YScale', 'log');
xlabel('Eb/No (dB)');
xlim([0 15]);
ylim([1e-7 1])
ylabel('BLER');
title('semilogarítmica');
grid on;
hold on
% Calculate uncoded block error rate (R=k/n=1)
pskBLER = 1-(1-berawgn(EbNodB,'psk',2,'nondiff')).^1;
pskBLER2 = 1-(1-berawgn(EbNodB,'psk',2,'nondiff')).^2;
semilogy(EbNodB,pskBLER,'--','LineWidth',1.5)
semilogy(EbNodB,pskBLER2,'k','LineWidth',1.5)
hold off
legend(sprintf('AE (%d,%d)',n,k),sprintf('BPSK (%d,%d)',1,1),sprintf('BPSK (%d,%d)',2,1))

% Graficar BER
figure;
sgtitle('BER vs Eb/No (BPSK)');
subplot(1,2,1);
errorbar(EbNo, mean_ber_cal, ci_95_cal, '-x','MarkerSize',10 ,'CapSize',25,'LineWidth',1 );
%set(gca, 'YScale', 'log');
xlabel('Eb/No (lineal)');
xlim([0 26]);
%ylim([1e-7 1])
ylabel('BER');
title('lineal');
grid on;
subplot(1,2,2);
errorbar(EbNodB, mean_ber_cal, ci_95_cal, '-x','MarkerSize',10 ,'CapSize',25,'LineWidth',1 );
set(gca, 'YScale', 'log');
xlabel('Eb/No (dB)');
xlim([0 15]);
ylim([1e-8 1])
ylabel('BER');
title('semilogarítmica');
grid on;
hold on
% Calculate uncoded block error rate (R=k/n=1)
pskBER = berawgn(EbNodB,'psk',2,'nondiff');
%pskBLER2 = 1-(1-berawgn(EbNodB,'psk',4,'nondiff')).^2;
semilogy(EbNodB,pskBER,'--','LineWidth',1.5)
%semilogy(EbNodB,pskBLER2,'k','LineWidth',1.5)
hold off
legend(sprintf('AE (%d,%d)',n,k),sprintf('BPSK (%d,%d)',2,1))
