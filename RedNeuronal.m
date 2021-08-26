%Red neuronal para detectar el clima
clear all; clc
Null=[1 15 21 8 0.1 14 20 8 0.5 13 20 7 0.6;
12 15 22 8 0.1 15 22 8 0.1 15 22 9 0.1;];
Ligero=[1 14 21 6 0 14 21 7 0.1 14 21 7 0.1;
1 14 21 7 0.5 14 21 7 0.7 15 21 8 0.3;
1 15 22 8 0.1 15 22 8 0.2 15 22 8 0.9;
2 15 22 9 0.8 16 22 9 0.6 15 22 8 2.6;
2 15 23 8 0.3 15 23 7 0.4 15 22 8 0.4;
2 16 23 9 0.3 17 24 9 0.3 17 24 9 0.6;
2 17 25 9 0.1 17 25 10 0.8 17 25 10 0.1;
3 18 26 10 0 17 25 10 0.3 18 26 10 0.2;
3 18 25 10 1.3 17 25 10 0.6 17 25 10 0.5;
3 18 26 11 0.4 18 26 10 0.3 18 26 10 0.2;
3 19 27 11 0.3 19 27 11 0.3 19 27 12 0.5;
4 20 28 12 0.5 20 27 13 0.1 20 28 12 0.1;
4 20 28 12 0.7 20 28 13 0.6 21 28 13 0.6;
4 21 29 13 0.4 21 29 13 1 22 29 14 1.2;
4 22 29 14 0.6 22 29 14 0.1 22 29 14 1;
5 22 29 14 0.9 22 29 14 0.7 22 29 15 0.7;
5 22 29 14 2.8 22 29 14 1.5 22 29 15 1.3;
5 21 28 15 3.1 21 28 14 3.2 21 28 14 1.9;
5 22 29 14 2.5 22 29 15 2.1 22 29 15 2;
6 21 29 14 2.2 21 28 14 2 21 28 14 1.5;
7 18 24 13 4.2 19 24 14 4.9 19 23 14 4.6;
7 19 24 13 4.8 19 24 14 4.2 19 24 13 3.3;
7 19 24 14 5.7 19 24 13 4.3 19 24 13 3.6;
8 19 24 13 4.5 19 25 13 3.4 19 25 13 3;
8 19 24 14 7.1 19 24 13 4.9 19 24 14 3.3;
8 19 24 14 4.5 19 24 14 3.9 19 25 13 5.5;
9 18 23 13 6.5 18 23 13 5.4 18 23 14 5.8;
9 18 23 13 4.9 18 23 13 3.2 18 23 12 2.9;
9 18 23 13 2.6 18 23 13 3.2 18 23 13 2;
10 17 22 12 2.7 17 23 11 1.5 17 23 12 1.4;
10 18 23 12 3.1 18 23 12 2.5 18 23 12 2.6;
10 18 24 13 2.3 18 24 12 1.2 17 23 11 1.9;
10 17 23 11 1 17 23 11 2 17 23 11 1.2;
11 16 23 10 0.3 16 23 10 0.5 16 23 10 0.8;
11 16 23 10 0.4 16 23 10 0.6 16 23 10 1.2;
11 16 22 10 0.5 16 22 9 0.2 16 22 9 0.4;
11 16 23 9 0.2 16 22 9 0.2 16 22 10 0.2;
12 16 22 9 0.3 15 22 8 0 16 22 9 0.1;
12 14 21 7 0.5 14 21 8 0.2 14 21 7 0.1; 
12 15 21 9 0.4 14 21 8 0.3 15 21 8 0.4;];

Moderada=[6 21 27 15 3.6 21 27 15 4.3 21 27 15 7.6;
6 20 26 15 4.7 20 26 15 3.6 20 26 15 4.9;
6 19 24 14 6.6 19 24 14 6.6 19 25 14 6.8;
7 19 24 14 9.5 19 24 14 9.1 19 24 14 9.2;
8 19 24 14 5.9 19 23 14 7.5 19 24 14 6.5;
9 19 23 14 6.1 19 23 14 6.7 19 23 14 4.5;];
%Vectores objetivo
for i=1:2
O_Null(:,i)=[1 0 0];
end
for i=1:40
O_Ligero(:,i)=[0 1 0];
end
for i=1:6
O_Moderada(:,i)=[0 0 1];
end
%Creación de vectores de entrada y targets
input=[Null' Ligero' Moderada'];
targets=[O_Null O_Ligero O_Moderada];
red=patternnet(5,'trainlm'); %Creamos una red con 5 neuronas
% en la capa oculta y el trainlm
% nos indica que la topologia de la red serábackpropagation
red.trainParam.epochs=[1000];%Num de epocas o veces que se va a iterar
red.trainParam.max_fail=100;%Cuantas veces va a verificar los resultados, va a buscar la mejor respuesta n veces
red.trainParam.min_grad=1e-29;%Error permitido
red.trainParam.mu=0.1;%Factor de aprendizaje
red.trainParam.mu_dec=0.1;%decreciente acercarse al valor deceado con delicadesa
red.trainParam.mu_inc=10;%Incremento lo más rapido posible
%capa1
%red.layers{1}.transferFcn='tansig';% cambia la función de activación
%capa 2
%red.layers{2}.transferFcn='tansig';% cambia la función de activación

configure(red,input,targets);
red.divideParam.trainRatio=90/100;
red.divideParam.valRatio=5/100;
red.divideParam.testRatio=5/100;
[red,tr]=train(red,input,targets);
%MUESTRAS ALEATORIAS PARA COMPROBAR EFECTIVIDAD DE LA RNA
X1=[1 15 22 8 0.1 15 22 8 0.2 15 22 8 0.9]; %0 1 0
X2=[3 18 26 11 0.4 18 26 10 0.3 18 26 10 0.2]; %0 1 0
X3=[6 20 26 15 4.7 20 26 15 3.6 20 26 15 4.9]; %0 0 1
X4=[8 19 24 13 4.5 19 25 13 3.4 19 25 13 3]; %0 1 0
X5=[1 15 21 8 0.1 14 20 8 0.5 13 20 7 0.6]; %1 0 0
X6=[9 19 23 14 6.1 19 23 14 6.7 19 23 14 4.5]; %0 0 1
X7=[10 18 24 13 2.3 18 24 12 1.2 17 23 11 1.9]; %0 1 0
X8=[12 16 22 9 0.3 15 22 8 0 16 22 9 0.1]; %0 1 0
X9=[12 15 22 8 0.1 15 22 8 0.1 15 22 9 0.1]; %1 0 0
X10=[12 15 21 9 0.4 14 21 8 0.3 15 21 8 0.4]; %0 1 0




