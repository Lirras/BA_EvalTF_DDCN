Es geht hier nur um die Trainingsdauer. Diese steht in den Plots in 
der dritten Zeile. 

Trainingssamples Targetdatensätze:  
SVHN: 732  
CALI: 240  

Hier verwendete Netze:  
Überall, wo kein Deep Cascade hinter steht, sind es Direct Cascade Netzwerke.

Klassifikation:  
Conv_MaxPool (CMP) (Deep Cascade)  
Classification_one_Dense (ClassOneDense (COD))  
OneDLilConv (1DConv (1DC))  
little_Conv (2DConv (2DC))  

Regression:  
regression_two (regr2 (Regr2)) (Deep Cascade)  
Regression_one_Layer (One_Layer (1Lay))  

Endung: "Cascade" heißt Deep Cascade Netzwerk 

Bedeutung der Dateikürzel:  
Netzkürzel - Netzwerk  
ohne weitere Kürzel: Cascade, mit TF  
complete - vollständiges Netzwerk  
wo - Cascade, ohne TF  

Zeitdauer in Sekunden in der Reihenfolge Cascade mit TF, Cascade ohne TF, 
vollständiges Netzwerk:  
CMP: 78, 25, 20  
COD: 79, 28, 13  
1DC: 207, 34, 30  
2DC: 23, 24, 40  
Regr2: 11, 12, 17  
1Lay: 16, 18, 11