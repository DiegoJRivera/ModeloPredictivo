# Tarea Número 2 Modelaiento Predictivo

# Integrantes
# Diego Gálvez - Diego Rivera - Cinthia Roa

# setting directorio de trabajo
setwd("~/GoogleDrive/UAI/1_MDS2019/6_ModelamientoPredictivo/Tarea2")

install.packages("devtools")                # --|
library(devtools)                           #   --> Para instalacion de paquete streamlineR
install_github("JianhuaHuang/streamlineR")  # --|
# librerías a utilizar
library(dplyr) # para manipulación de data frames
library(inspectdf) # para inspección en detalle del data frame
library(stringr) # para manejo de strings
install.packages('Amelia')
library(Amelia) # para graficar datos faltantes
library(ggplot2) # para visualización de datos
install.packages('tidyquant')
library(tidyquant) # para mejorar display de ggplot
library(gridExtra) # para visualización de datos conjuntos
library(corrplot) # para visualización de corrplot
library(caret) # para la separacion de la data en train y test
library(Matrix)
library(glmnet)
library(ROCR) # para cálculo de AUC (área bajo la curva ROC --> curva de característica operativa del recepto)
library(InformationValue) # para  cálculo de estadisticos
install.packages('streamlineR')
library(streamlineR)
install.packages("scorecard")
library(scorecard)

############################################
# CONTEXTUALIZACIÓN DEL CASO A DESARROLLAR #
############################################

# El conjunto de datos HMEQ entrega múltiple información sobre morosidad para 5960 préstamos con garantía hipotecaria otorgados por un
# banco. Un préstamo con garantía hipotecaria es un préstamo en el que el deudor utiliza el valor líquido (home equity) de su vivienda
# como garantía subyacente. El banco está interesado en determinar la fiabilidad de un cliente que solicita prestamos.
# Cuando un banco recibe una solicitud de préstamo, según el perfil del solicitante, el banco tiene que tomar una decisión sobre si 
# continuar con la aprobación del préstamo o no. 
# Dos tipos de riesgos están asociados con la decisión del banco:
#     i)  si el solicitante tiene un bajo riesgo, es decir, es probable que reembolse el préstamo, entonces no aprobar el préstamo a 
#         la persona resulta en una pérdida de negocios para el banco
#     ii) si el solicitante tiene un alto riesgo, es decir, no es probable que pague el préstamo, la aprobación del préstamo a la 
#         persona resulta en una pérdida financiera para el banco.

# OBJETIVO DEL DESARROLLO:
# Automatizar el proceso de toma de decisiones para la aprobación de líneas de crédito con garantía hipotecaria.
# Creando un modelo de puntuación de crédito empíricamente derivado y estadísticamente sólido. El modelo se basará en los datos 
# recopilados de solicitantes recientes que recibieron crédito a través del proceso actual de suscripción de préstamos. El modelo se
# construirá a partir de herramientas de modelado predictivo, pero el modelo creado debe ser lo suficientemente interpretable para 
# proporcionar una razón para cualquier acción adversa (rechazos).

##############################################
# INICIANDO DESARROLLO DEL MODELO PREDICTIVO #
##############################################

### 1. Identificación de los atributos y variable target ###

# Data Set HMEQ asignado a variable "hmeq_data"
hmeq_data <- read.csv(file.choose(), header = T, sep = ",")

# analizando la estructura del set de datos de las HMEQ
names(hmeq_data)

# Los datos del archivo HMEQ contienen datos sobre 13 atributos con las siguientes características:
# 1.-  BAD (Variable Target) --> 1 = el solicitante incumplió con el préstamo o se encuentra seriamente en mora;
#                                0 = préstamo pagado por el solicitante.
# 2.-  LOAN: Monto de la solicitud de préstamo.
# 3.-  MORTDUE: Monto adeudado por la hipoteca existente.
# 4.-  VALUE: Valor de la propiedad actual.
# 5.-  REASON: DebtCon = consolidación de la deuda. HomeImp = mejoras del hogar.
# 6.-  JOB: Categorías ocupacionales.
# 7.-  YOJ: Años en el trabajo actual
# 8.-  DEROG: Número de informes despectivos importantes
# 9.-  DELINQ: Número de líneas de crédito morosas
# 10.- CLAGE: Edad de la línea de crédito más antigua en meses
# 11.- NINQ: Número de consultas de crédito recientes
# 12.- CLNO: Numero de líneas de credito
# 13.- DEBTINC: Relación deuda-ingreso

### 2. Análisis descriptivo (estadísticos de resumen, gráficos, etc.) de los atributos ###
# Análisis exploratorio de las variables
str(hmeq_data)
summary(hmeq_data)

## Revisando presencia de datos NA, que serán tratados un poco más adelnate.
hmeq_data %>% inspect_na
hmeq_data %>% inspect_na %>% show_plot
hmeq_data %>% inspect_types()

# Generando un backup del set de datos orignales, previo a la modificación de ciertas variables para el cumplimiento del objetivo
# declarado previamente.
backUp1Hmeq <- hmeq_data

# Para darle mayor valor a la variable JOB dentro modelo a desarrollar se puntuará los distintos tipos de trabajos, dejando como "Other"
# a quienes no tengan dato alguno y mejor puntuación a quienes en base a trabajo tengan mejor chance de recibir una aprobación
# de líneas de crédito con garantía hipotecar. El criterio será el siguiente:
# JOB	       Propuesta puntuacion para variable JOB
# Mgr	               6
# Office	           5
# Sales	             4
# ProfExe	           3
# Self	             2
# Other = Blank	     1

# REASON	   Propuesta puntuacion para variable REASON
# DebtCon	           3
# HomeImp	           2
# Blank = Other      1

table(hmeq_data$REASON)
table(hmeq_data$JOB)

# primero en JOB --> pasando los campos vacios a Other
hmeq_data$JOB[which(hmeq_data$JOB=="")] = 'Other'
hmeq_data$JOB = factor(hmeq_data$JOB, labels = c('Mgr','Office','Other','ProfExe','Sales','Self'))
table(hmeq_data$JOB)

# luego en REASON --> pasando los campos vacios a Other
a = which(hmeq_data$REASON == "") 
hmeq_data$REASON = as.character(hmeq_data$REASON)
hmeq_data$REASON[a] = 'Other'
hmeq_data$REASON = as.factor(hmeq_data$REASON)
table(hmeq_data$REASON)

# Aprovechando de pasar como factor la variable target BAD
hmeq_data$BAD <- as.factor(hmeq_data$BAD)

# 2 backup de datos con arreglo de variable JOB y REASON, para respaldar set de datos antes de eliminacion de categorizacion de 
# variables JOB y REASON. backUp2Hmeq será usado unicamente para fines de visualización de datos de las variables JOB y REASON.
backUp2Hmeq <- hmeq_data

hmeq_data$JOB <- as.character(hmeq_data$JOB)
hmeq_data$REASON <- as.character(hmeq_data$REASON)

for(i in 1:5960) {
  # Ajustando variable JOB
  if (str_detect(hmeq_data$JOB[i], pattern = "Mgr")) {
      hmeq_data$JOB[i] <- "6"
  } else if (str_detect(hmeq_data$JOB[i], pattern = "Office")) {
      hmeq_data$JOB[i] <- "5"
  } else if (str_detect(hmeq_data$JOB[i], pattern = "Sales")) {
      hmeq_data$JOB[i] <- "4"
  } else if (str_detect(hmeq_data$JOB[i], pattern = "ProfExe")) {
      hmeq_data$JOB[i] <- "3"
  } else if (str_detect(hmeq_data$JOB[i], pattern = "Other")) {
      hmeq_data$JOB[i] <- "2"
  } else if (str_detect(hmeq_data$JOB[i], pattern = "Self")) {
      hmeq_data$JOB[i] <- "1"
  } else {hmeq_data$JOB[i] <- "0"}
  
  # Ajustando variable REASON
  if (str_detect(hmeq_data$REASON[i], pattern = "DebtCon")) {
    hmeq_data$REASON[i] <- "3"
  } else if (str_detect(hmeq_data$REASON[i], pattern = "HomeImp")) {
    hmeq_data$REASON[i] <- "2"
  } else {hmeq_data$REASON[i] <- "1"}
}

table(hmeq_data$REASON)
table(hmeq_data$JOB)

# Pasando puntuación final de variable JOB y REASON a factor por ser variables categoricas para data set hmeq_data y backUp2Hmeq.
hmeq_data$JOB <- as.factor(hmeq_data$JOB)
hmeq_data$REASON <- as.factor(hmeq_data$REASON)

backUp2Hmeq$JOB <- as.factor(backUp2Hmeq$JOB)
backUp2Hmeq$REASON <- as.factor(backUp2Hmeq$REASON)

str(hmeq_data)
str(backUp2Hmeq)

# nuevo backup de datos con arreglo de variable JOB y REASON, para respaldar set de datos antes de eliminacion de
# registros NA
backUp3Hmeq <- hmeq_data

# Retomando caso de valores NA's
missmap(hmeq_data, col= c('white','cyan')  , main ='Missmap de Datos de set HMEQ')
missmap(backUp2Hmeq, col= c('white','cyan')  , main ='Missmap de Datos de set backup de HMEQ')

# 5960 registros en 13 variables
dim(hmeq_data)
dim(backUp2Hmeq)

# Limpiando valores NA's
hmeq_data <- na.omit(hmeq_data)
missmap(hmeq_data, col= c('white','cyan')  , main ='Missmap de Datos de set HMEQ')

backUp2Hmeq <- na.omit(backUp2Hmeq)
missmap(backUp2Hmeq, col= c('white','cyan')  , main ='Missmap de Datos de set backup de HMEQ')

# 3515 registros en 13 variables
dim(hmeq_data)
dim(backUp2Hmeq)

str(hmeq_data)
summary(hmeq_data)

str(backUp2Hmeq)
summary(backUp2Hmeq)

# Visualizando la distribución de las variables numéricas en histogramas no balanceadas.
hist1 <- ggplot(data = backUp2Hmeq, aes(x = LOAN)) +
  geom_histogram() +
  theme_tq()

hist2 <- ggplot(data = backUp2Hmeq, aes(x = MORTDUE)) +
  geom_histogram() +
  theme_tq()

hist3 <- ggplot(data = backUp2Hmeq, aes(x = VALUE)) +
  geom_histogram() +
  theme_tq()

hist4 <- ggplot(data = backUp2Hmeq, aes(x = YOJ)) +
  geom_histogram() +
  theme_tq()

hist5 <- ggplot(data = backUp2Hmeq, aes(x = DEROG)) +
  geom_histogram() +
  theme_tq()

hist6 <- ggplot(data = backUp2Hmeq, aes(x = DELINQ)) +
  geom_histogram() +
  theme_tq()

hist7 <- ggplot(data = backUp2Hmeq, aes(x = CLAGE)) +
  geom_histogram() +
  theme_tq()

hist8 <- ggplot(data = backUp2Hmeq, aes(x = NINQ)) +
  geom_histogram() +
  theme_tq()

hist9 <- ggplot(data = backUp2Hmeq, aes(x = CLNO)) +
  geom_histogram() +
  theme_tq()

hist10 <- ggplot(data = backUp2Hmeq, aes(x = DEBTINC)) +
  geom_histogram() +
  theme_tq()

grid.arrange(hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, hist10)

# Análisis exploratorio de datos para cada variable, pariendo con potenciales variables categóricas.
# Donde:
#        - Rojo es 0 (préstamo pagado por el solicitante.
#        - Cyan (Celeste) es 1 (el solicitante incumplió con el préstamo o se encuentra seriamente en mora).

# REASON: DebtCon = consolidación de la deuda. HomeImp = mejoras del hogar.
# JOB: Categorías ocupacionales.
BadReason <- ggplot(backUp2Hmeq, aes(REASON, fill=BAD)) +
  labs(title = 'Relación BAD sobre REASON', x = 'REASON') +
  geom_bar(position='fill') + theme(legend.position="none")

BadJob <- ggplot(backUp2Hmeq, aes(JOB, fill=BAD)) +
  labs(title = 'Relación BAD sobre JOB') +
  geom_bar(position='fill') + theme(legend.position="none")

grid.arrange(BadReason, BadJob, ncol = 2)

# Analizando NINQ: Número de consultas de crédito recientes
BadNinq=ggplot(hmeq_data, aes(NINQ, fill=BAD)) +
  labs(title = 'Relación BAD sobre Número de consultas de crédito recientes') +
  geom_bar(position='fill') + theme(legend.position="none")
BadNinq

table(hmeq_data$NINQ)
str(hmeq_data$NINQ)
summary(hmeq_data$NINQ)

# Para lograr una mayor representatividad para el modelo a desarrollar, en el número de consultas de crédito recientes, todos los 
# valores mayores o iguales a 4 se agruparán y pasarán como factores.
a = hmeq_data$NINQ
hmeq_data$NINQ2 = 0
for ( i in 1:length(a) ) {
  if ( a[i] >= 4 ) {
    hmeq_data$NINQ2[i] = 4
  }
  else hmeq_data$NINQ2[i] = a[i]
}
table(hmeq_data$NINQ2)
hmeq_data$NINQ2 = as.factor(hmeq_data$NINQ2)

table(hmeq_data$NINQ2)
str(hmeq_data$NINQ2)
summary(hmeq_data)

BadNinq2 = ggplot(hmeq_data, aes(NINQ2, fill=BAD)) +
  labs(title = 'Relación BAD sobre Número de consultas de crédito recientes (Modificado)') +
  geom_bar(position='fill') + theme(legend.position="none")

grid.arrange(BadNinq, BadNinq2, ncol = 2)
# Por medio del gráfico de barras anterior, NINQ2 parece tener un efecto significativo en la variable de respuesta mejor que NINQ

# Analizando DEROG: Número de informes despectivos importantes
table(hmeq_data$DEROG)
# Para lograr una mayor representatividad para el modelo a desarrollar, en el número de informes despectivos importantes, todos los 
# valores mayores o iguales a 2 se agruparán y pasarán como factores.
b = hmeq_data$DEROG
hmeq_data$DEROG2 = 0
for ( i in 1:length(b) ) {
  if ( b[i] >= 2 ) {
    hmeq_data$DEROG2[i] = 2
  }
  else hmeq_data$DEROG2[i] = b[i]
}
table(hmeq_data$DEROG2)
hmeq_data$DEROG2 = as.factor(hmeq_data$DEROG2)

BadDerog = ggplot(hmeq_data, aes(DEROG,fill=BAD)) +
  labs(title = 'Relación BAD sobre Número de informes despectivos importantes') +
  geom_bar(position='fill') + theme(legend.position="none")

BadDerog2 = ggplot(hmeq_data, aes(DEROG2,fill=BAD)) +
  labs(title = 'Relación BAD sobre Número de informes despectivos importantes (Modificado)') +
  geom_bar(position='fill') + theme(legend.position="none")

grid.arrange(BadDerog, BadDerog2, ncol = 2)
# Por medio del gráfico de barras anterior, DEROG2 parece tener un efecto significativo en la variable de respuesta mejor que DEROG

# Analizando DELINQ: Número de líneas de crédito morosas
table(hmeq_data$DELINQ)
# Para lograr una mayor representatividad para el modelo a desarrollar, en el número de líneas de crédito morosas, todos los 
# valores mayores o iguales a 2 se agruparán y pasarán como factores.
c = hmeq_data$DELINQ
hmeq_data$DELINQ2 = 0
for ( i in 1:length(c) ) {
  if ( c[i] >= 2 ) {
    hmeq_data$DELINQ2[i]=2
  }
  else hmeq_data$DELINQ2[i] = c[i]
}
table(hmeq_data$DELINQ2)
hmeq_data$DELINQ2 = as.factor(hmeq_data$DELINQ2)

BadDelinq = ggplot(hmeq_data, aes(DELINQ,fill=BAD)) +
  labs(title = 'Relación BAD sobre Número de líneas de crédito morosas') +
  geom_bar(position='fill') + theme(legend.position="none")

BadDelinq2 = ggplot(hmeq_data, aes(DELINQ2,fill=BAD)) +
  labs(title = 'Relación BAD sobre Número de líneas de crédito morosas (Modificado)') +
  geom_bar(position='fill') + theme(legend.position="none")

grid.arrange(BadDelinq, BadDelinq2, ncol = 2)
# Por medio del gráfico de barras anterior, DELINQ2 parece tener un efecto significativo en la variable de respuesta mejor que DELINQ

str(hmeq_data)

# Visualizando outliers para variables continuas. (Opción de visualizarlos con grafico de desidades, sólo si queda tiempo)
# LOAN: Monto de la solicitud de préstamo.
BadLoan = hmeq_data %>%
  ggplot(aes(x=BAD, y=LOAN, fill=BAD)) + geom_boxplot()

# MORTDUE: Monto adeudado por la hipoteca existente.
BadMotude = hmeq_data %>%
  ggplot(aes(x=BAD, y=MORTDUE, fill=BAD)) + geom_boxplot()

# VALUE: Valor de la propiedad actual.
BadValue = hmeq_data %>%
  ggplot(aes(x=BAD, y=VALUE, fill=BAD)) + geom_boxplot()

# YOJ: Años en el trabajo actual
BadYoj = hmeq_data %>%
  ggplot(aes(x=BAD, y=YOJ, fill=BAD)) + geom_boxplot()

# CLAGE: Edad de la línea de crédito más antigua en meses
BadClage = hmeq_data %>%
  ggplot(aes(x=BAD, y=CLAGE, fill=BAD)) + geom_boxplot()

# CLNO: Numero de líneas de credito
BadClno = hmeq_data %>%
  ggplot(aes(x=BAD, y=CLNO, fill=BAD)) + geom_boxplot()

# DEBTINC: Relación deuda-ingreso
BadDebtinc = hmeq_data %>%
  ggplot(aes(x=BAD, y=DEBTINC, fill=BAD)) + geom_boxplot()

grid.arrange(BadLoan, BadMotude, BadValue, BadYoj, BadClage, BadClno, BadDebtinc, ncol=2)
# Los Boxplot, no es fácil visualizar las variables que tienen un efecto significativo en las variable de respuesta.

# Analizando correlaciones para las variables:
# 2.-  LOAN: Monto de la solicitud de préstamo.
# 3.-  MORTDUE: Monto adeudado por la hipoteca existente.
# 4.-  VALUE: Valor de la propiedad actual.
# 7.-  YOJ: Años en el trabajo actual
# 10.- CLAGE: Edad de la línea de crédito más antigua en meses
# 12.- CLNO: Numero de líneas de credito
# 13.- DEBTINC: Relación deuda-ingreso
str(hmeq_data)
dim(hmeq_data)

corrplot(cor(hmeq_data[,-c(1,5,6,8,9,11,14,15,16)]))
dev.off()
corrplot(cor(hmeq_data[,-c(1,5,6,8,9,11,14,15,16)]), method='number')
# Debido a la multicolinealidad entre MORTDUE y VALUE, se considerará una nueva variable que contemple el patrimonio real disponible del
# sujeto a evaluar, basado en el diferencial entre el valor de la propiedad actual y el monto adeudado por la hipoteca existente lo que
# considerará como la garantía real del solicitante del crédito.   

hmeq_data$GARANTIA = 0
for ( i in 1:length(hmeq_data$VALUE)) {
  hmeq_data$GARANTIA[i] = hmeq_data$VALUE[i] - hmeq_data$MORTDUE[i]
}

summary(hmeq_data$GARANTIA)
# Para aquellos valores con resultado negativos en la nueva variable GARANTIA, podrían considerarse como potenciales clientes no
# aptos para aprobación del crédito, debido que la garantía estaría basada en un pasivo y no en un activo.

# Data para trabajar el modelo
names(hmeq_data)

dataModelo <- hmeq_data[,-c(3,4,8,9,11)]
names(dataModelo)

dataModeloBalanceada <- data_balanceada[,-c(3,4,8,9,11)]
names(dataModeloBalanceada)

corrplot(cor(dataModelo[,-c(1,3,4,9,10,11)]))
plot.new()
corrplot(cor(dataModelo[,-c(1,3,4,9,10,11)]),method='number')
table(dataModelo$BAD)
# Ahora se aprecia una correlación ideal para las variables continuas a considerar para la generacion del modelo.

#######################################################################################################
### 4. Construcción del modelo: Utilizar técnicas de selección de variables, regularización ###


# Balanceando los datos para comparar el modelo final seleccionado entre datos balanceados y no balanceados.
summary(dataModelo$BAD)
balanceo <- sample_n(dataModelo, 309, replace = FALSE, weight = BAD == 0) # 309 filas
summary(balanceo)

balanceo_bad1 <- sample_n(dataModelo, 309, replace = FALSE, weight = BAD == 1)
summary(balanceo_bad1)

data_balanceada <- rbind(balanceo,balanceo_bad1)
summary(data_balanceada)

###

table(dataModelo$BAD)
table(data_balanceada$BAD)

# No balanceado
modM = glm(BAD~., data = dataModelo, family=binomial)
summary(modM)


# balanceado
modM_bal = glm(BAD~., data = data_balanceada, family=binomial)
summary(modM_bal)


# Seleccion de variables con stepwise
step(modM)


##modelo stepwise no balanceada 
##glm(formula = BAD ~ LOAN + JOB + CLAGE + CLNO + DEBTINC + NINQ2 +  DEROG2 + DELINQ2 + GARANTIA, family = binomial, data = dataModelo)
##SACA VARIABLES REASON y YOJ
# Data No Balanceada
str(dataModelo)
factores = model.matrix(~ JOB+ NINQ2+DEROG2+DELINQ2,data=dataModelo) # para dejarlas como factor las categoricas
numerica  = scale(dataModelo[,c(2,6,7,8,12)])
x = data.frame(factores,numerica)
x = as.matrix(x)
y = dataModelo[,"BAD"]

# regresion logistica con Lasso

  library(Matrix)
  library(glmnet)
fitLasso = glmnet(x,y,alpha=1,family="binomial")

par(mfrow=c(1,2))
plot(fitLasso,xvar="lambda")
plot(fitLasso,xvar="norm")

set.seed(124)
cvLasso = cv.glmnet(x,y,alpha=1,nfolds=5,family="binomial")
# lambda gorro
cvLasso$lambda.min  
which(cvLasso$lambda==cvLasso$lambda.min)
#0.001577991
# este es el valor de lambda que minimiza el error de testeo estimado
cvLasso$cvm[41]
## 0.4765155
# este es el estimador del error de testeo cuando lambda=41.
plot(cvLasso)

# coeficientes estimados por lasso
coef(fitLasso,s=cvLasso$lambda.min)

# predicciones en la muestra
prob.pred.Lasso = predict(fitLasso, newx=x, s=cvLasso$lambda.min,type = "response")

# vamos a calcular la matriz de confusion
clase.pred = ifelse(prob.pred.Lasso>=0.5,1,0)
# resultados
caret::confusionMatrix(as.factor(clase.pred), as.factor(y))

##accuracy 0,9218
##sensitivity:0,9925
##specificity :0,1877

# Seleccion de variables con stepwise BALANCEADA

step(modM_bal)
##glm(formula = BAD ~ LOAN + JOB + YOJ + CLAGE + CLNO + DEBTINC + NINQ2 + DEROG2 + DELINQ2, family = binomial, data = data_balanceada)

# Data Balanceada
str(data_balanceada)
factores_bal = model.matrix(~JOB+NINQ2+DEROG2+DELINQ2,data=data_balanceada) #VARIABLES CATEGORICAS
numerica_bal  = scale(data_balanceada[,c(2,5,6,7,8)])
x_bal = data.frame(factores_bal,numerica_bal)
x_bal = as.matrix(x_bal)
y_bal = data_balanceada[,"BAD"]

# regresion logistica con Lasso CON BALANCEADA

fitLasso_bal = glmnet(x_bal,y_bal,alpha=1,family="binomial")

par(mfrow=c(1,2))
plot(fitLasso_bal,xvar="lambda")
plot(fitLasso_bal,xvar="norm")

set.seed(124)
cvLasso_bal = cv.glmnet(x_bal,y_bal,alpha=1,nfolds=5,family="binomial")
# lambda gorro
cvLasso_bal$lambda.min  
which(cvLasso_bal$lambda==cvLasso_bal$lambda.min)
#0.004187969
# este es el valor de lambda que minimiza el error de testeo estimado
cvLasso_bal$cvm[38] ##38
## 1.166703
# este es el estimador del error de testeo cuando lambda=41.
plot(cvLasso_bal)

# coeficientes estimados por lasso
coef(fitLasso_bal,s=cvLasso_bal$lambda.min)

# predicciones en la muestra
prob.pred.Lasso_bal = predict(fitLasso_bal, newx=x_bal, s=cvLasso_bal$lambda.min,type = "response")

# vamos a calcular la matriz de confusion
clase.pred_bal = ifelse(prob.pred.Lasso_bal>=0.5,1,0)
# resultados
caret::confusionMatrix(as.factor(clase.pred_bal), as.factor(y_bal))

##accuracy 0,7314
##sensitivity 0,8091
##specificity 0,6602

#no balanceada
modelo_lasso= dataModelo["BAD","LOAN", "JOB", "NINQ2","DELINQ2","DEROG2", "DELINQ", "YOJ" ,"CLNO","GARANTIA"]
##balanceada
modelo_lasso_bal= data_balanceada["BAD","LOAN","JOB", "CLAGE", "NINQ2","DELINQ2", "DEROG2", "DELINQ", "YOJ" ,"CLNO"]

### DIVIDIR LA DATA EN ENTRENAMIENTO Y TESTEO

# Incluya discusiones.
### 3. Elección del modelo a desarrollar ###
# Dividiendo la data sin balanceo
set.seed(12345)
trainIndex <- createDataPartition(dataModelo$BAD, times = 1, p = 0.8, list=FALSE)
setTrain <- dataModelo[trainIndex,]
setTest <- dataModelo[-trainIndex,]

dim(setTrain) # 2813 observaciones correspondiente al 80% de la muestra trainIdex

# MODELO MULTIVARIADO
modM = glm(BAD~., data = setTrain, family=binomial)
summary(modM)

# Seleccion de variables con stepwise
step(modM)

##modelo de stepwise : glm(formula = BAD ~ LOAN + CLAGE + CLNO + DEBTINC + NINQ2 + 
#DEROG2 + DELINQ2, family = binomial, data = setTrain)

# ajuste modelo final opcion 1
modeloFinal = glm(formula = BAD ~ LOAN   + CLAGE + CLNO + DEBTINC + 
                 NINQ2 + DEROG2 + DELINQ2, family = binomial, data = setTrain)

summary(modeloFinal)
OR = exp(coef(glm(BAD~LOAN   + CLAGE + CLNO + DEBTINC + 
                    NINQ2 + DEROG2 + DELINQ2,data=setTrain,family=binomial)))[-1]
OR
table(dataModelo$NINQ2)
table(dataModelo$DEROG2)
table(dataModelo$DELINQ2)

# vif relativamente buenos, todos bajo 2
car::vif(modeloFinal)

perf_eva(label=modeloFinal$y, 
         pred=predict(modeloFinal,type="response"), 
         binomial_metric = c("ks"),
         title = "Train Data")
##ks= 0,48
# predeccion para la data de testeo
predictions = predict(modeloFinal, newdata = setTest, type="response")
ROCRpred = prediction(predictions, setTest$BAD)
ROCRperf = performance(ROCRpred, measure = "tpr", x.measure = "fpr")
aucTest = performance(ROCRpred, measure = "auc")

aucTest = aucTest@y.values[[1]]
aucTest
##AUC= 0,76
perf.auc(model = modeloFinal, setTrain, setTest)


###data balanceada###
######################
set.seed(12345)
trainIndex_bal <- createDataPartition(data_balanceada$BAD, times = 1, p = 0.8, list=FALSE)
setTrain_bal <- data_balanceada[trainIndex_bal,]
setTest_bal <- data_balanceada[-trainIndex_bal,]


# ajuste modelo final opcion 2 , con muestra balanceada
modM_bal = glm(BAD~., data = setTrain_bal, family=binomial)
summary(modM_bal)
# Seleccion de variables con stepwise
step(modM_bal)

##modelo stepwise : glm(formula = BAD ~ LOAN + YOJ + CLAGE CLNO+ NINQ2 + 
# DEROG2 + DELINQ2, family = binomial, data = setTrain_bal)
modeloFinal2 = glm(formula = BAD ~ LOAN + YOJ + CLAGE + CLNO + DEBTINC + 
                     NINQ2 + DEROG2 + DELINQ2, family = binomial, data = setTrain_bal)
#Saca JOB + REASON + GARANTIA
summary(modeloFinal2)

car::vif(modeloFinal2)
##TODOS SON BAJO 5

perf_eva(label=modeloFinal2$y, 
         pred=predict(modeloFinal2,type="response"), 
         binomial_metric = c("ks"),
         title = "Train Data")
##KS= 0,45

predictions = predict(modeloFinal2, newdata = setTest_bal, type="response")
ROCRpred = prediction(predictions, setTest_bal$BAD)
ROCRperf = performance(ROCRpred, measure = "tpr", x.measure = "fpr")
aucTest_bal = performance(ROCRpred, measure = "auc")
aucTest_bal = aucTest_bal@y.values[[1]]
aucTest_bal
##0,7989

perf.auc(model = modeloFinal2, setTrain_bal, setTest_bal)

### 5. Evaluar la calidad predictiva del modelo final ###
## Para Modelo 1
# matriz de confusion entrenamiento, con punto de corte 0.5
caret::confusionMatrix(as.factor(setTrain$BAD),as.factor((predict(modeloFinal, newdata=setTrain, type="response")>=0.5)*1))
##acuraccy 0,925
##sensitivity 0,9296
##especificidad=0,7857

# calculamos la matriz de confusion en la muestra de validacion
caret::confusionMatrix(as.factor(setTest$BAD),as.factor((predict(modeloFinal, newdata=setTest, type="response")>=0.5)*1))
##acuraccy 0,9217
##sensitivity 0,9284
##especificidad=0,6667

## Para Modelo 2
# matriz de confusion entrenamiento con data balanceada
caret::confusionMatrix(as.factor(setTrain_bal$BAD),as.factor((predict(modeloFinal2, newdata=setTrain_bal, type="response")>=0.5)*1))
##acuraccy 0,7198
##sensitivity 0,6926
##especificidad=0,7559
# calculamos la matriz de confusion en la muestra de validacion
caret::confusionMatrix(as.factor(setTest_bal$BAD),as.factor((predict(modeloFinal2, newdata=setTest_bal, type="response")>=0.5)*1))

##acuraccy 0,7295
##sensitivity 0,7000
##especificidad=0,7692

# Ahora continuamos con el calculo del WOE para las variables que quedaron de stepwise
# LOAN + JOB + YOJ + CLAGE + CLNO + DEBTINC + NINQ2 + DEROG2 + DELINQ2 + GARANTIA

### 6. Interpretar el peso de los atributos en el modelo final ###

# Calculo del WOE para las variables 
bins = woebin(setTrain[,c("LOAN", "JOB", "YOJ", "CLAGE", "CLNO", "DEBTINC", "NINQ2", "DEROG2", "DELINQ2", "GARANTIA"
                          ,"BAD")], y = 'BAD')
bins

# Calculo del Information Value para.
# Eligiendo todas las variables del modelo, variables con valor de informacion menos a 0.02 debiesen ser descartadas, a 
# excepcion de la variable GARANTIA que tiene un valor explicativo de comportamiento de los Clientes en relacion a 
# a las garantías entregadas en los creditos solicitados.
iv = iv(setTrain[,c("LOAN", "JOB", "YOJ", "CLAGE", "CLNO", "DEBTINC", "NINQ2", "DEROG2", "DELINQ2", "GARANTIA"
                    ,"BAD")], y = 'BAD') %>%
  as_tibble() %>%
  mutate( info_value = round(info_value, 3) ) %>%
  arrange( desc(info_value) )
iv

# Guardamos los valores del WOE de train como variable
modelo3_woe = woebin_ply(setTrain[,c("LOAN" ,"YOJ", "CLNO", "NINQ2", "DEROG2", "DELINQ2" ,"BAD")], bins ) %>%
as_tibble()

# Guardamos los valores del WOE de test como variable
modelo3_woeTest = woebin_ply(setTest[,c("LOAN", "YOJ", "CLNO", "NINQ2", "DEROG2", "DELINQ2",
                                        "BAD")], bins ) %>%
as_tibble()

# SCORECARD - train
modeloWOE2 = glm(BAD~., modelo3_woe, family = 'binomial')
summary(modeloWOE2)

vif(modeloWOE2)

# SCORECARD - test
modeloWOE2Test = glm(BAD~., modelo3_woeTest, family = 'binomial')
summary(modeloWOE2Test)

vif(modeloWOE2Test)

# Model Performance for train data: 
perf_eva(label=modeloWOE2$y, 
         pred=predict(modeloWOE2,type="response"), 
         binomial_metric = c("ks"),
         title = "Train Data")

# Model Performance for test data: 
perf_eva(label=modeloWOE2Test$y, 
         pred=predict(modeloWOE2Test,type="response"), 
         binomial_metric = c("ks"),
         title = "Train Data")

# matriz de confusion entrenamiento
caret::confusionMatrix(as.factor(setTrain$BAD),as.factor((predict(modeloWOE2, newdata = modelo3_woe, type="response")
                                                          >=0.5)*1))

# matriz de confusion testeo
caret::confusionMatrix(as.factor(setTest$BAD),as.factor((predict(modeloWOE2Test, newdata = modelo3_woeTest, type="response")
                                                          >=0.5)*1))

# scorecard
logit = predict(modeloWOE2)

# ver opcion de poner otra escala de puntaje
points0 = 600
odds0 = 50
pdo = 20

my_card = scorecard( bins , modeloWOE2
                     , points0 = points0 
                     , odds0 = 1/odds0 # scorecard wants the inverse
                     , pdo = pdo 
)

my_card

score = scorecard_ply(setTrain, my_card)
score

str(score)
summary(score)

my_card[[2]]

# media del score
mean(score$score)

hist(score$score)
abline(v=mean(score$score),lwd=2,col="red")
