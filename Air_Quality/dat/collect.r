require ( "data.table" )
require ( "magrittr" )
require ( "manipulate" )

setwd ( "C:/Users/David D'Haese/Documents/_TEACHING/AI_PRINCIPLES/COURSE/AI_Principles_Challenges/Air_Quality/dat" )

dat <- fread ( "AirQualityUCI.csv" )

colnames ( dat ) <- c ( "Date", "Time", "CO_Ref", "CO", "NMHC_Ref", "C6H6",
                        "NMHC", "NOX_Ref", "NOX", "NO2", "NO2", "O3", "T", "RH",
                        "AH" )

dat <- dat[ NOX > 0]
     
# Slow
# pairs(dat[,.(`CO(GT)`, `PT08.S1(CO)`, `NMHC(GT)`, `C6H6(GT)`,
#              `PT08.S2(NMHC)`, AH, RH, `T`, `PT08.S5(O3)`,
#              `PT08.S4(NO2)`, `NO2(GT)`, NOX,
#              `NOx(GT)`)], pch=".", gap=0, col=rgb(0, 0, 0, .1))

dat[ NOX > 0, scatter.smooth( O3 ~ NOX, pch = ".", col="steelblue",
                    lpars = list ( col = "red", lwd = 2, lty=3 ))]
dat_noz <- dat[ NOX > 0 ]
deg_max <- 10
cols <- rainbow ( deg_max )

x_pred <- seq ( 400, 2600, 10)

for (deg in 1:deg_max) {
  model <- lm ( O3 ~ poly(NOX, deg ), data = dat_noz )
  y_pred <- predict ( model, newdata = data.table(NOX=x_pred) )
  dat_noz [, lines (y_pred ~ x_pred, col=cols[deg], lwd=2 )] 
}

legend ( "topright",
         legend = c ( "loess", paste ( "Poly", 1:deg_max)),
         lwd = 2, lty=c(3, rep(1, deg_max)),
         col=c ("red", rainbow(deg_max)), ncol = 2 )

# --- for coach

par ( bg = "darkgrey" )

dat[, plot ( O3 ~ NOX, pch = ".", main = "Air quality\nrelation between O3 and NOx" )]

for (deg in 1:deg_max) {
  model <- lm ( O3 ~ poly(NOX, deg ), data = dat_noz )
  y_pred <- predict ( model, newdata = data.table(NOX=x_pred) )
  dat_noz [, lines (y_pred ~ x_pred, col=cols[deg], lwd=2 )] 
}


legend ( "topright", legend = paste ( "Poly", 1:deg_max), lwd = 2,
         lty= rep ( 1, deg_max ), col = rainbow(deg_max), ncol = 2 )


dat[ , plot ( O3 ~ NOX, main = "Air quality\nrelation between O3 and NOx" , type="n")]

train <- runif (nrow ( dat )) < .99
test <- train %>% not

dat[ train, points ( O3 ~ NOX, pch = "." )]
dat[ test, points ( O3 ~ NOX, col = "cyan" )]
