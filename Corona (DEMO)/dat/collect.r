require ( "data.table" )
require ( "magrittr" )
require ( "manipulate" )
require ( "rworldmap" )

setwd ( paste0 ( "C:/Users/David D'Haese/Documents/_TEACHING/AI_PRINCIPLES/",
                 "COURSE/AI_Principles_Challenges/Corona/dat" ))

dat <- fread ( "corona.tab" )

colnames(dat)

dat$V47
dat$V47 <- NULL

colnames(dat) %>% paste ( collapse = '", "' ) %>% cat

colnames(dat) <-
  c ( "Country", "Province", "Lat", "Lon", "2020-01-22", "2020-01-23",
      "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28",
      "2020-01-29", "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02",
      "2020-02-03", "2020-02-04", "2020-02-05", "2020-02-06", "2020-02-07",
      "2020-02-08", "2020-02-09", "2020-02-10", "2020-02-11", "2020-02-12",
      "2020-02-13", "2020-02-14", "2020-02-15", "2020-02-16", "2020-02-17",
      "2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-22",
      "2020-02-23", "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27",
      "2020-02-28", "2020-02-29", "2020-03-01", "2020-03-02", "2020-03-03" )

dat <- melt ( dat, id.vars = c ( "Country", "Province", "Lat", "Lon" ),
              variable.name = "Date", value.name = "Count" ) 

dat %>% dim

dat_map <- joinCountryData2Map ( dat, joinCode = "NAME",
                               nameJoinColumn = "Country", verbose = TRUE)

dat [ Country == "Mainland China", Country := "China" ]
dat [ Country == "North Macedonia", Country := "Macedonia" ]
dat [ Country == "Others", Province %>% unique ]

dat <- dat [ Country != "Others" ]
dates <- dat$Date %>% levels %>% sort
countries <- dat$Country %>% unique %>% sort

par ( bg = "darkgrey" )

manipulate({
  dat_map <- joinCountryData2Map (
    dat [ Date == date_cur ], joinCode = "NAME",
    nameJoinColumn = "Country" )
  
  dat_map %>%
    mapCountryData ( nameColumnToPlot = "Count", catMethod = "logFixedWidth" )

}, date_cur = do.call ( picker, args = as.list ( dates )))

manipulate({
  dat_cnt_cur <- dat [ Country == cnt_cur, .( Count = sum ( Count )), Date ]
  dat_cnt_cur [, plot ( Count ~ Date, main = cnt_cur )]
}, cnt_cur = do.call ( picker, args = as.list ( countries )))


cnt_cur <- "Iran"
dat_cnt_cur [, Date := as.numeric ( as.Date ( Date ))]

# S = Start, M = Max, H = Half-Max Date, Cf = Hill coefficient ~
#  Steepness at H
formula <- Count ~ S + M * ( Date^Cf / ( H^Cf + Date^Cf ))

model <- nls ( formula, data = dat_cnt_cur,
               start = c ( S = as.numeric ( as.Date ( "2020-02-10" )),
                           M = 2000,
                           H = 10,
                           Cf = 3 ))
