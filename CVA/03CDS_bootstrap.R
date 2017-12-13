print('NOTE: REQUIRES packages credule and xlsx')
library(credule)
library(xlsx)
spd_data = read.xlsx('out11-cds-spreads.xlsx', sheetName = 'Sheet1')
yc_data = read.xlsx('out10-ois_rates.xlsx', sheetName = 'Sheet1')
yc_data <- yc_data[2]
ycrates <- yc_data[-1,]

spd_data <- spd_data[-1,]
yctenor <- spd_data[1]
yctenor <- unlist(yctenor)

cdstenor <- yctenor
cdsspd_1 <- spd_data[2]
cdsspd_1 <- unlist(cdsspd_1)
cdsspd_2 = spd_data[3]
cdsspd_2 <- unlist(cdsspd_2)
RR = 0.40
premfreq = 2
cc_scp_1 = bootstrapCDS(yieldcurveTenor = yctenor, yieldcurveRate = ycrates, cdsTenors = cdstenor,
                     cdsSpreads = cdsspd_1, recoveryRate = RR, numberPremiumPerYear = premfreq)
cc_scp_2 = bootstrapCDS(yieldcurveTenor = yctenor, yieldcurveRate = ycrates, cdsTenors = cdstenor,
                        cdsSpreads = cdsspd_2, recoveryRate = RR, numberPremiumPerYear = premfreq)
write.xlsx(cc_scp_1, "out13-cds_bs.xlsx", sheetName = 'nobump')
write.xlsx(cc_scp_2, "out13-cds_bs.xlsx", sheetName = 'bumped', append = TRUE)
