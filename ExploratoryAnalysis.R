
library(DataCombine)
library(ggplot2)
library(data.table)
library(readr)
library(plotly)

path.to.data <- 'D:\\Downloads\\mouse 4 (24705), first 8min interval'

files <- list.files( path.to.data, full.names = T)

spiketrain_1.5 <- read.csv( files[6])

#Plot spiketrain and deconvolved
thisMouse = 17797

sessionNumber = mouseInfo$sessionNumber[mouseInfo$MouseID == thisMouse]
scanIndex = mouseInfo$scanIndex[mouseInfo$MouseID == thisMouse]

# Spiketrains
#spiketrain_0.5 = as.data.frame(read.general(paste0("m_", "NewThreshold_0.5dc"), thisMouse))
#spiketrain_1.5 = as.data.frame(read.general(paste0("m_", "NewThreshold_1.5dc"), thisMouse))
#spiketrain_2 = as.data.frame(read.general(paste0("m_", "NewThreshold_2dc"), thisMouse))
#spiketrain_3 = as.data.frame(read.general(paste0("m_", "NewThreshold_3dc"), thisMouse))

# df/f
spiketrain_1.5 <- as.data.frame(fread(paste0(files[6])))
# Deconvolved
deconvolved = as.matrix(fread(paste0("~/Data/General/mouse_",
                                     thisMouse, "/s",
                                     sessionNumber, "_idx",
                                     scanIndex, "_deconvolved.csv")))
# Fluorescence
fluorescense = as.matrix(fread(paste0("~/Data/General/mouse_",
                                      thisMouse, "/s",
                                      sessionNumber, "_idx",
                                      scanIndex, "_fluorescense.csv")))

#Set seed, for picking the same random neurons to plot each time
set.seed(1)

sampleNeurons <- sample(colnames(spiketrain_1.5), 10)

#sampleNeurons = sample(mouseInfo$numberOfNeurons[mouseInfo$MouseID == thisMouse], 10)

#Reset seed
set.seed(seed = NULL)

for (neuronID in sampleNeurons) {
  
  nameOfPlot = paste0("~/ITE/SideridisLog/Results/firingEventsVsDeconvolved/mouse-",
                      thisMouse, "_neuron-",
                      neuronID, ".rds")
  
  if(!file.exists(nameOfPlot)){
    
    thisdf = data.frame("frames" = seq(nrow(spiketrain_1.5)),
                        "spiketrain_1.5" = ifelse(spiketrain_1.5[,neuronID] == 1, 0, NA))
                        # "df_f" = df_f[,neuronID],
                        # "deconvolved" = deconvolved[,neuronID],
                        # "fluorescense" = fluorescense[,neuronID],
                        # "spiketrain_0.5" = ifelse(spiketrain_0.5[,neuronID] == 1, 0, NA),
                        # "spiketrain_1.5" = ifelse(spiketrain_1.5[,neuronID] == 1, 0, NA),
                        # "spiketrain_2" = ifelse(spiketrain_2[,neuronID] == 1, 0, NA),
                        # "spiketrain_3" = ifelse(spiketrain_3[,neuronID] == 1, 0, NA))
                        # 
    fig <- plot_ly(thisdf, x = ~frames) 
    
    fig <- fig %>% add_trace(y = ~deconvolved,
                             name = 'Deconvolved',
                             mode = 'lines',
                             type = 'scatter')
    
    fig <- fig %>% add_trace(y = ~fluorescense,
                             name = 'Fluorescense',
                             mode = 'lines',
                             visible = "legendonly",
                             type = 'scatter')
    
    fig <- fig %>% add_trace(y = ~df_f*100,
                             name = 'df/f (x100)',
                             mode = 'lines',
                             visible = "legendonly",
                             type = 'scatter')
    
    fig <- fig %>% add_trace(y = ~spiketrain_0.5,
                             name = 'Firing Event (0.5dc)',
                             mode = 'markers',
                             type = 'scatter',
                             marker = list(size=10))
    
    fig <- fig %>% add_trace(y = ~spiketrain_1.5,
                             name = 'Firing Event (1.5dc)',
                             mode = 'markers',
                             visible = "legendonly",
                             type = 'scatter',
                             marker = list(size=10))
    
    fig <- fig %>% add_trace(y = ~spiketrain_2,
                             name = 'Firing Event (2dc)',
                             mode = 'markers',
                             visible = "legendonly",
                             type = 'scatter',
                             marker = list(size=10))
    
    fig <- fig %>% add_trace(y = ~spiketrain_3,
                             name = 'Firing Event (3dc)',
                             mode = 'markers',
                             visible = "legendonly",
                             type = 'scatter',
                             marker = list(size=10))
    
    fig <- fig %>% layout(barmode = "overlay",
                          xaxis = list(title = "Frames",
                                       titlefont = list(size = 30),
                                       tickfont = list(size = 25)),
                          yaxis = list(title = "",
                                       titlefont = list(size = 30),
                                       tickfont = list(size = 25)),
                          legend = list(font = list(size = 20)),
                          title = list(text=paste0("Mouse ", "mouseInfo$mouseName[mouseInfo$MouseID==thisMouse]",
                                                   ", Neuron: ", neuronID), font=list(size=10), y=0.99),
                          autosize = F, width = 1000, height=400)
    
    saveRDS(fig, nameOfPlot)
  }
}