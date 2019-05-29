library(ggplot2)
library(plotly)
library(shinyWidgets)
library(GGally)

chrisData = read.csv2('/home/itskov/workspace/lab/DeepSemantic/DeepAEChristian/ChristianOutput_small.csv', sep = ",", stringsAsFactors = F, header = T)

chrisData$Neuron <- factor(chrisData$Neuron)
chrisData$Cond <- factor(chrisData$Cond)

chrisData$LatentVar1 <- as.numeric(chrisData$LatentVar1)
chrisData$LatentVar2 <- as.numeric(chrisData$LatentVar2)
chrisData$LatentVar3 <- as.numeric(chrisData$LatentVar3)
chrisData$LatentVar4 <- as.numeric(chrisData$LatentVar4)


library(shiny)
if (interactive()) {
  neurons  = unique(chrisData$Neuron)
  conds = unique(chrisData$Cond)
  #strainTypes = unique(eddysData$Strain)
  
  ui <- fluidPage(
    sidebarLayout(
      sidebarPanel(
        selectInput('NeuronName', 'Neuron Name', neurons, selected = NULL, multiple = F,
                    selectize = TRUE, width = NULL, size = NULL),    
        
        selectInput('CondName', 'Conds', conds, selected = NULL, multiple = T,
                    selectize = TRUE, width = NULL, size = NULL)
        

        

        #electInput('shapeToggle', 'Shape indicates:', c('Neuron','Chem','Strain'), selected = NULL, multiple = F,
        #           selectize = TRUE, width = NULL, size = NULL)     
        
        
        #materialSwitch('colorToggle', label = 'Color For Neuron / Chem', value = FALSE,
        #               status = "default", right = FALSE, inline = FALSE, width = NULL)
      ),
      
      mainPanel(
        verbatimTextOutput("x"),
        plotlyOutput('plotOutput1')
      )
    )
  )
  server <- function(input, output) {
    updatedData <- reactive({ newData <- chrisData
    newData <- newData[newData$Neuron %in% input$NeuronName,]
    newData <- newData[newData$Cond %in% input$CondName,]
    return (newData)})
    
    #colorField <- reactive({
    #                        if (input$colorToggle == F) {
    #                           colorField = 'Neuron'
    #                        } else {
    #                          colorField = 'Chem'
    #                        }
    #  
    #                        return(colorField)
    #                        })
    
    output$x = renderText(updatedData())
    
    output$plotOutput1 <- renderPlot(ggpairs(chrisData[,5:8]))
    #output$plotOutput1 <- renderPlot(ggpairs(chrisData[,5:8], aes(colour = updatedData()$Cond, alpha = 0.4), upper = 'blank'))
    
    #output$plotOutput1<-renderPlotly({ggplot(data=updatedData()) + 
    #    geom_point(aes_string(x='LatNeuron1', y='LatNeuron2', shape='Step', text='Name', color=input$colorToggle), size=4) })
    #output$plotOutput2<-renderPlotly({ggplot(data=updatedData()) + 
    #    geom_point(aes_string(x='LatNeuron2', y='LatNeuron3', shape='Step', text='Name', color=input$colorToggle), size=4) })
    #output$plotOutput3<-renderPlotly({ggplot(data=updatedData()) + 
    #    geom_point(aes_string(x='LatNeuron1', y='LatNeuron3', shape='Step', text='Name', color=input$colorToggle), size=4) })
    
  }
  shinyApp(ui, server)
}
