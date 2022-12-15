library("arrow")
library("torch")
library("plotly")
library("RColorBrewer")
library("shiny")
library("shinydashboard")
library("shinyWidgets")
library("shinyFiles")

server <- function(input, output, session) {

    axis_defaults <- list( gridcolor = 'rgb(255,255,255)', showgrid = TRUE
                         , showline = FALSE, showticklabels = TRUE
                         , tickcolor = 'rgb(127,127,127)', ticks = "outside"
                         , zeroline = FALSE )

    palette <- brewer.pal(n = 8, name = "Dark2")
    #palette <- rep(brewer.pal(n = 8, name = "Dark2"), each = 2)

    #data_base <- "./data"
    #fault_data <- read.csv(paste(data_base, "classData.csv", sep = "/"))
    #ceff_data  <- read.csv(paste(data_base, "classData.csv", sep = "/"))
    #fault_range <- as.data.frame(apply(fault_data,2,range))

    model_root <- "./models"
    default_ftd_path <- paste(model_root, "ftd-trace.pt", sep = "/")
    default_wkg_path <- paste(model_root, "wkg-trace.pt", sep = "/")

    ftd_model <- jit_load(default_ftd_path)
    wkg_model <- jit_load(default_wkg_path)

    shinyFileChoose( input, "file_ftd", root = c(root = model_root)
                   , filetypes = c("", "pt") , session = session )

    observe({
        file_info <- parseFilePaths( roots = c(root = model_root)
                                  , selection = input$file_ftd)
        if (length(file_info$datapath) != 0) {
            file <- unname(file_info$datapath)
        } else {
            file <- paste(model_root, "ftd-trace.pt", sep = "/")
        }
        output$text_ftd <- renderPrint({file})
        ftd_model <<- jit_load(file)
    })

    output$plot_ftd <- renderPlotly({
        input <- c( input$slider_Ia, input$slider_Ib, input$slider_Ic
                  , input$slider_Va, input$slider_Vb, input$slider_Vc )

        x <- torch_reshape(torch_tensor(input), list(1, length(input)))
        y <- as_array(ftd_model(x))

        output <- as.data.frame.table(y)

        fig <- plot_ly( x = c("G", "C", "B", "A")
                      , y = y
                      , name = "Fault"
                      , type = "bar" )

        fig %>% layout( title = "Fault Detection"
                      , paper_bgcolor = "rgb(255,255,255)"
                      , plot_bgcolor  = "rgb(229,229,229)"
                      , xaxis = c(list(title = "Conductor"), axis_defaults)
                      , yaxis = c(list(title = "Fault"), axis_defaults))
    })

    shinyFileChoose( input, "file_wkg", root = c(root = model_root)
                   , filetypes = c("", "pt") , session = session )

    observe({
        file_info <- parseFilePaths( roots = c(root = model_root)
                                  , selection = input$file_wkg)
        if (length(file_info$datapath) != 0) {
            file <- unname(file_info$datapath)
        } else {
            file <- paste(model_root, "wkg-trace.pt", sep = "/")
        }
        output$text_wkg <- renderPrint({file})
        wkg_model <<- jit_load(file)
    })

    output$plot_wkg <- renderPlotly({
        # input <- c( input$slider_Ia, input$slider_Ib, input$slider_Ic
        #           , input$slider_Va, input$slider_Vb, input$slider_Vc )

        # x <- torch_reshape(torch_tensor(input), list(1, length(input)))
        # y <- as_array(ftd_model(x))

        # output <- as.data.frame.table(y)

        # fig <- plot_ly( x = c("G", "C", "B", "A")
        #               , y = y
        #               , name = "Fault"
        #               , type = "bar" )

        # fig %>% layout( title = "Fault Detection"
        #               , paper_bgcolor = "rgb(255,255,255)"
        #               , plot_bgcolor  = "rgb(229,229,229)"
        #               , xaxis = c(list(title = "Conductor"), axis_defaults)
        #               , yaxis = c(list(title = "Fault"), axis_defaults))
    })
}
