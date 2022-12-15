library("optparse")

# library("arrow")
# library("torch")
# library("plotly")
# library("RColorBrewer")
# library("shiny")
# library("shinydashboard")
# library("shinyWidgets")

opts <- list( make_option( c("-m", "--model")
                         , type = "character"
                         , default = "../models/trace.pt"
                         , help = "Torch Trace Model"
                         , metavar = "filepath" )
            , make_option( c("-H", "--host")
                         , type = "character"
                         , default = "0.0.0.0"
                         , help = "Host address"
                         , metavar = "HOST" )
            , make_option( c("-P", "--port")
                         , type = "integer"
                         , default = 6006
                         , help = "Host address port"
                         , metavar = "PORT" ) )

opt_parser <- OptionParser(option_list = opts);
args <- parse_args(opt_parser)

source("./R/ui.R")
source("./R/server.R")

if (interactive()) {
    shinyApp(ui, server)
} else {
    runApp( list(ui = ui, server = server)
          , host = args$host, port = args$port # , model_path = args$model
          , launch.browser = FALSE )
}
