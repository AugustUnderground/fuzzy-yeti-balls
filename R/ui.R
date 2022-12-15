library("plotly")
library("shiny")
library("shinydashboard")
library("shinyWidgets")
library("shinyFiles")

## Side Bar Menus
home_menu  <- menuItem( "Home", tabName = "tab_home"
                      , icon = icon("home"))

ftd_menu <- menuItem( "Fault Detection", tabName = "tab_ftd"
                      , icon = icon("chart-line"))

wkg_menu  <- menuItem( "Conversion Efficency", tabName = "tab_wkg"
                      , icon = icon("chart-line"))

home_tab <- tabItem( "tab_home"
                   , fluidRow( box( title = "Fault Detection Model"
                                  , background = "maroon"
                                  , solidHeader = TRUE
                                  , shinyFilesButton( "file_ftd"
                                                    , label = "Browse"
                                                    , title = "Select Fault Detection Model"
                                                    , multiple = FALSE )
                                  , br(), br()
                                  , verbatimTextOutput("text_ftd") )
                             , box( title = "Energy Convertion Model"
                                  , background = "teal"
                                  , solidHeader = TRUE
                                  , shinyFilesButton( "file_wkg"
                                                    , label = "Browse"
                                                    , title = "Select Energy Conversion Model"
                                                    , multiple = FALSE )
                                  , br(), br()
                                  , verbatimTextOutput("text_wkg") )
                             ))

ftd_tab <- tabItem( "tab_ftd"
                  , fluidRow( box( background = "maroon"
                                 , plotlyOutput("plot_ftd", height = 666) )
                            , box( title = "Controls"
                                 , sliderInput( "slider_Ia", "Ia"
                                               , -800, 800, 0 )
                                 , sliderInput( "slider_Ib", "Ib"
                                              , -800, 800, 0 )
                                 , sliderInput( "slider_Ic", "Ic"
                                              , -800, 800, 0 )
                                 , sliderInput( "slider_Va", "Va"
                                              , -0.5, 0.5, 0 )
                                 , sliderInput( "slider_Vb", "Vb"
                                              , -0.5, 0.5, 0 )
                                 , sliderInput( "slider_Vc", "Vc"
                                              , -0.5, 0.5, 0 ) )))

wkg_tab <- tabItem( "tab_wkg"
                  , fluidRow( box( background = "teal"
                                 , plotlyOutput("plot_wkg", height = 666) )
                            , box( title = "Controls"
                                 , sliderInput( "slider_rl", "Load Resistance"
                                              , 6, 16, 12 )
                                 , sliderInput( "slider_d", "Duty Cycle"
                                              , 1, 100, 50 )
                                 , sliderInput( "slider_vin", "Vin"
                                              , 12, 14, 13 ) )))

header <- dashboardHeader(title = "Fuzzy Yeti Balls")

sidebar <- dashboardSidebar(sidebarMenu(id = "tabs", home_menu, ftd_menu, wkg_menu))

body <- dashboardBody(tabItems(home_tab, ftd_tab, wkg_tab))

ui <- dashboardPage(skin = "black", header, sidebar, body)
