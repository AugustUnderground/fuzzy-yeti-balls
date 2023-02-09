{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Simple Plot functions for convenience
module Graphics.Vega.VegaLite.Simple (scatter, save) where

import           System.Directory               (canonicalizePath)
import           Graphics.Vega.VegaLite as VL
import qualified Data.Text              as TXT

defaultHeight :: Double
defaultHeight = 500

defaultWidth :: Double
defaultWidth = 800

scatter :: String -> String -> [Double] -> [Double] -> VegaLite
scatter xLabel yLabel xVec yVec = toVegaLite [ dat [], enc [], mrk []
                                             , height defaultHeight
                                             , width defaultWidth ]
  where
    xlb = TXT.pack xLabel
    ylb = TXT.pack yLabel
    dat = dataFromColumns [] . dataColumn xlb (Numbers xVec)
                             . dataColumn ylb (Numbers yVec)
    enc = encoding . position X [ PName xlb, PmType Quantitative ]
                   . position Y [ PName ylb, PmType Quantitative ]
    mrk = mark Point  

save :: FilePath -> VegaLite -> IO ()
save = toHtmlFile
