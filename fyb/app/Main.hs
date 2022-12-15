{-# OPTIONS_GHC -Wall #-}

module Main (main) where

import qualified Ftd
import qualified Wkg

main :: IO ()
main = do
    putStrLn "Fuzzy Yeti Balls"
    Ftd.train >>= Ftd.saveTrace modelPath numInputs
    Ftd.loadModel tracePath >>= Ftd.test
    putStrLn "Fuzzy Yeti Balls"
  where
    numInputs  = 6
    modelPath  = "../models"
    tracePath  = modelPath ++ "/ftd-trace.pt"
    -- tracePath  = modelPath ++ "/wkg-trace.pt"
