{-# OPTIONS_GHC -Wall #-}

module Main (main) where

import           Lib

main :: IO ()
main = do
    putStrLn "Fuzzy Yeti Balls"
    train >>= saveModel modelPath numInputs
    loadModel modelPath >>= test
    putStrLn "Fuzzy Yeti Balls"
  where
    numInputs  = 6
    modelPath  = "../models/trace.pt"
