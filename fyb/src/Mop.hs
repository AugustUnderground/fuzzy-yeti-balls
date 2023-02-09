{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Multi-Objective Optimization
module Mop where

import Data.List (intersperse)
import Control.Monad (forM_)
import Moo.GeneticAlgorithm.Continuous
import Moo.GeneticAlgorithm.Constraints
import Moo.GeneticAlgorithm.Multiobjective

import qualified Graphics.Vega.VegaLite.Simple as Plt

popSize :: Int
popSize = 100

generations :: Int
generations = 100

ub :: Double
ub = 3.0

lb :: Double
lb = -3.0

n :: Int
n = 2

f1' :: Double -> Double -> Double
f1' x y = 0.5 * ((x ** 2) + (y ** 2)) + sin ((x ** 2) + (y ** 2))

f1 :: [Double] -> Double
f1 [x,y] = f1' x y
f1 _     = undefined

f2' :: Double -> Double -> Double
f2' x y = ((3.0 * x - 2 * y + 4) ** 2) / 8.0 + ((x - y + 1) ** 2) / 27.0 + 15.0

f2 :: [Double] -> Double
f2 [x,y] = f2' x y
f2 _     = undefined

f3' :: Double -> Double -> Double
f3' x y = 1.0 / ((x ** 2) + (y ** 2) + 1) - 1.1 * exp (negate ( (x ** 2) + (y ** 2)))

f3 :: [Double] -> Double
f3 [x,y] = f3' x y
f3 _     = undefined

mop :: MultiObjectiveProblem ([Double] -> Double)
mop = [ (Minimizing, f1), (Minimizing, f2), (Minimizing, f3) ]

constraints :: [Constraint Double Double]
constraints = [ lb .<= (!! n') <=. ub | n' <- [0 .. n - 1] ]

initialize :: Rand [Genome Double]
initialize = getRandomGenomes popSize (replicate n (lb, ub))

step :: StepGA Rand Double
step = stepConstrainedNSGA2bt constraints (degreeOfViolation 1.0 0.0)
       mop unimodalCrossoverRP (gaussianMutate 0.01 0.5)

showCoordinate :: [Double] -> String
showCoordinate xs = foldl1 (++) . intersperse "\t" $ map show xs

run :: IO ()
run = do
  result <- runGA initialize $ loop (Generations generations) step
  let solutions = map takeGenome $ takeWhile ((<= 10.0) . takeObjectiveValue) result
  let ovs = map takeObjectiveValues $ evalAllObjectives mop solutions
  forM_ ovs $ putStrLn . showCoordinate
  let xs = map (!!0) ovs
      ys = map (!!1) ovs
      zs = map (!!2) ovs
  Plt.save "../plots/pareto-x.html" $ Plt.scatter "x" "y" xs ys
  Plt.save "../plots/pareto-y.html" $ Plt.scatter "x" "z" xs zs
  Plt.save "../plots/pareto-z.html" $ Plt.scatter "y" "z" ys zs
