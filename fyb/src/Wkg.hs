{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Neural Network Definition and Training
module Wkg where

import           GHC.Generics
import qualified Torch            as T
import qualified Torch.NN         as NN
import qualified Torch.Extensions as T
import           Data.Frame       as DF

-- | Neural Network Specification
data NetSpec = NetSpec { numX   :: Int       -- ^ Number of input neurons
                       , numY   :: Int       -- ^ Number of output neurons
                       } deriving (Show, Eq)

-- | Network Architecture
data Net = Net { fc0 :: T.Linear
               , fc1 :: T.Linear
               , fc2 :: T.Linear
               , fc3 :: T.Linear
               , fc4 :: T.Linear
               } deriving (Generic, Show, T.Parameterized)

-- | Neural Network Weight initialization
instance T.Randomizable NetSpec Net where
    sample NetSpec{..} = Net <$> T.sample (T.LinearSpec   numX 64)
                             <*> T.sample (T.LinearSpec   64   128)
                             <*> T.sample (T.LinearSpec   128  64)
                             <*> T.sample (T.LinearSpec   54   32)
                             <*> T.sample (T.LinearSpec   32   numY)

-- | Neural Network Forward Pass with scaled Data
forward' :: Net -> T.Tensor -> T.Tensor
forward' Net{..} = T.linear fc4 . T.relu 
                 . T.linear fc3 . T.relu 
                 . T.linear fc2 . T.relu 
                 . T.linear fc1 . T.relu 
                 . T.linear fc0

-- | Calculate η after forward pass
forward :: Net -> T.Tensor -> T.Tensor
forward net x = y
  where
    y'   = forward' net x
    pIn  = T.select 1 0 y'
    pOut = T.select 1 1 y'
    n    = T.div pOut pIn
    y    = T.cat (T.Dim 1) [ y', n ]

-- | Remove Gradient for tracing / scripting
noGrad :: (NN.Parameterized f) => f -> IO f
noGrad net = do
    params  <- mapM (T.detach . T.toDependent) $ NN.flattenParameters net
    params' <- mapM (`T.makeIndependentWithRequiresGrad` False) params
    pure $ NN.replaceParameters net params'

-- | Save Model and Optimizer Checkpoint
saveCheckPoint :: FilePath -> Net -> T.Adam -> IO ()
saveCheckPoint path net opt = do
    T.saveParams net  (path ++ "/model.pt")
    T.save (T.m1 opt) (path ++ "/M1.pt")
    T.save (T.m2 opt) (path ++ "/M2.pt")

-- | Load a Saved Model and Optimizer CheckPoint
loadCheckPoint :: FilePath -> NetSpec -> Int -> IO (Net, T.Adam)
loadCheckPoint path spec iter = do
    net <- T.sample spec >>= (`T.loadParams` (path ++ "/model.pt"))
    m1' <- T.load (path ++ "/M1.pt")
    m2' <- T.load (path ++ "/M2.pt")
    let opt = T.Adam 0.9 0.999 m1' m2' iter
    pure (net, opt)

-- | Trace and Return a Script Module
traceModel :: Int -> (T.Tensor -> T.Tensor) -> IO T.ScriptModule
traceModel num predict = do
        T.trace name "forward" fun data' >>= T.toScriptModule
  where
    fun   = mapM (T.detach . predict)
    name  = "fyb"
    data' = [T.ones' [1, num]]

-- | Save a Traced ScriptModule
saveInferenceModel :: FilePath -> T.ScriptModule -> IO ()
saveInferenceModel path model = T.saveScript model path

-- | Load a Traced ScriptModule
loadInferenceModel :: FilePath -> IO T.ScriptModule
loadInferenceModel = T.loadScript T.WithoutRequiredGrad

-- | Run one Update Step
trainStep :: T.Tensor -> T.Tensor -> Net -> T.Adam -> IO (Net, T.Adam, T.Tensor)
trainStep trueX trueY net opt = do
    (net', opt') <- T.runStep net opt loss alpha
    pure (net', opt', loss)
  where
    predY = forward net trueX
    loss  = T.mseLoss trueY predY
    alpha = 1.0e-3

-- | Run through all Batches performing an update for each
trainingEpoch :: [T.Tensor]  -> [T.Tensor] -> [T.Tensor] -> Net -> T.Adam 
              -> IO (Net, T.Adam, T.Tensor)
trainingEpoch   _      []   losses net opt = pure (net, opt, losses')
  where
    losses' = T.cat (T.Dim 0) . map (T.reshape [-1]) $ losses
trainingEpoch   []     _    losses net opt = pure (net, opt, losses')
  where
    losses' = T.cat (T.Dim 0) . map (T.reshape [-1]) $ losses
trainingEpoch (x:xs) (y:ys) losses net opt = do
    trainStep x y net opt >>= trainingEpoch' 
  where
    trainingEpoch' (n, o, l) = trainingEpoch xs ys (l:losses) n o

-- | Run one Update Step
validStep :: T.Tensor -> T.Tensor -> Net -> IO T.Tensor
validStep trueX trueY net = T.detach loss
  where
    predY = forward net trueX
    loss  = T.l1Loss T.ReduceMean trueY predY

-- | Run through all Batches performing an update for each
validationEpoch :: [T.Tensor] -> [T.Tensor] -> Net -> [T.Tensor] -> IO T.Tensor
validationEpoch   _      []   _   losses = pure . T.cat (T.Dim 0) 
                                         . map (T.reshape [-1]) $ losses
validationEpoch   []     _    _   losses = pure . T.cat (T.Dim 0) 
                                         . map (T.reshape [-1]) $ losses
validationEpoch (x:xs) (y:ys) net losses = validStep x y net 
                                            >>= validationEpoch xs ys net . (:losses) 

-- | Run Training and Validation for a given number of Epochs
runEpochs :: FilePath -> Int -> [T.Tensor] -> [T.Tensor] -> [T.Tensor] 
          -> [T.Tensor] -> Net -> T.Adam -> IO (Net, T.Adam)
runEpochs path 0     _       _       _       _       net opt = do
    saveCheckPoint path net opt
    pure (net, opt)
runEpochs path epoch trainXs validXs trainYs validYs net opt = do
    putStrLn $ "Epoch " ++ show epoch ++ ":"
    (net', opt', mse) <- trainingEpoch trainXs trainYs [] net opt
    putStrLn $ "\tTraining Loss: " ++ show (T.mean mse)

    mae  <- validationEpoch validXs validYs net' []
    putStrLn $ "\tValidataion Loss: " ++ show (T.mean mae)

    saveCheckPoint path net' opt'
    runEpochs path epoch' trainXs validXs trainYs validYs net' opt'
  where
    epoch' = epoch - 1

saveTrace :: FilePath -> Int -> (T.Tensor -> T.Tensor) -> IO ()
saveTrace path num mdl = do
    data' <- (:[]) <$> T.randIO' [10, num]
    rm <- T.trace name "forward" fun data' 
    T.setRuntimeMode rm T.Eval
    T.toScriptModule rm >>= flip T.saveScript scriptPath
  where
    fun        = mapM (pure . mdl)
    name       = "fyb"
    scriptPath = path ++ "/trace.pt"
    -- data'     = [T.ones' [1, num]]

loadModel :: FilePath -> IO (T.Tensor -> T.Tensor)
loadModel path = do
    mdl <- T.loadScript T.WithoutRequiredGrad path
    let fun x = let T.IVTensor y = T.runMethod1 mdl "forward" $ T.IVTensor x
                  in y
    pure fun

train :: IO (T.Tensor -> T.Tensor)
train = do
    dfRaw <- DF.fromCsv path

    let dfY' = DF.lookup yKeys dfRaw
        dfX' = T.trafo xMask <$> DF.lookup xKeys dfRaw

        minX = T.max' xMin' . fst . T.minDim (T.Dim 0) T.RemoveDim . values $ dfX'
        maxX = T.min' xMax' . fst . T.maxDim (T.Dim 0) T.RemoveDim . values $ dfX'
        minY = T.max' yMin' . fst . T.minDim (T.Dim 0) T.RemoveDim . values $ dfY'
        maxY = T.min' yMax' . fst . T.maxDim (T.Dim 0) T.RemoveDim . values $ dfY'

        !dfX = T.scale minX maxX <$> dfX'
        !dfY = T.scale minY maxY <$> dfY'

    net      <- T.toDevice T.gpu <$> T.sample (NetSpec numInputs numOutputs)
    let opt  =  T.mkAdam 0 0.9 0.999 $ NN.flattenParameters net

    !df  <- DF.shuffleIO . DF.dropNan $ DF.union dfX dfY
    let (!trainX', !validX', !trainY', !validY') 
               = DF.trainTestSplit xKeys yKeys ratio df

    let trainX = T.split batchSize (T.Dim 0) . T.toDevice T.gpu $ trainX'
        trainY = T.split batchSize (T.Dim 0) . T.toDevice T.gpu $ trainY'
        validX = T.split batchSize (T.Dim 0) . T.toDevice T.gpu $ validX'
        validY = T.split batchSize (T.Dim 0) . T.toDevice T.gpu $ validY'

    (net', opt') <- runEpochs modelPath numEpochs trainX validX trainY validY net opt

    saveCheckPoint modelPath net' opt'

    -- !net'' <- T.toDevice T.cpu <$> noGrad net'
    !net''       <- loadCheckPoint modelPath (NetSpec numInputs numOutputs) numEpochs
                        >>= noGrad . T.toDevice T.cpu . fst

    let predict = T.scale' minY maxY . forward net'' . T.scale minX maxX . T.trafo xMask
    
    pure predict
  where
    xKeys      = ["f", "Rl", "d", "Vin"] -- , "Vout", "Iout"]
    yKeys      = ["n", "Pin", "Pout"]
    xMin'      = T.asTensor ([100, 5, 0, 10] :: [Float])
    xMax'      = T.asTensor ([100000, 20, 100, 15] :: [Float])
    yMin'      = T.asTensor ([0] :: [Float])
    yMax'      = T.asTensor ([100] :: [Float])
    xMask      = T.boolMask' ["f"] xKeys
    numInputs  = length xKeys
    numOutputs = length yKeys - 1
    ratio      = 0.7
    path       = "../data/wirkungsgrad.csv"
    modelPath  = "../models"
    batchSize  = 4
    numEpochs  = 10

test :: (T.Tensor -> T.Tensor) -> IO ()
test mdl = do
    df  <- DF.fromCsv dataPath >>= DF.sampleIO numSamples False

    let x  = DF.lookup xKeys df
        y  = DF.lookup yKeys df
        y' = mdl $ DF.values x
    print x
    print y
    print y'
  where
    numSamples = 10
    xKeys      = ["f", "Rl", "d", "Vin"] -- , "Vout", "Iout"]
    yKeys      = ["n", "Pin", "Pout"]
    dataPath   = "../data/wirkungsgrad.csv"
