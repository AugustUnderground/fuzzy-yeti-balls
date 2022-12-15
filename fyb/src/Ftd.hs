{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Neural Network Definition and Training
module Ftd where

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
               } deriving (Generic, Show, T.Parameterized)

-- | Neural Network Weight initialization
instance T.Randomizable NetSpec Net where
    sample NetSpec{..} = Net <$> T.sample (T.LinearSpec   numX 64)
                             <*> T.sample (T.LinearSpec   64  128)
                             <*> T.sample (T.LinearSpec   128  numY)

-- | Neural Network Forward Pass with scaled Data
forward :: Net -> T.Tensor -> T.Tensor
forward Net{..} = T.logSoftmax (T.Dim 1) 
                . T.linear fc2 . T.relu 
                . T.linear fc1 . T.relu 
                . T.linear fc0

-- | Remove Gradient for tracing / scripting
noGrad :: (NN.Parameterized f) => f -> IO f
noGrad net = do
    params  <- mapM (T.detach . T.toDependent) $ NN.flattenParameters net
    params' <- mapM (`T.makeIndependentWithRequiresGrad` False) params
    pure $ NN.replaceParameters net params'

-- | Save Model and Optimizer Checkpoint
saveCheckPoint :: FilePath -> Net -> T.Adam -> IO ()
saveCheckPoint path net opt = do
    T.saveParams net  (path ++ "/ftd-cp.pt")
    T.save (T.m1 opt) (path ++ "/M1.pt")
    T.save (T.m2 opt) (path ++ "/M2.pt")

-- | Load a Saved Model and Optimizer CheckPoint
loadCheckPoint :: FilePath -> NetSpec -> Int -> IO (Net, T.Adam)
loadCheckPoint path spec iter = do
    net <- T.sample spec >>= (`T.loadParams` (path ++ "/ftd-cp.pt"))
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
    loss  = T.nllLoss' trueY predY
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
    loss  = T.nllLoss' trueY predY

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
    scriptPath = path ++ "/ftd-trace.pt"
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

    let dfY  = DF.lookup yKeys dfRaw
        dfX' = DF.lookup xKeys dfRaw
        minX = fst . T.minDim (T.Dim 0) T.RemoveDim . values $ dfX'
        maxX = fst . T.maxDim (T.Dim 0) T.RemoveDim . values $ dfX'
        !dfX = T.scale minX maxX <$> dfX'

    net      <- T.toDevice T.gpu <$> T.sample (NetSpec numInputs numOutputs)
    let opt  =  T.mkAdam 0 0.9 0.999 $ NN.flattenParameters net

    !df  <- DF.shuffleIO . DF.dropNan $ DF.union dfX dfY
    let (!trainX', !validX', !trainY', !validY') 
               = DF.trainTestSplit xKeys yKeys ratio df

    let trainX = T.split batchSize (T.Dim 0) . T.toDevice T.gpu $ trainX'
        trainY = T.split batchSize (T.Dim 0) . T.toDevice T.gpu . T.fromBits
                                             . T.toDType T.Int64 $ trainY'
        validX = T.split batchSize (T.Dim 0) . T.toDevice T.gpu $ validX'
        validY = T.split batchSize (T.Dim 0) . T.toDevice T.gpu . T.fromBits
                                             . T.toDType T.Int64 $ validY'

    (net', opt') <- runEpochs modelPath numEpochs trainX validX trainY validY net opt

    saveCheckPoint modelPath net' opt'

    !net''       <- loadCheckPoint modelPath (NetSpec numInputs numOutputs) numEpochs
                        >>= noGrad . T.toDevice T.cpu . fst

    let predict = T.asBits n . snd . T.maxDim (T.Dim 1) T.RemoveDim 
                . forward net'' . T.scale minX maxX
    
    pure predict
  where
    xKeys      = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
    yKeys      = ["G", "C", "B", "A"]
    numInputs  = length xKeys
    n          = length yKeys
    numOutputs = 2 ^ n
    ratio      = 0.7
    path       = "../data/classData.csv"
    modelPath  = "../models"
    batchSize  = 4
    numEpochs  = 24

test :: (T.Tensor -> T.Tensor) -> IO ()
test mdl = do
    df  <- DF.fromCsv dataPath >>= DF.sampleIO numSamples False

    let x  = DF.lookup xKeys df
        y  = DF.lookup yKeys df
        y' = mdl $ DF.values x
    print x
    print y
    print y'

    pure ()
  where
    numSamples = 10
    xKeys      = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
    yKeys      = ["G", "C", "B", "A"]
    dataPath   = "../data/classData.csv"
