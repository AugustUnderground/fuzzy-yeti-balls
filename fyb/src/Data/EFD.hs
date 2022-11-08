{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Data.EFD ( EFD (..)
                , FeatureDim
                , LabelDim
                , initEFD
                ) where

import           GHC.TypeLits
import           Control.DeepSeq             (force)
import qualified Torch                 as UT
import qualified Torch.Typed           as T
import           Data.Kind
import qualified Data.Frame            as DF
import qualified Data.Set              as S
import qualified Data.ByteString       as BS
import qualified Data.ByteString.Char8 as CS

type FeatureDim = 6
type LabelDim   = 16

newtype EFD (m :: Type -> Type) (device :: (T.DeviceType, Nat)) (batchSize :: Nat) 
        = EFD { efdData :: DF.DataFrame UT.Tensor }

instance (KnownNat batchSize, T.KnownDevice device, Applicative m)
    => UT.Dataset m (EFD m device batchSize) Int
        ( T.Tensor device 'T.Float '[batchSize, FeatureDim]
        , T.Tensor device 'T.Int64 '[batchSize] )
 where
   getItem EFD{..} ix = let bs       = T.natValI @batchSize 
                            idx      = [ix * bs .. (ix + 1) * bs - 1]
                            inputs'  = UT.toDevice (UT.Device UT.CPU 0)
                                     . selectFeatures idx $ efdData
                            outputs' = UT.toDevice (UT.Device UT.CPU 0)
                                     . selectLabels idx $ efdData
                            (inputs :: T.CPUTensor 'T.Float '[batchSize, FeatureDim]) 
                                     = T.UnsafeMkTensor inputs'
                            (outputs :: T.CPUTensor 'T.Int64 '[batchSize]) 
                                     = T.UnsafeMkTensor outputs'
                         in pure(T.toDevice @device inputs, T.toDevice @device outputs)
   keys EFD{..}       = let n = DF.nRows efdData `div` (T.natValI @batchSize) - 1
                         in S.fromList [0 .. n]

selectFeatures :: [Int] -> DF.DataFrame UT.Tensor -> UT.Tensor
selectFeatures idx = DF.values . DF.rowSelect' idx . DF.lookup featureKeys
  where
    featureKeys = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

selectLabels :: [Int] -> DF.DataFrame UT.Tensor -> UT.Tensor
selectLabels idx df = labels'
  where
    expt    = UT.asTensor ([[2.0 ^ e | e <- [0 .. length labelKeys - 1]]] :: [[Float]])
    labels' = UT.sumDim (UT.Dim 1) UT.RemoveDim UT.Int64 . UT.mul expt 
            . DF.values . DF.rowSelect' idx $ DF.lookup labelKeys df
    labelKeys = ["G", "C", "B", "A"]

initEFD :: IO (EFD IO device batchSize, EFD IO device batchSize)
initEFD = do
    (header':values') <- CS.lines <$> BS.readFile path
    let header = force . CS.unpack <$> CS.split ','  header'
        values = force . fmap (read . CS.unpack) . CS.split ',' 
                    <$> values' :: [[Float]]
        df     = DF.DataFrame header $ UT.asTensor values
    
    (dfT,dfV) <- DF.splitIO' ratio df
    pure (EFD dfT, EFD dfV)
  where
    ratio = 0.7
    path = "./data/classData.csv"
