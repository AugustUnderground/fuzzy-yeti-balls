{-# OPTIONS_GHC -Wall #-}

-- | Hyper Parameters of OpNet
module Torch.Extensions where

import           GHC.Float                      (float2Double)
import           Data.List                      (elemIndex)
import           Data.Maybe                     (fromJust, isJust)
import qualified Torch                     as T
import qualified Torch.Functional.Internal as T ( nan_to_num, powScalar', roll
                                                , linalg_multi_dot, block_diag
                                                , avg_pool2d, sort, where'
                                                , angle )

-- | Type alias for Padding Kernel
type Kernel  = (Int, Int)

-- | Type alias for Padding Stride
type Stride  = (Int, Int)

-- | Type alias for Padding
type Padding = (Int, Int)

-- | GPU
gpu :: T.Device
gpu = T.Device T.CUDA 1

-- | CPU
cpu :: T.Device
cpu = T.Device T.CPU 0

-- | Empty Float Tensor on CPU
empty :: T.Tensor
empty = T.asTensor ([] :: [Float])

-- | The inverse of `log10`
pow10 :: T.Tensor -> T.Tensor
pow10 = T.powScalar' 10.0

-- | 20 * log |x|
db20 :: T.Tensor -> T.Tensor
db20 = T.mulScalar (20.0 :: Float) .  T.log10 . T.abs

-- | Like chainMatmul, but that's deprecated
multiDot :: [T.Tensor] -> T.Tensor
multiDot = T.linalg_multi_dot

-- | Not using snake_case
blockDiag :: [T.Tensor] -> T.Tensor
blockDiag = T.block_diag

avgPool2D' :: Kernel -> Stride -> Padding -> Bool -> Bool -> Int -> T.Tensor 
           -> T.Tensor
avgPool2D' k s p m c d t = T.avg_pool2d t k s p m c d 

avgPool2D :: Kernel -> Stride -> Padding -> T.Tensor -> T.Tensor
avgPool2D k s p t = T.avg_pool2d t k s p True True 1 

-- | Because snake_case sucks and this project uses Float instead of Double
nanToNum :: Float -> Float -> Float -> T.Tensor -> T.Tensor
nanToNum nan' posinf' neginf' self = T.nan_to_num self nan posinf neginf
  where
    nan    = float2Double nan'
    posinf = float2Double posinf'
    neginf = float2Double neginf'

-- | Default limits for `nanToNum`
nanToNum' :: T.Tensor -> T.Tensor
nanToNum' self = T.nan_to_num self nan posinf neginf
  where
    nan    = 0.0     :: Double
    posinf = 2.0e32  :: Double
    neginf = -2.0e32 :: Double

-- | Default limits for `nanToNum` (0.0)
nanToNum'' :: T.Tensor -> T.Tensor
nanToNum'' self = T.nan_to_num self nan posinf neginf
  where
    nan    = 0.0 :: Double
    posinf = 0.0 :: Double
    neginf = 0.0 :: Double

-- | Syntactic Sugar for minDim
minDim' :: T.Dim -> T.Tensor -> T.Tensor
minDim' dim = fst . T.minDim dim T.RemoveDim

-- | Syntactic Sugar for maxDim
maxDim' :: T.Dim -> T.Tensor -> T.Tensor
maxDim' dim = fst . T.maxDim dim T.RemoveDim

-- | Sort descending
sort :: T.Dim -> T.Tensor -> (T.Tensor, T.Tensor)
sort (T.Dim dim) input = T.sort input dim True

-- | Sort ascending
sort' :: T.Dim -> T.Tensor -> (T.Tensor, T.Tensor)
sort' (T.Dim dim) input = T.sort input dim False

-- | Quickly select index
index' :: Int -> T.Tensor -> T.Tensor
index' idx = T.index [idx']
  where
    idx' = T.asTensor' idx $ T.withDType T.Int64 T.defaultOpts

-- | convert Tensor to List of 1D vectors (columnwise)
cols :: T.Tensor -> [T.Tensor]
cols = map T.squeezeAll . T.split 1 (T.Dim 1)

-- | convert Tensor to List of 1D vectors (rowwise)
rows :: T.Tensor -> [T.Tensor]
rows = map T.squeezeAll . T.split 1 (T.Dim 0)

-- | Vertically stack a list of tensors
vstack :: [T.Tensor] -> T.Tensor
vstack = T.stack (T.Dim 0)

-- | Horizontally stack a list of tensors
hstack :: [T.Tensor] -> T.Tensor
hstack = T.stack (T.Dim 1)

-- | Same as fullLike, but with default options
fullLike' :: (T.Scalar a) => T.Tensor -> a -> T.Tensor
fullLike' t = T.full' $ T.shape t

-- | Shorthand diagonal matrix elements as vector
diagonal2D :: T.Tensor -> T.Tensor
diagonal2D = T.diagonal (T.Diag 0) (T.Dim 0) (T.Dim 1) 

-- | Uniformly sampled values in range [lo;hi]
uniformIO :: T.Tensor -> T.Tensor -> IO T.Tensor
uniformIO lo hi = do
    r <- T.randLikeIO' lo
    pure $ (r * (hi - lo)) + lo

-- | Covariance of 2 variables
cov' ::  T.Tensor -> T.Tensor -> T.Tensor
cov' x y = c' / n'
  where
    n' = T.toDType T.Float $ T.asTensor (head $ T.shape x :: Int)
    x' = T.mean x
    y' = T.mean y
    c' = T.sumAll $ (x - x') * (y - y')

-- | Estimates the covariance matrix of the variables given by the `input`
-- matrix, where rows are the variables and columns are the observations.
cov :: T.Tensor -> T.Tensor
cov x = nanToNum'' c
  where
    n  = head $ T.shape x
    n' = T.toDType T.Float $ T.asTensor (last $ T.shape x :: Int)
    μ  = T.meanDim (T.Dim 1) T.KeepDim T.Float x
    x' = x - μ
    cs = [ T.sumDim (T.Dim 1) T.RemoveDim T.Float 
         $ (x' * T.roll x' y 0) / (n' - 1.0)
         | y <- [1 .. n] ]
    c' = T.zeros' [n,n]
    c  = foldl fillC c' . zip (take n . drop 1 $ cycle [0 .. pred n]) $ cs
    fillC :: T.Tensor -> (Int, T.Tensor) -> T.Tensor
    fillC m (i, v) = m'
      where
        o  = T.withDType T.Int64 T.defaultOpts
        z  = T.asTensor' ([0 .. pred n] :: [Int]) o
        s  = T.roll z i 0
        m' = T.indexPut False [z,s] v m

-- | Estimates the Pearson product-moment correlation coefficient matrix of the
-- variables given by the `input` matrix, where rows are the variables and
-- columns are the observations.
corrcoef :: T.Tensor -> T.Tensor
corrcoef x = nanToNum'' . T.clamp (-1.0) 1.0 $ c / c'
  where
    c   = cov x
    dc  = diagonal2D c
    dc' = T.reshape [-1,1] dc
    c'  = T.sqrt . T.abs $ dc * dc'

-- | Mean Absolute Percentage Error
mapeLoss :: T.Tensor -> T.Tensor -> T.Tensor
mapeLoss x y = T.clamp 0.0 100.0 . T.mul 100.0 . T.abs
             . T.meanDim (T.Dim 0) T.RemoveDim T.Float 
             . flip T.div y . T.abs $ (x - y)


-- | Create a boolean mask from a subset of column names
boolMask :: [String] -> [String] -> [Bool]
boolMask sub = map (`elem` sub)

-- | Create a boolean mask Tensor from a subset of column names
boolMask' :: [String] -> [String] -> T.Tensor
boolMask' sub set = T.asTensor' (boolMask sub set) 
                  $ T.withDType T.Bool T.defaultOpts

-- | Scale data to [0,1]
scale :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
scale xMin xMax x = (x - xMin) / (xMax - xMin)

-- | Un-Scale data from [0,1]
scale' :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
scale' xMin xMax x = (x * (xMax - xMin)) + xMin

-- | Apply log10 to masked data
trafo :: T.Tensor -> T.Tensor -> T.Tensor
trafo xMask x = T.where' xMask' (T.log10 $ T.abs x) x
  where
    xMask' = T.logicalAnd xMask $ T.gt x 0.0

-- | Apply pow10 to masked data
trafo' :: T.Tensor -> T.Tensor -> T.Tensor
trafo' xMask x = T.where' xMask (pow10 x) x

-- | Torch.ScriptModule forward pass
evalModel :: T.ScriptModule -> T.Tensor -> T.Tensor
evalModel m x = y
  where
    T.IVTensor y = T.forward m [T.IVTensor x]

-- | Load Script Module
loadModule :: FilePath -> IO (T.Tensor -> T.Tensor)
loadModule modPath = evalModel <$> T.loadScript T.WithoutRequiredGrad modPath

-- | Load a Pickled Tensor from file
loadTensor :: FilePath -> IO T.Tensor
loadTensor path = do
    T.IVTensor t <- T.pickleLoad path
    pure t

-- | Pickle a Tensor and Save to file
saveTensor :: T.Tensor -> FilePath -> IO ()
saveTensor t path = do
    let t' = T.IVTensor t
    T.pickleSave t' path

-- | Create Integer Index for a subset of column names
intIdx :: [String] -> [String] -> [Int]
intIdx set = fromJust . sequence . filter isJust . map (`elemIndex` set)

-- | Create Integer Index Tensor for a subset of column names
intIdx' :: [String] -> [String] -> T.Tensor
intIdx' set sub = T.asTensor' (intIdx set sub) 
                $ T.withDType T.Int64 T.defaultOpts

-- | Find Closest Index
findClosestIdx :: T.Tensor -> Float -> T.Tensor
findClosestIdx t x = T.argmin x' 0 False
  where
    x' = T.abs . T.subScalar x $ T.squeezeAll t

-- | Find closest Index as Int
findClosestIdx' :: T.Tensor -> Float -> Int
findClosestIdx' t x = T.asValue $ findClosestIdx t x

-- | Angle of Complex Tensor in Degrees
angle' :: T.Tensor -> T.Tensor
angle' = T.divScalar (pi :: Float) . T.mulScalar (180.0 :: Float) . T.angle
