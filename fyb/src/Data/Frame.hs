{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}

-- | A module for storing Tabular Data as Tensors
module Data.Frame where

import           Control.DeepSeq                       (force)
import           Data.List                             (elemIndex)
import           Prelude                        hiding (lookup, concat)
import           Data.Maybe
import qualified Data.Map                  as M
import qualified Data.ByteString           as BS
import qualified Data.ByteString.Char8     as CS
import qualified Torch                     as T
import qualified Torch.Extensions          as T
import qualified Torch.Functional.Internal as T        (isinf, argsort)

-- | Type Alias for Data Set, where fst = Features and snd = Labels
data DataSet a = DataSet { xData :: DataFrame a
                         , yData :: DataFrame a
                         } deriving (Show)

-- | Split Data Frame into Set
split :: [String] -> [String] -> DataFrame T.Tensor -> DataSet T.Tensor
split px py df = DataSet { xData = lookup px df
                         , yData = lookup py df }

-- | Split Dataframe in Train and Test set according to given ratio < 1
split' :: Float -> DataFrame T.Tensor -> (DataFrame T.Tensor, DataFrame T.Tensor)
split' ratio df = (dfT, dfV)
  where
    num  = nRows df
    numT = ceiling . (ratio *) . fromIntegral $ num :: Int
    idx = T.arange' 0 num 1
    [idxT, idxV] = T.toDType T.Int64 <$>  T.split numT (T.Dim 0) idx
    dfT = rowSelect idxT df
    dfV = rowSelect idxV df

-- | Shuffle before splitting into Train and Test set
splitIO' :: Float -> DataFrame T.Tensor -> IO (DataFrame T.Tensor, DataFrame T.Tensor)
splitIO' ratio df = split' ratio <$> shuffleIO df

-- | Combine x and y Data into Frame
combine :: DataSet T.Tensor -> DataFrame T.Tensor
combine DataSet{..} = xData `union` yData

-- | Sample same indices from both x and y data set
sampleIO' :: Int -> Bool -> DataSet T.Tensor -> IO (DataSet T.Tensor)
sampleIO' num rep DataSet{..} = do
    idx <- sampleIdx num len rep
    pure $ DataSet (rowSelect idx xData) (rowSelect idx yData)
  where 
    len = nRows xData

-- | Push new data into Frame
push' :: Int -> DataFrame T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
push' lim df df' = DataFrame cols vals
  where
    v     = values df
    v'    = values df'
    vals' = T.cat (T.Dim 0) [v,v']
    lim'  = min (pred lim) . pred . head . T.shape $ vals'
    vals  = T.indexSelect' 0 [0 .. lim'] vals'
    cols  = columns df

-- | Push new data into set
push :: Int -> [String] -> [String] -> DataSet T.Tensor -> DataFrame T.Tensor 
     -> DataSet T.Tensor
push lim px py DataSet{..} df = DataSet xData' yData'
  where
    xData' = push' lim xData $ lookup px df
    yData' = push' lim yData $ lookup py df

-- | Apply functions to tensors in x and y Data Set
apply' :: (a -> b) -> (a -> b) -> DataSet a -> DataSet b
apply' fa fb DataSet{..} = DataSet (fmap fa xData) (fmap fb yData)

-- | @apply f == apply' f f@
apply :: (a -> b) -> DataSet a -> DataSet b
apply f = apply' f f

-- | map function over data set
dmap :: (DataFrame a -> DataFrame b) -> DataSet a -> DataSet b
dmap f DataSet{..} = DataSet (f xData) (f yData)

-- | Data Frame
data DataFrame a = DataFrame { columns :: [String] -- ^ Unique Column Identifier
                             , values  :: a        -- ^ Data
                             } deriving (Show)

-- | Functor instance for Mapping over values
instance Functor DataFrame where
  fmap f (DataFrame c v) = DataFrame c (f v)

-- | Rename columns (must align)
rename :: [String] -> DataFrame a -> DataFrame a
rename columns' df = df { columns = columns'}

-- | Number of Rows in DataFrame
nRows :: DataFrame T.Tensor -> Int
nRows DataFrame{..} = head $ T.shape values

-- | Number of Columns in DataFrame
nCols :: DataFrame T.Tensor -> Int
nCols DataFrame{..} = length columns

-- | Shape of Data in Frame
shape :: DataFrame T.Tensor -> [Int]
shape DataFrame{..} = T.shape values

-- | Load Tensor from file and construct DataFrame with given Header
fromFile' :: [String] -> FilePath -> IO (DataFrame T.Tensor)
fromFile' cols path = do
    vals <- T.loadTensor path
    pure $ DataFrame cols vals

-- | Load columns file and Tensor
fromFile :: FilePath -> FilePath -> IO (DataFrame T.Tensor)
fromFile cPath vPath = do
    cols <- words <$> readFile cPath 
    fromFile' cols vPath

-- | Load CSV into DataFrame
fromCsv :: FilePath -> IO (DataFrame T.Tensor)
fromCsv path = do
    (header':values') <- CS.lines <$> BS.readFile path
    let header = force . CS.unpack . (\a -> fromMaybe a (CS.stripSuffix "\r" a))
                    <$> CS.split ','  header'
        values = force . fmap (read . CS.unpack) . CS.split ',' 
                    <$> values' :: [[Float]]
    pure . DataFrame header $ T.asTensor values
    
-- | Load from files with identical prefix
fromPrefix :: String -> IO (DataFrame T.Tensor)
fromPrefix pf = fromFile cPath vPath
  where
    cPath = pf ++ ".txt"
    vPath = pf ++ ".pt"

-- | Save DataFrame to files
toFile :: String -> DataFrame T.Tensor -> IO ()
toFile prefix DataFrame{..} = do
    T.saveTensor values tPath
    writeFile cPath $ unwords columns
  where
    tPath = prefix ++ ".pt"
    cPath = prefix ++ ".txt"

-- | Look up columns
lookup :: [String] -> DataFrame T.Tensor -> DataFrame T.Tensor
lookup cols DataFrame{..} = DataFrame cols vals
  where
    dev  = T.device values
    idx  = T.asTensor' (T.intIdx columns cols) 
         . T.withDevice dev . T.withDType T.Int64 $ T.defaultOpts
    vals = T.indexSelect 1 idx values

-- | Shorthand for looking up a single key
(??) :: DataFrame T.Tensor -> String -> T.Tensor
(??) df key = values $ lookup [key] df

-- | Lookup Rows by index
rowSelect' :: [Int] -> DataFrame T.Tensor -> DataFrame T.Tensor
rowSelect' idx df@DataFrame{..} = rowSelect idx' df
  where
    dev     = T.device values 
    idx'    = T.asTensor' idx
            . T.withDevice dev . T.withDType T.Int64 $ T.defaultOpts

-- | Lookup Rows by index
rowSelect :: T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
rowSelect idx DataFrame{..} = DataFrame columns values'
  where
    values' = T.indexSelect 0 idx values

-- | Filter Rows by condtion
rowFilter :: T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
rowFilter msk = rowSelect idx
  where
    idx = T.squeezeAll . T.indexSelect' 1 [0] . T.nonzero $ msk

-- | Sort Data Frame Ascending or Descending
sort :: Bool -> String -> DataFrame T.Tensor -> DataFrame T.Tensor
sort desc col df@DataFrame{..} = rowSelect idx df
  where
    dev          = T.device values
    colIdx'      = fromJust $ elemIndex col columns
    colIdx       = T.asTensor' ([colIdx'] :: [Int])
                 . T.withDevice dev . T.withDType T.Int64 $ T.defaultOpts
    idx          = T.squeezeAll . T.indexSelect 1 colIdx
                 $ T.argsort values 0 desc

-- | Drop given Rows from Data Frame
rowDrop :: T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
rowDrop idx df = rowSelect rows df
  where
    dev           = T.device $ values df
    idx'          = T.arange 0 (nRows df) 1
                  . T.withDType T.Int64 . T.withDevice dev $ T.defaultOpts
    idx''         = if not . null . T.shape $ idx
                       then T.cat (T.Dim 0) [idx', idx]
                       else idx'
    (unq, _, cnt) = T.uniqueDim 0 True False True idx''
    rows          = T.maskedSelect (T.lt cnt 2) unq

-- | Drop given Rows from Data Frame
rowDrop' :: [Int] -> DataFrame T.Tensor -> DataFrame T.Tensor
rowDrop' idx = rowDrop idx'
  where
    idx' = T.asTensor idx

-- | Row index of all Infs in Data Frame
idxInf :: DataFrame T.Tensor -> T.Tensor
idxInf DataFrame{..} = T.squeezeAll . T.indexSelect 1 idx
                     . T.nonzero . T.isinf $ values
  where
    dev = T.device values
    idx = T.asTensor' ([0] :: [Int]) 
        . T.withDevice dev . T.withDType T.Int64 $ T.defaultOpts

-- | Row index of all NaNs in Data Frame
idxNan :: DataFrame T.Tensor -> T.Tensor
idxNan DataFrame{..} = T.squeezeAll . T.indexSelect 1 idx 
                     . T.nonzero . T.isnan $ values
  where
    dev = T.device values
    idx = T.asTensor' ([0] :: [Int]) 
        . T.withDevice dev . T.withDType T.Int64 $ T.defaultOpts

-- | Drop all Rows with NaNs and Infs (just calls idxNan and rowDrop)
dropNan :: DataFrame T.Tensor -> DataFrame T.Tensor
dropNan df = rowDrop (idxNan df) df

-- | Update given columns with new values (Tensor dimensions must match)
update :: [String] -> T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
update cols vals DataFrame{..} = DataFrame columns values'
  where
    idx     = T.asTensor' (T.intIdx columns cols) 
            $ T.withDType T.Int64 T.defaultOpts
    values' = T.transpose2D $ T.indexPut False [idx] 
                (T.transpose2D vals) (T.transpose2D values)

-- | Union of two data frames
union :: DataFrame T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
union  df df' = DataFrame cols vals
  where
    cols = columns df ++ columns df'
    vals = T.cat (T.Dim 1) [values df, values df']

-- | Add columns with data
insert :: [String] -> T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
insert cols vals df = df `union` DataFrame cols vals

-- | Join 2 DataFrames, columns must line up
join :: DataFrame T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
join df df' = DataFrame columns' values'
  where
    idx      = T.intIdx (columns df) (columns df')
    vals     = T.indexSelect' 1 idx (values df')
    values'  = T.cat (T.Dim 0) [values df, vals]
    columns' = columns df

-- | Concatenate a list of Data Frames
concat :: [DataFrame T.Tensor] -> DataFrame T.Tensor
concat = foldl1 join

-- | Generate Random Index
sampleIdx :: Int -> Int -> Bool -> IO T.Tensor
sampleIdx num len rep = T.toDType T.Int64 <$> T.multinomialIO idx num rep'
  where
    rep' = rep && (len <= num)
    idx  = T.arange' 0 len 1

-- | Take n Random samples from Data Frame
sampleIO :: Int -> Bool -> DataFrame T.Tensor -> IO (DataFrame T.Tensor)
sampleIO num rep df = flip rowSelect df . T.toDevice dev <$> sampleIdx num len rep
  where 
    dev = T.device $ values df
    len = nRows df

-- | Shuffle all rows
shuffleIO :: DataFrame T.Tensor -> IO (DataFrame T.Tensor)
shuffleIO df = sampleIO (nRows df) False df

-- | Convert DataFrame to Map
asMap :: DataFrame T.Tensor -> M.Map String T.Tensor
asMap df@DataFrame{..} = M.fromList . zip columns $ map (df ??) columns

-- | Convert DataFrame to Map with native data types
asMap' :: DataFrame T.Tensor -> M.Map String [Float]
asMap' = M.map T.asValue . asMap

-- | Convert a Map to a Tensor Data Frame
fromMap :: M.Map String T.Tensor -> DataFrame T.Tensor
fromMap m = DataFrame cols vals
  where
    cols = M.keys m
    vals = T.transpose2D . T.vstack $ M.elems m

-- | Convert a Float Map to a Tensor Data Frame
fromMap' :: M.Map String [Float] -> DataFrame T.Tensor
fromMap' = fromMap . M.map T.asTensor

-- | Calculates Pearson Correlation Matrix
corr :: DataFrame T.Tensor -> T.Tensor
corr DataFrame{..} = T.corrcoef $ T.transpose2D values

-- | Split a dataframe according to a given ratio
trainTestSplit :: [String] -> [String] -> Float -> DataFrame T.Tensor
               -> (T.Tensor, T.Tensor, T.Tensor, T.Tensor)
trainTestSplit paramsX paramsY trainSize df = (trainX, validX, trainY, validY)
  where
    trainLen = round $     trainSize     * realToFrac (nRows df)
    validLen = round $ (1.0 - trainSize) * realToFrac (nRows df)
    trainIdx = T.arange 0               trainLen       1 
             $ T.withDType T.Int64 T.defaultOpts
    validIdx = T.arange trainLen (trainLen + validLen) 1 
             $ T.withDType T.Int64 T.defaultOpts
    trainX   = values . rowSelect trainIdx $ lookup paramsX df
    validX   = values . rowSelect validIdx $ lookup paramsX df
    trainY   = values . rowSelect trainIdx $ lookup paramsY df
    validY   = values . rowSelect validIdx $ lookup paramsY df
