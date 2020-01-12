python run.py ^
--name "pg_beam" ^
--model "policy-gradient" ^
--batch_per_epoch 10 ^
--epoch 1 ^
--dataset "OpenSubtitles" ^
--dataid "D:\Documents\Tsinghua\cotk_data\iwslt14" ^
--wvid "D:\Documents\Tsinghua\cotk_data\glove.6B.300d.txt" ^
--decode_mode "samplek" ^
--lr 0.005 ^
--eh_size 175 ^
--dh_size 175 ^
--droprate 0.2 ^
--epoch 35 ^
--decode_mode "samplek" ^
--mode "test" ^
--restore "pg_samplek_best.model"
Rem --dataid "D:/.cotk_cache/9cf4d4fbf4394c0725c4ad16bf60afd4a40e64c8465bde38d038586118a54888_unzip/opensubtitles/" ^
Rem --wvclass None ^
Rem --mode "test" ^