python run.py ^
--model "scheduled-sampling"
--batch_per_epoch 10 ^
--epoch 1 ^
--dataset "OpenSubtitles" ^
--dataid "~/cotk_data/iwslt14" ^
--wvid "~/cotk_data/glove.6B.300d.txt" ^
--decode_mode "samplek" ^
--lr 0.001
Rem --dataid "D:/.cotk_cache/9cf4d4fbf4394c0725c4ad16bf60afd4a40e64c8465bde38d038586118a54888_unzip/opensubtitles/" ^
Rem --wvclass None ^
Rem --mode "test" ^
