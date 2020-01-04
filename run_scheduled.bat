python run.py ^
--model "scheduled-sampling"
--batch_per_epoch 10 ^
--epoch 1 ^
--dataset "OpenSubtitles" ^
--dataid "D:/Documents/THU/Cotk/data/iwslt14" ^
--wvid "D:\.cotk_cache\9e07c1c22c2bcbb2ba87d6d07985c0c8dcfca1427a133b062aa2f01c75b03376_unzip\glove.6B.300d.txt" ^
--decode_mode "samplek" ^
--lr 0.001
Rem --dataid "D:/.cotk_cache/9cf4d4fbf4394c0725c4ad16bf60afd4a40e64c8465bde38d038586118a54888_unzip/opensubtitles/" ^
Rem --wvclass None ^
Rem --mode "test" ^
